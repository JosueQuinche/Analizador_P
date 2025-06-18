import customtkinter as ctk # type: ignore
from tkinter import filedialog, messagebox
import tkinter as tk
import pandas as pd # type: ignore
from PIL import ImageTk, Image # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
import os
import tempfile
from fpdf import FPDF # type: ignore
from app import SentimentAnalysis, ExpError

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Sentimientos")
        self.root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        self.root.configure(bg="#2c2c2c")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.model_path = "./model_7.h5"
        self.data_path = None
        self.analysis = None
        self.column_names = []
        self.positive_reviews, self.negative_reviews = [], []
        self.positive, self.negative = 0, 0
        self.positive_pct, self.negative_pct = 0, 0
        self.keyword, self.category_selected, self.keyword_pct = "", "", 0
        self.keyword_comments = []
        self.fig1 = self.figure2 = None

        self.create_widgets()

    def on_close(self):
        self.root.destroy()
        os._exit(0)

    def create_widgets(self):
        style_btn = {"corner_radius": 10, "height": 50, "font": ("Segoe UI", 18)}

        side_frame = ctk.CTkFrame(self.root, fg_color="#1f1f1f", width=220)
        side_frame.pack(side="left", fill="y")

        ctk.CTkLabel(side_frame, text="Menú", text_color="white", font=("Segoe UI", 20, "bold")).pack(pady=25)

        ctk.CTkButton(side_frame, text="POSITIVE", fg_color="#4CAF50", command=lambda: self.analyze_word("positive"), **style_btn).pack(pady=10, fill="x", padx=10)
        ctk.CTkButton(side_frame, text="NEGATIVE", fg_color="#9e0d0d", command=lambda: self.analyze_word("negative"), **style_btn).pack(pady=10, fill="x", padx=10)
        ctk.CTkButton(side_frame, text="Exportar PDF", fg_color="#2196F3", command=self.save_pdf_report, **style_btn).pack(pady=15, fill="x", padx=10)
        ctk.CTkButton(side_frame, text="Limpiar", fg_color="#9E9E9E", text_color="black", command=self.reset_app, **style_btn).pack(pady=10, fill="x", padx=10)

        main_frame = ctk.CTkFrame(self.root, fg_color="#f5f5f5")
        main_frame.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(main_frame, text="Análisis de Sentimientos", font=("Segoe UI", 24, "bold"), text_color="black").pack(pady=25)

        top_controls_frame = ctk.CTkFrame(main_frame, fg_color="#f5f5f5")
        top_controls_frame.pack(pady=10)

        ctk.CTkButton(top_controls_frame, text="Cargar CSV", fg_color="#4CAF50", command=self.load_csv, **style_btn).pack(side="left", padx=10)
        ctk.CTkButton(top_controls_frame, text="Procesar Sentimientos", fg_color="#333333", command=self.process_sentiments, **style_btn).pack(side="left", padx=10)

        self.checkboxes = []
        self.column_frame = ctk.CTkFrame(main_frame, fg_color="#f5f5f5")
        self.column_frame.pack(pady=(25, 10))

        ctk.CTkLabel(main_frame, text="Palabra clave:", font=("Segoe UI", 18, "bold"), text_color="black").pack(pady=(10, 5))
        self.entry_word = ctk.CTkEntry(main_frame, placeholder_text="Escribe una palabra clave...",
                                       font=("Segoe UI", 18), height=45, width=300, border_width=2,
                                       corner_radius=15, fg_color="white", text_color="black",
                                       border_color="black")
        self.entry_word.pack(pady=(5, 15))

        self.canvas_frame = ctk.CTkFrame(main_frame, fg_color="#f5f5f5")
        self.canvas_frame.pack(pady=20, fill="both", expand=True)

        self.text_output = tk.Text(main_frame, height=10, wrap="word", font=("Segoe UI", 16))
        self.text_output.pack(pady=20, padx=30, fill="x")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data_path = file_path
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                self.analysis = SentimentAnalysis(file_path)
                for cb in self.checkboxes:
                    cb[0].destroy()
                self.checkboxes = []
                for col in df.columns:
                    var = tk.BooleanVar()
                    cb = ctk.CTkCheckBox(self.column_frame, text=col, variable=var, text_color="black")
                    cb.pack(side="left", padx=5)
                    self.checkboxes.append((cb, var))
                self.text_output.insert(tk.END, "Archivo CSV cargado correctamente.\n")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def process_sentiments(self):
        selected = [cb.cget("text") for cb, var in self.checkboxes if var.get()]
        if not selected:
            messagebox.showwarning("Advertencia", "Debes seleccionar al menos una columna.")
            return
        try:
            self.column_names = selected
            self.positive, self.negative, pos_rw, neg_rw = self.analysis.make_predictions(
                path=self.model_path, column_names=selected, csv_file=self.data_path)
            self.positive_reviews, self.negative_reviews = pos_rw, neg_rw
            self.positive_pct, self.negative_pct = self.analysis.count_sentiments(self.positive, self.negative)

            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, f"Positivos: {self.positive_pct:.2f}% ({self.positive})\n")
            self.text_output.insert(tk.END, f"Negativos: {self.negative_pct:.2f}% ({self.negative})\n")

            self.fig1 = self.plot_bar(["Positivos", "Negativos"], [self.positive, self.negative], "Resultados de Sentimientos")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_bar(self, labels, values, title):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, values, color=["green", "red"])
        ax.set_title(title)
        ax.set_ylabel("Cantidad")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        return fig

    def analyze_word(self, mode):
        word = self.entry_word.get().strip()
        if not word:
            messagebox.showwarning("Advertencia", "Ingresa una palabra.")
            return
        try:
            self.keyword = word
            self.category_selected = mode
            insight, reviews = self.analysis.insights(
                self.positive_reviews, self.negative_reviews, word, mode=mode)
            self.keyword_pct, self.keyword_comments = insight, reviews

            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, f"Palabra: '{word}' en comentarios {mode}\n")
            self.text_output.insert(tk.END, f"Coincidencias: {insight:.2f}%\n\n")

            for i, r in enumerate(reviews[:5], start=1):
                try:
                    parts = r.split("Usuario_")
                    comentario = parts[0].strip()
                    user_info = "Usuario_" + parts[1] if len(parts) > 1 else "N/A"
                    user_parts = user_info.split(" ", 1)
                    usuario = user_parts[0]
                    fecha = user_parts[1] if len(user_parts) > 1 else "Fecha no disponible"

                    self.text_output.insert(tk.END, f"{i}. Comentario: {comentario}\n")
                    self.text_output.insert(tk.END, f"   Usuario: {usuario}\n")
                    self.text_output.insert(tk.END, f"   Fecha: {fecha}\n\n")
                except Exception:
                    self.text_output.insert(tk.END, f"{i}. {r}\n\n")

            self.figure2 = self.plot_bar([f"{mode.capitalize()} con '{word}'", "Otros"],
                                         [insight, 100 - insight], f"Frecuencia de '{word}'")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clean_text(self, text):
        replacements = {'’': "'", '“': '"', '”': '"', '\n': ' ', '\r': ' '}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def save_pdf_report(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, self.clean_text("Reporte de Análisis de Sentimientos"), ln=True, align="C")
        pdf.ln(10)
        pdf.cell(0, 10, f"Positivos: {self.positive_pct:.2f}% ({self.positive})", ln=True)
        pdf.cell(0, 10, f"Negativos: {self.negative_pct:.2f}% ({self.negative})", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            if self.fig1:
                self.fig1.savefig(tmpfile.name)
                pdf.image(tmpfile.name, x=10, w=180)

        if self.keyword:
            pdf.ln(10)
            pdf.cell(0, 10, f"Palabra clave: '{self.keyword}' en comentarios {self.category_selected}", ln=True)
            pdf.cell(0, 10, f"Porcentaje: {self.keyword_pct:.2f}%", ln=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
                if self.figure2:
                    self.figure2.savefig(tmpfile2.name)
                    pdf.image(tmpfile2.name, x=10, w=180)

            pdf.ln(10)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 10, "Comentarios:")
            for i, comment in enumerate(self.keyword_comments[:10], start=1):
                try:
                    parts = comment.split("Usuario_")
                    comentario = self.clean_text(parts[0].strip())
                    user_info = "Usuario_" + parts[1] if len(parts) > 1 else "N/A"
                    user_parts = user_info.split(" ", 1)
                    usuario = user_parts[0]
                    fecha = user_parts[1] if len(user_parts) > 1 else "Fecha no disponible"
                    pdf.multi_cell(0, 10, f"{i}. Comentario: {comentario}\n   Usuario: {usuario}\n   Fecha: {fecha}\n")
                except Exception:
                    pdf.multi_cell(0, 10, f"{i}. {self.clean_text(comment)}\n")

        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if save_path:
            pdf.output(save_path)
            messagebox.showinfo("Éxito", f"Reporte guardado en: {save_path}")

    def reset_app(self):
        self.data_path = None
        self.analysis = None
        self.column_names = []
        self.positive_reviews, self.negative_reviews = [], []
        self.positive, self.negative = 0, 0
        self.positive_pct, self.negative_pct = 0, 0
        self.keyword, self.category_selected, self.keyword_pct = "", "", 0
        self.keyword_comments = []
        self.fig1 = self.figure2 = None
        for widget in self.column_frame.winfo_children():
            widget.destroy()
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        self.checkboxes = []
        self.entry_word.delete(0, tk.END)
        self.text_output.delete(1.0, tk.END)

if __name__ == "__main__":
    root = ctk.CTk()
    app = SentimentApp(root)
    root.mainloop()
