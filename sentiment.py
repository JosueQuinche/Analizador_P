from app import ExpError, SentimentAnalysis
import os

os.system("cls")
print("ANÁLISIS DE SENTIMIENTOS CON ARCHIVOS CSV")
print("--------------------------------------------------")
print()
data = input("Ingresa la ruta del archivo CSV: ")
print("--------------------------------------------------")
print()
values = input("Ingresa el nombre de la columna que contiene los comentarios: ")
print("--------------------------------------------------")
print()

sentiment = SentimentAnalysis(data, values)
model = sentiment.load_model("./model_7.h5")
load_data = sentiment.load_data(data)

positive, negative, positive_reviews, negative_reviews = sentiment.make_predictions("./model_7.h5", values, data)
positive_percentage, negative_percentage = sentiment.count_sentiments(positive, negative)
sentiment.plot_results(positive, negative)

os.system("cls")
print(f"El porcentaje de comentarios positivos es: {positive_percentage:.2f}% ({positive})")
print(f"El porcentaje de comentarios negativos es: {negative_percentage:.2f}% ({negative})")
print("--------------------------------------------------")
print()

mode = input("En qué categoría deseas analizar los comentarios?: ").lower().replace(" ", "")
print("--------------------------------------------------")
print()
word_insight = input("Ingresa la palabra que deseas analizar: ").lower().replace(" ", "")
print("--------------------------------------------------")

insight, review = sentiment.insights(positive_reviews, negative_reviews, word_insight, mode=mode)

print(f"El porcentaje de comentarios {mode} que contienen la palabra '{word_insight}' es: {insight:.2f}%")
print("--------------------------------------------------")
print()
print("Comentarios:")
for i, r in enumerate(review):
    print(f"{i + 1}. {r}")