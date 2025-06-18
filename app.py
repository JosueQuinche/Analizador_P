import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class ExpError(Exception):
    pass

class SentimentAnalysis:
    def __init__(self, file, review_columns=None):
        self.file = file
        self.review_columns = review_columns or []

    def load_model(self, path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            raise ExpError(f"Error loading model: {e}")

    def load_data(self, path):
        try:
            return pd.read_csv(path, encoding='utf-8')
        except Exception as e:
            raise ExpError(f"Error loading data: {e}")

    def make_predictions(self, path, column_names, csv_file=None):
        model = self.load_model(path)
        data = self.load_data(csv_file)

        reviews = data[column_names].astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

        with open("./tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)

        review_sequences = tokenizer.texts_to_sequences(reviews)
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(review_sequences, maxlen=132, truncating='post', padding='post')

        predictions = model.predict(pad_sequences)

        positive_reviews, negative_reviews = [], []
        positive, negative = 0, 0

        for i, review in enumerate(reviews):
            if predictions[i] > 0.5:
                positive_reviews.append(review)
                positive += 1
            else:
                negative_reviews.append(review)
                negative += 1

        return positive, negative, positive_reviews, negative_reviews

    def count_sentiments(self, positive, negative):
        total = positive + negative
        if total == 0:
            raise ExpError("No reviews to analyze.")
        return (positive / total) * 100, (negative / total) * 100

    def insights(self, positive_reviews, negative_reviews, word, mode="positive"):
        word = word.lower()
        if mode == "positive":
            filtered = [r for r in positive_reviews if word in r.lower()]
            return (len(filtered) / len(positive_reviews)) * 100, filtered
        elif mode == "negative":
            filtered = [r for r in negative_reviews if word in r.lower()]
            return (len(filtered) / len(negative_reviews)) * 100, filtered
        else:
            raise ExpError("Invalid mode. Use 'positive' or 'negative'.")

    def plot_results(self, positive, negative):
        labels = ["Positive", "Negative"]
        counts = [positive, negative]
        plt.bar(labels, counts, color=["blue", "red"])
        plt.title("Sentiment Analysis Results")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.show()
