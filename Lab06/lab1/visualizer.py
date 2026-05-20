"""Plot generation helpers for Lab 1."""

import os
from collections import Counter
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(_DIR, "plots")


def generate_filename():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    base_time = datetime.now()
    for offset_seconds in range(0, 60):
        timestamp = (base_time + timedelta(seconds=offset_seconds)).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        candidate = os.path.join(PLOTS_DIR, f"Sentence_{timestamp}.png")
        if not os.path.exists(candidate):
            return candidate

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(PLOTS_DIR, f"Sentence_{timestamp}.png")


def plot_token_length_histogram(tokens):
    if not tokens:
        return None

    lengths = [len(t) for t in tokens]

    plt.figure(figsize=(8, 5))
    plt.hist(
        lengths,
        bins=range(1, max(lengths) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    plt.title("Token Length Histogram")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")

    filepath = generate_filename()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    return filepath


def plot_wordcloud(text):
    filepath = generate_filename()
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
        ).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
        return filepath
    except ValueError:
        return None


def plot_most_common_words(tokens):
    counts = Counter(tokens).most_common(10)
    if not counts:
        return None

    words, freqs = zip(*counts)

    plt.figure(figsize=(10, 6))
    plt.bar(words, freqs, color="skyblue")
    plt.title("Most Common Tokens")
    plt.xlabel("Tokens")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    filepath = generate_filename()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    return filepath
