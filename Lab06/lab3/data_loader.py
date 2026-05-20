"""Dataset loading and custom sentiment dataset helpers for Lab 3."""

import csv
import os
import re

import numpy as np
import pandas as pd

from lab3.config import CUSTOM_DATASET_FILE, CUSTOM_LABELS, LABEL_ALIASES, MAX_SAMPLES


def load_dataset(name, max_samples=MAX_SAMPLES):
    loaders = {
        "amazon": _load_amazon,
        "imdb": _load_imdb,
        "custom": _load_custom,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders)}")

    texts, labels, label_names = loader()
    labels = np.array(labels)
    texts, labels = _drop_empty_texts(texts, labels)

    if max_samples and len(texts) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[index] for index in indices]
        labels = labels[indices]

    return texts, labels, label_names


def _drop_empty_texts(texts, labels):
    valid = [
        (text, label)
        for text, label in zip(texts, labels)
        if isinstance(text, str) and text.strip()
    ]
    if not valid:
        return [], np.array([])

    valid_texts, valid_labels = zip(*valid)
    return list(valid_texts), np.array(valid_labels)


def _load_imdb():
    from datasets import load_dataset as hf_load

    dataset = hf_load("imdb")
    texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
    texts = [re.sub(r"<[^>]+>", " ", text) for text in texts]
    labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"])
    return texts, labels, ["negative", "positive"]


def _load_amazon():
    from datasets import load_dataset as hf_load

    dataset = hf_load("amazon_polarity", split="test")
    texts = [
        f"{title} {content}"
        for title, content in zip(dataset["title"], dataset["content"])
    ]
    labels = list(dataset["label"])
    return texts, labels, ["negative", "positive"]


def _load_custom():
    if not os.path.exists(CUSTOM_DATASET_FILE):
        raise FileNotFoundError(
            f"Custom dataset not found: {CUSTOM_DATASET_FILE}\n"
            "Use /add_sentiment to add data first."
        )

    dataframe = pd.read_csv(CUSTOM_DATASET_FILE)
    _validate_custom_dataframe(dataframe)
    dataframe = dataframe.dropna(subset=["text", "label"])
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe["label"] = dataframe["label"].astype(str).str.strip().str.lower()
    dataframe["label"] = dataframe["label"].map(
        lambda label: LABEL_ALIASES.get(label, label)
    )
    dataframe = dataframe[dataframe["text"] != ""]

    if dataframe.empty:
        raise ValueError("Custom dataset is empty. Use /add_sentiment to add data.")

    invalid_labels = sorted(set(dataframe["label"]) - set(CUSTOM_LABELS))
    if invalid_labels:
        raise ValueError(
            "Custom dataset contains invalid labels: "
            f"{', '.join(invalid_labels)}. Allowed: {', '.join(CUSTOM_LABELS)}"
        )

    label_names = list(CUSTOM_LABELS)
    label_map = {label: index for index, label in enumerate(label_names)}
    texts = dataframe["text"].tolist()
    labels = [label_map[label] for label in dataframe["label"]]
    return texts, labels, label_names


def _validate_custom_dataframe(dataframe):
    expected = {"text", "label"}
    if not expected.issubset(dataframe.columns):
        raise ValueError("Custom dataset must contain text and label columns.")


def add_record(text, label):
    normalized = LABEL_ALIASES.get(label.strip().lower())
    if not normalized:
        raise ValueError(
            f"Unknown label: '{label}'. Allowed: {', '.join(CUSTOM_LABELS)}"
        )

    os.makedirs(os.path.dirname(CUSTOM_DATASET_FILE), exist_ok=True)
    file_exists = os.path.exists(CUSTOM_DATASET_FILE)
    needs_header = not file_exists or os.path.getsize(CUSTOM_DATASET_FILE) == 0
    needs_newline = False
    if file_exists and os.path.getsize(CUSTOM_DATASET_FILE) > 0:
        with open(CUSTOM_DATASET_FILE, "rb") as file:
            file.seek(-1, os.SEEK_END)
            needs_newline = file.read(1) != b"\n"

    with open(CUSTOM_DATASET_FILE, "a", newline="", encoding="utf-8") as file:
        if needs_newline:
            file.write("\n")
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if needs_header:
            writer.writerow(["text", "label"])
        writer.writerow([text.strip(), normalized])
    return normalized


def get_custom_stats():
    if not os.path.exists(CUSTOM_DATASET_FILE):
        return None

    dataframe = pd.read_csv(CUSTOM_DATASET_FILE)
    if dataframe.empty or "label" not in dataframe:
        return {"total": 0, "classes": {}}
    return {
        "total": len(dataframe),
        "classes": dataframe["label"].value_counts().to_dict(),
    }
