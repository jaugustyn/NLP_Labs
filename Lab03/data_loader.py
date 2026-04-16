import os
import re
import csv
import numpy as np
import pandas as pd

from config import CUSTOM_DATASET_FILE, CUSTOM_LABELS, LABEL_ALIASES, MAX_SAMPLES


def load_dataset(name, max_samples=MAX_SAMPLES):
    """Load a dataset by name. Returns (texts, labels, label_names)."""
    loaders = {
        "amazon": _load_amazon,
        "imdb": _load_imdb,
        "custom": _load_custom,
    }
    loader = loaders.get(name)
    if not loader:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

    texts, labels, label_names = loader()
    labels = np.array(labels)

    if max_samples and len(texts) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in idx]
        labels = labels[idx]

    valid = [(t, l) for t, l in zip(texts, labels) if t and t.strip()]
    if valid:
        texts, labels = zip(*valid)
        texts = list(texts)
        labels = np.array(labels)

    return texts, labels, label_names


def _load_imdb():
    from datasets import load_dataset as hf_load

    ds = hf_load("imdb")
    texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    texts = [re.sub(r"<[^>]+>", " ", t) for t in texts]
    labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    return texts, labels, ["negative", "positive"]


def _load_amazon():
    from datasets import load_dataset as hf_load

    ds = hf_load("amazon_polarity", split="test")
    texts = [f"{t} {c}" for t, c in zip(ds["title"], ds["content"])]
    labels = list(ds["label"])
    return texts, labels, ["negative", "positive"]


def _load_custom():
    if not os.path.exists(CUSTOM_DATASET_FILE):
        raise FileNotFoundError(
            f"Custom dataset not found: {CUSTOM_DATASET_FILE}\n"
            "Use /add_sentiment to add data first."
        )
    df = pd.read_csv(CUSTOM_DATASET_FILE)
    if df.empty:
        raise ValueError("Custom dataset is empty. Use /add_sentiment to add data.")

    label_names = sorted(df["label"].unique().tolist())
    label_map = {name: i for i, name in enumerate(label_names)}
    texts = df["text"].tolist()
    labels = [label_map[l] for l in df["label"]]
    return texts, labels, label_names


def add_record(text, label):
    """Add a record to sentiment_dataset.csv. Returns normalized label."""
    normalized = LABEL_ALIASES.get(label.strip().lower())
    if not normalized:
        raise ValueError(
            f"Unknown label: '{label}'. Allowed: {', '.join(CUSTOM_LABELS)}"
        )
    file_exists = os.path.exists(CUSTOM_DATASET_FILE)
    if file_exists:
        with open(CUSTOM_DATASET_FILE, "rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    else:
        needs_newline = False
    with open(CUSTOM_DATASET_FILE, "a", newline="", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["text", "label"])
        writer.writerow([text, normalized])
    return normalized


def get_custom_stats():
    """Get stats for custom dataset. Returns None if file doesn't exist."""
    if not os.path.exists(CUSTOM_DATASET_FILE):
        return None
    df = pd.read_csv(CUSTOM_DATASET_FILE)
    return {"total": len(df), "classes": df["label"].value_counts().to_dict()}
