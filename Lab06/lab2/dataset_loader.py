"""Dataset loading helpers for Lab 2 experiments."""

import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups


MAX_SAMPLES = 5000

DATASET_QUERY_WORDS = {
    "20news_group": ["space", "computer", "science", "music", "car"],
    "imdb": ["movie", "actor", "great", "terrible", "plot"],
    "amazon": ["product", "quality", "price", "recommend", "terrible"],
    "ag_news": ["politics", "sports", "technology", "business", "science"],
}

AVAILABLE_DATASETS = ("20news_group", "imdb", "amazon", "ag_news")


def load_dataset(name, max_samples=MAX_SAMPLES):
    loaders = {
        "20news_group": load_20news,
        "imdb": load_imdb,
        "amazon": load_amazon,
        "ag_news": load_ag_news,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {AVAILABLE_DATASETS}")

    texts, labels, label_names = loader()
    labels = np.array(labels)
    texts, labels = _drop_empty_texts(texts, labels)

    if max_samples and len(texts) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = labels[indices]

    return texts, labels, label_names


def _drop_empty_texts(texts, labels):
    valid = [(text, label) for text, label in zip(texts, labels) if text and text.strip()]
    if not valid:
        return [], np.array([])

    valid_texts, valid_labels = zip(*valid)
    return list(valid_texts), np.array(valid_labels)


def load_20news():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return list(data.data), data.target.tolist(), list(data.target_names)


def load_imdb():
    from datasets import load_dataset as hf_load

    dataset = hf_load("imdb")
    texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
    texts = [re.sub(r"<[^>]+>", " ", text) for text in texts]
    labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"])
    return texts, labels, ["negative", "positive"]


def load_amazon():
    from datasets import load_dataset as hf_load

    dataset = hf_load("amazon_polarity", split="test")
    texts = [
        f"{title} {content}"
        for title, content in zip(dataset["title"], dataset["content"])
    ]
    labels = list(dataset["label"])
    return texts, labels, ["negative", "positive"]


def load_ag_news():
    from datasets import load_dataset as hf_load

    dataset = hf_load("ag_news")
    texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
    labels = list(dataset["train"]["label"]) + list(dataset["test"]["label"])
    return texts, labels, ["World", "Sports", "Business", "Sci/Tech"]
