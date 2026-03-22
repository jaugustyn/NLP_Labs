import numpy as np
from sklearn.datasets import fetch_20newsgroups

MAX_SAMPLES = 5000

DATASET_QUERY_WORDS = {
    "20news_group": ["space", "computer", "science", "music", "car"],
    "imdb": ["movie", "actor", "great", "terrible", "plot"],
    "amazon": ["product", "quality", "price", "recommend", "terrible"],
    "ag_news": ["politics", "sports", "technology", "business", "science"],
}

AVAILABLE_DATASETS = ["20news_group", "imdb", "amazon", "ag_news"]


def load_dataset(name, max_samples=MAX_SAMPLES):
    """Load a dataset by name.

    Returns:
        texts: list[str]
        labels: np.ndarray of int
        label_names: list[str]
    """
    loaders = {
        "20news_group": _load_20news,
        "imdb": _load_imdb,
        "amazon": _load_amazon,
        "ag_news": _load_ag_news,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {AVAILABLE_DATASETS}")

    texts, labels, label_names = loader()
    labels = np.array(labels)

    # Subsample large datasets
    if max_samples and len(texts) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = labels[indices]

    # Filter out empty texts
    valid = [(t, l) for t, l in zip(texts, labels) if t and t.strip()]
    if valid:
        texts, labels = zip(*valid)
        texts = list(texts)
        labels = np.array(labels)

    return texts, labels, label_names


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _load_20news():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return list(data.data), data.target.tolist(), list(data.target_names)


def _load_imdb():
    from datasets import load_dataset as hf_load

    ds = hf_load("imdb")
    texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    return texts, labels, ["negative", "positive"]


def _load_amazon():
    from datasets import load_dataset as hf_load

    ds = hf_load("amazon_polarity", split="test")
    texts = [f"{t} {c}" for t, c in zip(ds["title"], ds["content"])]
    labels = list(ds["label"])
    return texts, labels, ["negative", "positive"]


def _load_ag_news():
    from datasets import load_dataset as hf_load

    ds = hf_load("ag_news")
    texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    return texts, labels, ["World", "Sports", "Business", "Sci/Tech"]
