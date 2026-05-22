"""Plot generation helpers for Lab 2."""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from wordcloud import WordCloud


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "lab2plots")
MAX_VIZ_SAMPLES = 2000


def ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def safe_filename_part(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def plot_wordcloud_corpus(texts):
    ensure_dir()
    all_text = " ".join(texts)
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            min_word_length=3,
        ).generate(all_text)
    except ValueError:
        return None

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Full Corpus")
    path = os.path.join(PLOTS_DIR, "wordcloud_corpus.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_wordcloud_per_class(texts, labels, label_names):
    ensure_dir()
    paths = []
    for class_index, class_name in enumerate(label_names):
        class_texts = [text for text, label in zip(texts, labels) if label == class_index]
        if not class_texts:
            continue

        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                min_word_length=3,
            ).generate(" ".join(class_texts))
        except ValueError:
            continue

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {class_name}")
        path = os.path.join(
            PLOTS_DIR,
            f"wordcloud_class_{safe_filename_part(class_name)}.png",
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def reduce_dim(X, method, n_components=2):
    X_dense = _prepare_dense_matrix(X, method)
    if X_dense.shape[0] < 2 or X_dense.shape[1] < 1:
        return np.zeros((X_dense.shape[0], n_components))

    if method == "pca":
        reduced = _pca(X_dense, n_components)
    elif method == "tsne":
        reduced = _tsne(X_dense, n_components)
    elif method == "svd":
        reduced = _svd(X_dense, n_components)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return _pad_components(reduced, n_components)


def _prepare_dense_matrix(X, method):
    if not issparse(X):
        return np.asarray(X)

    if method == "svd":
        return X

    max_components = min(50, X.shape[0] - 1, X.shape[1] - 1)
    if max_components < 1:
        return X.toarray()
    return TruncatedSVD(n_components=max_components, random_state=42).fit_transform(X)


def _pca(X_dense, n_components):
    max_components = min(n_components, X_dense.shape[0], X_dense.shape[1])
    if max_components < 1:
        return np.zeros((X_dense.shape[0], n_components))
    return PCA(n_components=max_components, random_state=42).fit_transform(X_dense)


def _tsne(X_dense, n_components):
    if issparse(X_dense):
        X_dense = _svd(X_dense, min(50, X_dense.shape[0] - 1, X_dense.shape[1] - 1))
    elif X_dense.shape[1] > 50:
        X_dense = _pca(X_dense, 50)

    if X_dense.shape[0] < 3:
        return _pca(X_dense, n_components)

    perplexity = min(30, X_dense.shape[0] - 1)
    return TSNE(
        n_components=n_components,
        random_state=42,
        perplexity=max(2, perplexity),
        init="random",
        learning_rate="auto",
    ).fit_transform(X_dense)


def _svd(X_dense, n_components):
    max_components = min(n_components, X_dense.shape[0] - 1, X_dense.shape[1] - 1)
    if max_components < 1:
        return np.zeros((X_dense.shape[0], n_components))
    return TruncatedSVD(n_components=max_components, random_state=42).fit_transform(X_dense)


def _pad_components(values, n_components):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] >= n_components:
        return values[:, :n_components]

    padding = np.zeros((values.shape[0], n_components - values.shape[1]))
    return np.hstack([values, padding])


def reduce_for_visualization(X, labels, method):
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    if n_samples > MAX_VIZ_SAMPLES:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_samples, MAX_VIZ_SAMPLES, replace=False)
        X = X[indices]
        labels = labels[indices]
    return reduce_dim(X, method), labels


def plot_embedding_2d(X_2d, labels, label_names, method, filename):
    ensure_dir()
    X_2d = _pad_components(X_2d, 2)
    labels = np.asarray(labels)
    if X_2d.shape[0] == 0:
        return None

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        mask = labels == label
        label_index = int(label)
        name = label_names[label_index] if label_index < len(label_names) else str(label)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.5, s=10)

    plt.legend(markerscale=3, fontsize=8)
    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion(y_true, y_pred, label_names, filename):
    ensure_dir()
    matrix = confusion_matrix(y_true, y_pred)
    n_classes = len(label_names)
    fig_width = max(8, n_classes * 0.5)
    fig_height = max(6, n_classes * 0.4)
    _, ax = plt.subplots(figsize=(fig_width, fig_height))

    display_labels = label_names if n_classes <= 25 else None
    display = ConfusionMatrixDisplay(matrix, display_labels=display_labels)
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    if n_classes > 10:
        plt.xticks(rotation=45, ha="right", fontsize=5)
        plt.yticks(fontsize=5)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_word_embeddings(keyed_vectors, words, method, filename):
    ensure_dir()
    existing_words = [word for word in words if word in keyed_vectors]
    if len(existing_words) < 2:
        return None

    vectors = np.array([keyed_vectors[word] for word in existing_words])
    if method == "pca":
        coordinates = _pca(vectors, 2)
    elif method == "tsne":
        if len(existing_words) < 3:
            return None
        coordinates = _tsne(vectors, 2)
    else:
        raise ValueError(f"Unsupported method for word embeddings: {method}")

    coordinates = _pad_components(coordinates, 2)
    plt.figure(figsize=(10, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c="steelblue", s=100)
    for index, word in enumerate(existing_words):
        plt.annotate(
            word,
            (coordinates[index, 0], coordinates[index, 1]),
            fontsize=11,
            textcoords="offset points",
            xytext=(5, 5),
        )
    plt.title(f"Word Embeddings ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
