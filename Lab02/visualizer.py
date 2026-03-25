import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.sparse import issparse
from wordcloud import WordCloud
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PLOTS_DIR = "lab2plots"
MAX_VIZ_SAMPLES = 2000

def ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_wordcloud_corpus(texts):
    ensure_dir()
    all_text = " ".join(texts)
    try:
        wc = WordCloud(width=800, height=400, background_color="white",
                       min_word_length=3).generate(all_text)
    except ValueError:
        return None
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Full Corpus")
    path = os.path.join(PLOTS_DIR, "wordcloud_corpus.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

def plot_wordcloud_per_class(texts, labels, label_names):
    ensure_dir()
    paths = []
    for cls_idx, cls_name in enumerate(label_names):
        cls_texts = [t for t, l in zip(texts, labels) if l == cls_idx]
        if not cls_texts:
            continue
        all_text = " ".join(cls_texts)
        try:
            wc = WordCloud(width=800, height=400, background_color="white",
                           min_word_length=3).generate(all_text)
        except ValueError:
            continue
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {cls_name}")
        safe_name = cls_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(PLOTS_DIR, f"wordcloud_class_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def reduce_dim(X, method, n_components=2):
    if issparse(X):
        if method == "svd":
            return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(X)
        n_pre = min(50, X.shape[1] - 1)
        X_dense = TruncatedSVD(n_components=n_pre, random_state=42).fit_transform(X)
    else:
        X_dense = np.asarray(X)

    if method == "pca":
        n = min(n_components, X_dense.shape[1], X_dense.shape[0])
        return PCA(n_components=n, random_state=42).fit_transform(X_dense)
    elif method == "tsne":
        if X_dense.shape[1] > 50:
            X_dense = PCA(n_components=50, random_state=42).fit_transform(X_dense)
        perp = min(30, X_dense.shape[0] - 1)
        return TSNE(n_components=n_components, random_state=42, perplexity=max(2, perp)).fit_transform(X_dense)
    elif method == "svd":
        return TruncatedSVD(n_components=n_components, random_state=42).fit_transform(X_dense)
    raise ValueError(f"Unknown reduction method: {method}")


def reduce_for_visualization(X, labels, method):
    n = X.shape[0]
    if n > MAX_VIZ_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, MAX_VIZ_SAMPLES, replace=False)
        X = X[idx]
        labels = labels[idx]
    X_2d = reduce_dim(X, method)
    return X_2d, labels

def plot_embedding_visualization(X, labels, label_names, method, filename):
    ensure_dir()
    X_2d, labels = reduce_for_visualization(X, labels, method)
    return plot_embedding_2d(X_2d, labels, label_names, method, filename)

def plot_embedding_2d(X_2d, labels, label_names, method, filename):
    ensure_dir()
    plt.figure(figsize=(10, 8))
    unique = np.unique(labels)
    for lbl in unique:
        mask = labels == lbl
        name = label_names[lbl] if lbl < len(label_names) else str(lbl)
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
    cm = confusion_matrix(y_true, y_pred)

    n_classes = len(label_names)
    fig_w = max(8, n_classes * 0.5)
    fig_h = max(6, n_classes * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    display_labels = label_names if n_classes <= 25 else None
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
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
    existing = [w for w in words if w in keyed_vectors]
    if len(existing) < 2:
        return None

    vectors = np.array([keyed_vectors[w] for w in existing])

    if method == "pca":
        n = min(2, vectors.shape[0], vectors.shape[1])
        reducer = PCA(n_components=n)
    elif method == "tsne":
        perp = max(2, min(5, len(existing) - 1))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perp)
    else:
        raise ValueError(f"Unsupported method for word embeddings: {method}")

    coords = reducer.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c="steelblue", s=100)
    for i, word in enumerate(existing):
        plt.annotate(word, (coords[i, 0], coords[i, 1]),
                     fontsize=11, textcoords="offset points", xytext=(5, 5))
    plt.title(f"Word Embeddings ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
