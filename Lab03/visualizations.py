import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import PLOTS_DIR


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_training_history(history, model_type, dataset_name):
    """Plot accuracy and loss curves from Keras training history."""
    _ensure_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["accuracy"], label="Train Accuracy")
    ax1.plot(history["val_accuracy"], label="Val Accuracy")
    ax1.set_title(f"{model_type.upper()} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title(f"{model_type.upper()} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Training History: {model_type.upper()} on {dataset_name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"train_history_{model_type}_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, label_names, method, dataset_name):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    _ensure_dir()
    cm = confusion_matrix(y_true, y_pred)
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {method} on {dataset_name}")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_{method}_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_comparison(results_df, dataset_name):
    """Bar chart comparing methods by accuracy and macro_f1."""
    _ensure_dir()
    df = results_df[results_df["dataset"] == dataset_name].copy()
    if df.empty:
        return None

    metrics = ["accuracy", "macro_f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = df.sort_values(metric, ascending=False)
        colors = sns.color_palette("viridis", len(data))
        bars = ax.bar(data["method"], data[metric], color=colors)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, data[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Method Comparison — {dataset_name}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"compare_methods_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_wordcloud(texts, label, dataset_name):
    from wordcloud import WordCloud

    _ensure_dir()
    all_text = " ".join(texts)
    if not all_text.strip():
        return None
    try:
        wc = WordCloud(
            width=800, height=400, background_color="white", min_word_length=3,
        ).generate(all_text)
    except ValueError:
        return None

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {label} ({dataset_name})")
    safe = label.replace(" ", "_").replace("/", "_")
    path = os.path.join(PLOTS_DIR, f"wordcloud_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_class_distribution(label_counts, dataset_name):
    _ensure_dir()
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("pastel", len(labels))
    bars = plt.bar(labels, counts, color=colors)
    for bar, c in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(c), ha="center", va="bottom")
    plt.title(f"Class Distribution — {dataset_name}")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"class_distribution_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
