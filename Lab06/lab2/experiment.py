"""Experiment runner for Lab 2 text classification."""

import csv
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from lab2 import visualizer as viz
from lab2.dataset_loader import DATASET_QUERY_WORDS, load_dataset
from lab2.models import get_grid_params, get_model, resolve_methods
from lab2.text_embeddings import EMBEDDING_NAMES, EmbeddingUnavailable, get_embedding


SEEDS = (42, 1337, 2024)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "lab2results.csv")
SIMILAR_WORDS_FILE = os.path.join(RESULTS_DIR, "lab2_similar_words.txt")
FEATURE_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, "lab2_feature_importance.txt")
REDUCTION_METHODS = ("pca", "tsne", "svd")


def run_experiment(
    dataset_name,
    method_str,
    gridsearch,
    n_runs,
    progress_callback=None,
):
    def progress(message):
        if progress_callback:
            try:
                progress_callback(message)
            except Exception:
                pass
        print(message)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = SEEDS[:n_runs]
    method_names = resolve_methods(method_str)

    progress(f"Loading dataset '{dataset_name}'...")
    texts, labels, label_names = load_dataset(dataset_name)
    labels = np.asarray(labels)
    if len(texts) < 2:
        raise ValueError("Dataset must contain at least two non-empty texts.")

    progress(f"Loaded {len(texts)} samples, {len(label_names)} classes")

    progress("Generating word clouds...")
    viz.plot_wordcloud_corpus(texts)
    viz.plot_wordcloud_per_class(texts, labels, label_names)
    progress("Word clouds done.")

    results = []
    skipped_embeddings = []
    feature_importance_lines = []
    word_vector_models = {}

    for embedding_name in EMBEDDING_NAMES:
        progress(f"Preparing '{embedding_name}' representation...")
        preview_embedding = _fit_embedding(embedding_name, texts, progress)
        if preview_embedding is None:
            skipped_embeddings.append(embedding_name)
            continue

        X_preview = preview_embedding.transform(texts)
        _plot_embedding_reductions(
            X_preview,
            labels,
            label_names,
            dataset_name,
            method_names,
            embedding_name,
            seeds[-1],
            progress,
        )

        if embedding_name in ("word2vec", "glove"):
            word_vector_models[embedding_name] = preview_embedding

        for model_name in method_names:
            progress(f"Training {model_name} on {embedding_name}...")
            run_metrics = []
            last_y_true = None
            last_y_pred = None
            last_model = None
            last_embedding = None

            for seed in seeds:
                train_idx, test_idx = _split_indices(labels, seed)
                train_texts = [texts[index] for index in train_idx]
                test_texts = [texts[index] for index in test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                embedding = _fit_embedding(embedding_name, train_texts, progress)
                if embedding is None:
                    skipped_embeddings.append(embedding_name)
                    break

                X_train = embedding.transform(train_texts)
                X_test = embedding.transform(test_texts)
                model = _fit_model(
                    model_name,
                    embedding_name,
                    X_train,
                    y_train,
                    gridsearch,
                    seed,
                    progress,
                )

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

                results.append(
                    {
                        "embedding": embedding_name,
                        "model": model_name,
                        "accuracy": round(accuracy, 4),
                        "macro_f1": round(macro_f1, 4),
                        "seed": seed,
                    }
                )
                run_metrics.append({"accuracy": accuracy, "macro_f1": macro_f1})

                last_y_true = y_test
                last_y_pred = y_pred
                last_model = model
                last_embedding = embedding

            if not run_metrics:
                continue

            avg_accuracy = np.mean([row["accuracy"] for row in run_metrics])
            avg_macro_f1 = np.mean([row["macro_f1"] for row in run_metrics])
            progress(
                f"  -> avg accuracy={avg_accuracy:.4f} "
                f"macro_f1={avg_macro_f1:.4f}"
            )

            viz.plot_confusion(
                last_y_true,
                last_y_pred,
                label_names,
                f"confusion_{embedding_name}_{model_name}.png",
            )

            feature_importance = extract_feature_importance(
                last_model,
                last_embedding,
                model_name,
                embedding_name,
                label_names,
            )
            if feature_importance:
                feature_importance_lines.append(feature_importance)

    if word_vector_models:
        save_similar_words(word_vector_models, dataset_name)
        progress("Similar words saved.")
        _plot_word_vector_examples(word_vector_models, dataset_name)
        progress("Word embedding visualisations done.")

    save_results_csv(results)
    progress(f"Results saved to {_relative_path(RESULTS_FILE)}")

    if feature_importance_lines:
        save_feature_importance_file(feature_importance_lines)
        progress(f"Feature importance saved to {_relative_path(FEATURE_IMPORTANCE_FILE)}")

    return format_summary(results, skipped_embeddings)


def _fit_embedding(embedding_name, texts, progress):
    try:
        embedding = get_embedding(embedding_name)
        embedding.fit(texts)
        return embedding
    except EmbeddingUnavailable as exc:
        progress(f"  Skipping {embedding_name}: {exc}")
    except Exception as exc:
        progress(f"  Skipping {embedding_name}: {type(exc).__name__}: {exc}")
    return None


def _split_indices(labels, seed):
    indices = np.arange(len(labels))
    stratify = _stratify_or_none(labels)
    return train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )


def _stratify_or_none(labels):
    unique, counts = np.unique(labels, return_counts=True)
    expected_test_size = int(np.ceil(len(labels) * 0.2))
    if len(unique) < 2 or counts.min() < 2 or expected_test_size < len(unique):
        return None
    return labels


def _fit_model(model_name, embedding_name, X_train, y_train, gridsearch, seed, progress):
    model = get_model(model_name, embedding_name, seed)
    if not gridsearch:
        model.fit(X_train, y_train)
        return model

    params = get_grid_params(model_name, embedding_name)
    if not params:
        model.fit(X_train, y_train)
        return model

    _, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    cv = min(3, min_class_count)
    if cv < 2:
        progress("  GridSearch skipped: too few samples per class.")
        model.fit(X_train, y_train)
        return model

    grid = GridSearchCV(
        model,
        params,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score="raise",
    )
    grid.fit(X_train, y_train)
    progress(f"  GridSearch best: {grid.best_params_}")
    return grid.best_estimator_


def _plot_embedding_reductions(
    X_all,
    labels,
    label_names,
    dataset_name,
    method_names,
    embedding_name,
    seed,
    progress,
):
    _, test_idx = _split_indices(labels, seed)
    X_test = X_all[test_idx]
    y_test = labels[test_idx]

    for reduction_name in REDUCTION_METHODS:
        progress(f"  Reducing {embedding_name} with {reduction_name.upper()}...")
        X_2d, y_sub = viz.reduce_for_visualization(X_test, y_test, reduction_name)
        for model_name in method_names:
            filename = (
                f"{dataset_name}_{model_name}_{embedding_name}_"
                f"{reduction_name}_embedding.png"
            )
            viz.plot_embedding_2d(X_2d, y_sub, label_names, reduction_name, filename)
    progress(f"  Embedding visualisations for {embedding_name} done.")


def _plot_word_vector_examples(word_vector_models, dataset_name):
    query_words = DATASET_QUERY_WORDS.get(dataset_name, ["good", "bad", "great"])
    wrote_generic_files = False
    for embedding_name, embedding in word_vector_models.items():
        keyed_vectors = embedding.get_word_vectors()
        if keyed_vectors is None:
            continue

        expanded_words = list(query_words)
        for word in query_words:
            if word in keyed_vectors:
                expanded_words.extend(
                    similar_word
                    for similar_word, _ in keyed_vectors.most_similar(word, topn=3)
                )
        expanded_words = list(dict.fromkeys(expanded_words))
        viz.plot_word_embeddings(
            keyed_vectors,
            expanded_words,
            "pca",
            f"{embedding_name}_word_embedding_pca.png",
        )
        viz.plot_word_embeddings(
            keyed_vectors,
            expanded_words,
            "tsne",
            f"{embedding_name}_word_embedding_tsne.png",
        )

        if not wrote_generic_files:
            viz.plot_word_embeddings(
                keyed_vectors,
                expanded_words,
                "pca",
                "word_embedding_pca.png",
            )
            viz.plot_word_embeddings(
                keyed_vectors,
                expanded_words,
                "tsne",
                "word_embedding_tsne.png",
            )
            wrote_generic_files = True


def save_results_csv(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["embedding", "model", "accuracy", "macro_f1", "seed"],
        )
        writer.writeheader()
        writer.writerows(results)


def format_summary(results, skipped_embeddings=None):
    skipped_embeddings = list(dict.fromkeys(skipped_embeddings or []))
    lines = ["=== EXPERIMENT RESULTS ===\n"]

    if not results:
        lines.append("No experiment finished successfully.")
    else:
        combos = {}
        for row in results:
            key = (row["embedding"], row["model"])
            combos.setdefault(key, []).append(row)

        for (embedding_name, model_name), rows in combos.items():
            avg_accuracy = np.mean([row["accuracy"] for row in rows])
            avg_macro_f1 = np.mean([row["macro_f1"] for row in rows])
            lines.append(
                f"{embedding_name:>10} + {model_name:<8}  "
                f"accuracy={avg_accuracy:.4f}  "
                f"macro_f1={avg_macro_f1:.4f}  "
                f"(runs={len(rows)})"
            )

    if skipped_embeddings:
        lines.append("\nSkipped embeddings: " + ", ".join(skipped_embeddings))

    lines.append(f"\nTotal experiments: {len(results)}")
    lines.append(f"Results CSV: {_relative_path(RESULTS_FILE)}")
    lines.append(f"Plots directory: {_relative_path(viz.PLOTS_DIR)}/")
    return "\n".join(lines)


def extract_feature_importance(model, embedding, model_name, embedding_name, label_names):
    if model is None or embedding is None:
        return None

    feature_names = embedding.get_feature_names()
    if feature_names is None:
        return None

    feature_names = list(feature_names)
    lines = [f"\n=== {embedding_name} + {model_name} ==="]

    try:
        if model_name == "logreg" and hasattr(model, "coef_"):
            _append_logreg_features(lines, model.coef_, feature_names, label_names)
        elif model_name == "nb":
            _append_nb_features(lines, model, feature_names, label_names)
        elif model_name == "rf" and hasattr(model, "feature_importances_"):
            _append_rf_features(lines, model.feature_importances_, feature_names)
        else:
            return None
    except Exception:
        return None

    return "\n".join(lines)


def _append_logreg_features(lines, coef, feature_names, label_names):
    if coef.shape[0] == 1:
        lines.append("Top features (global):")
        for index in np.argsort(np.abs(coef[0]))[-10:][::-1]:
            lines.append(f"  {feature_names[index]:>20}  {coef[0][index]:+.4f}")
        return

    for class_index, class_name in enumerate(label_names):
        if class_index >= coef.shape[0]:
            break
        lines.append(f"Class '{class_name}':")
        for index in np.argsort(np.abs(coef[class_index]))[-10:][::-1]:
            lines.append(
                f"  {feature_names[index]:>20}  "
                f"{coef[class_index][index]:+.4f}"
            )


def _append_nb_features(lines, model, feature_names, label_names):
    if hasattr(model, "feature_log_prob_"):
        values = model.feature_log_prob_
    elif hasattr(model, "theta_"):
        values = model.theta_
    else:
        return

    for class_index, class_name in enumerate(label_names):
        if class_index >= values.shape[0]:
            break
        lines.append(f"Class '{class_name}':")
        for index in np.argsort(values[class_index])[-10:][::-1]:
            lines.append(
                f"  {feature_names[index]:>20}  "
                f"{values[class_index][index]:.4f}"
            )


def _append_rf_features(lines, importances, feature_names):
    lines.append("Top features (global):")
    for index in np.argsort(importances)[-10:][::-1]:
        lines.append(f"  {feature_names[index]:>20}  {importances[index]:.4f}")


def save_feature_importance_file(blocks):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FEATURE_IMPORTANCE_FILE, "w", encoding="utf-8") as file:
        file.write("FEATURE IMPORTANCE REPORT\n")
        file.write("=" * 50 + "\n")
        for block in blocks:
            file.write(block + "\n")


def save_similar_words(word_vector_models, dataset_name):
    query_words = DATASET_QUERY_WORDS.get(dataset_name, ["good", "bad", "great"])
    lines = ["SIMILAR WORDS REPORT", "=" * 50, ""]

    for embedding_name, embedding in word_vector_models.items():
        keyed_vectors = embedding.get_word_vectors()
        if keyed_vectors is None:
            continue
        lines.append(f"--- {embedding_name.upper()} ---")
        for word in query_words:
            if word not in keyed_vectors:
                lines.append(f"  '{word}' not in vocabulary")
                continue

            similar_words = keyed_vectors.most_similar(word, topn=10)
            lines.append(f"  Query: '{word}'")
            for rank, (similar_word, score) in enumerate(similar_words, 1):
                lines.append(f"    {rank:>2}. {similar_word:<20} {score:.4f}")
        lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SIMILAR_WORDS_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def _relative_path(path):
    path = os.path.abspath(path)
    try:
        if os.path.commonpath([path, BASE_DIR]) == BASE_DIR:
            return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
    except ValueError:
        pass
    return path
