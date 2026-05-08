import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

from lab2.dataset_loader import load_dataset, DATASET_QUERY_WORDS
from lab2.text_embeddings import get_embedding, EMBEDDING_NAMES
from lab2.models import get_model, get_grid_params, resolve_methods
from lab2 import visualizer as viz

SEEDS = [42, 1337, 2137]

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "lab2results.csv")
SIMILAR_WORDS_FILE = os.path.join(RESULTS_DIR, "lab2_similar_words.txt")
FEATURE_IMPORTANCE_FILE = os.path.join(RESULTS_DIR, "lab2_feature_importance.txt")


def run_experiment(dataset_name, method_str, gridsearch, n_runs,
                   progress_callback=None):

    def progress(msg):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass
        print(msg)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = SEEDS[:n_runs]
    method_names = resolve_methods(method_str)

    # 1. Load dataset
    progress(f"Loading dataset '{dataset_name}'...")
    texts, labels, label_names = load_dataset(dataset_name)
    progress(f"Loaded {len(texts)} samples, {len(label_names)} classes")

    # 2. Word clouds
    progress("Generating word clouds...")
    viz.plot_wordcloud_corpus(texts)
    viz.plot_wordcloud_per_class(texts, labels, label_names)
    progress("Word clouds done.")

    # 3. Pre-compute embeddings on full corpus
    all_embeddings = {}
    word_vector_models = {}

    for emb_name in EMBEDDING_NAMES:
        progress(f"Computing '{emb_name}' embeddings...")
        emb = get_embedding(emb_name)
        X = emb.fit_transform(texts)
        all_embeddings[emb_name] = (X, emb)
        if emb_name in ("word2vec", "glove"):
            word_vector_models[emb_name] = emb
        shape = X.shape if hasattr(X, "shape") else "?"
        progress(f"  {emb_name} done ÔÇö shape {shape}")

    # 4. Run classification experiments
    results = []
    feature_importance_lines = []

    for emb_name in EMBEDDING_NAMES:
        X_all, emb = all_embeddings[emb_name]

        # Pre-compute embedding visualisations once per embedding
        last_seed = seeds[-1]
        idx = np.arange(len(texts))
        _, test_idx_viz = train_test_split(
            idx, test_size=0.2, random_state=last_seed, stratify=labels,
        )
        X_test_viz = X_all[test_idx_viz]
        y_test_viz = labels[test_idx_viz]

        for vm in ["pca", "tsne", "svd"]:
            progress(f"  Reducing {emb_name} with {vm.upper()}...")
            X_2d, y_sub = viz.reduce_for_visualization(X_test_viz, y_test_viz, vm)
            for model_name in method_names:
                viz.plot_embedding_2d(
                    X_2d, y_sub, label_names, vm,
                    f"{dataset_name}_{model_name}_{emb_name}_{vm}_embedding.png",
                )
        progress(f"  Embedding visualisations for {emb_name} done.")

        for model_name in method_names:
            progress(f"Training {model_name} on {emb_name}...")
            run_metrics = []
            last_y_true = last_y_pred = None
            last_model = None

            for seed in seeds:
                indices = np.arange(len(texts))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.2, random_state=seed, stratify=labels,
                )
                X_train, X_test = X_all[train_idx], X_all[test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                model = get_model(model_name, emb_name, seed)

                if gridsearch:
                    params = get_grid_params(model_name, emb_name)
                    if params:
                        gs = GridSearchCV(
                            model, params, cv=3,
                            scoring="f1_macro", n_jobs=-1,
                            error_score="raise",
                        )
                        gs.fit(X_train, y_train)
                        model = gs.best_estimator_
                        progress(f"  GridSearch best: {gs.best_params_}")
                    else:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                results.append({
                    "embedding": emb_name,
                    "model": model_name,
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                    "seed": seed,
                })
                run_metrics.append({"acc": acc, "f1": f1})

                last_y_true, last_y_pred = y_test, y_pred
                last_model = model

            avg_acc = np.mean([r["acc"] for r in run_metrics])
            avg_f1 = np.mean([r["f1"] for r in run_metrics])
            progress(f"  => avg acc={avg_acc:.4f}  macro_f1={avg_f1:.4f}")

            viz.plot_confusion(
                last_y_true, last_y_pred, label_names,
                f"confusion_{emb_name}_{model_name}.png",
            )

            fi = extract_feature_importance(
                last_model, emb, model_name, emb_name, label_names,
            )
            if fi:
                feature_importance_lines.append(fi)

    # 5. Similar words (word2vec / glove)
    if word_vector_models:
        save_similar_words(word_vector_models, dataset_name)
        progress("Similar words saved.")

    # 6. Word-level embedding visualisations
    query_words = DATASET_QUERY_WORDS.get(dataset_name, ["good", "bad", "great"])
    for wm_name, wm_emb in word_vector_models.items():
        wv = wm_emb.get_word_vectors()
        if wv is None:
            continue
        expanded = list(query_words)
        for w in query_words:
            if w in wv:
                expanded.extend([s[0] for s in wv.most_similar(w, topn=3)])
        expanded = list(dict.fromkeys(expanded))
        viz.plot_word_embeddings(wv, expanded, "pca", "word_embedding_pca.png")
        viz.plot_word_embeddings(wv, expanded, "tsne", "word_embedding_tsne.png")
    progress("Word embedding visualisations done.")

    # 7. Save CSV
    save_results_csv(results)
    progress(f"Results saved to {RESULTS_FILE}")

    # 8. Feature importance file
    if feature_importance_lines:
        save_feature_importance_file(feature_importance_lines)
        progress(f"Feature importance saved to {FEATURE_IMPORTANCE_FILE}")

    return format_summary(results, n_runs)


def save_results_csv(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding", "model", "accuracy", "macro_f1", "seed"])
        writer.writeheader()
        writer.writerows(results)

def format_summary(results, n_runs):
    lines = ["=== EXPERIMENT RESULTS ===\n"]

    combos = {}
    for r in results:
        key = (r["embedding"], r["model"])
        combos.setdefault(key, []).append(r)

    for (emb, mdl), rows in combos.items():
        avg_acc = np.mean([r["accuracy"] for r in rows])
        avg_f1 = np.mean([r["macro_f1"] for r in rows])
        lines.append(f"{emb:>10} + {mdl:<8}  acc={avg_acc:.4f}  f1={avg_f1:.4f}  (runs={len(rows)})")

    lines.append(f"\nTotal experiments: {len(results)}")
    lines.append(f"Results CSV: {RESULTS_FILE}")
    lines.append(f"Plots directory: {viz.PLOTS_DIR}/")
    return "\n".join(lines)


def extract_feature_importance(model, embedding, model_name, emb_name, label_names):
    feature_names = embedding.get_feature_names()
    if feature_names is None:
        return None

    feature_names = list(feature_names)
    lines = [f"\n=== {emb_name} + {model_name} ==="]

    try:
        if model_name == "logreg" and hasattr(model, "coef_"):
            coef = model.coef_
            if coef.shape[0] == 1:
                top = np.argsort(np.abs(coef[0]))[-10:][::-1]
                lines.append("Top features (global):")
                for i in top:
                    lines.append(f"  {feature_names[i]:>20}  {coef[0][i]:+.4f}")
            else:
                for ci, cname in enumerate(label_names):
                    if ci >= coef.shape[0]:
                        break
                    top = np.argsort(np.abs(coef[ci]))[-10:][::-1]
                    lines.append(f"Class '{cname}':")
                    for i in top:
                        lines.append(f"  {feature_names[i]:>20}  {coef[ci][i]:+.4f}")

        elif model_name == "nb":
            if hasattr(model, "feature_log_prob_"):
                probs = model.feature_log_prob_
                for ci, cname in enumerate(label_names):
                    if ci >= probs.shape[0]:
                        break
                    top = np.argsort(probs[ci])[-10:][::-1]
                    lines.append(f"Class '{cname}':")
                    for i in top:
                        lines.append(f"  {feature_names[i]:>20}  {probs[ci][i]:.4f}")
            elif hasattr(model, "theta_"):
                theta = model.theta_
                for ci, cname in enumerate(label_names):
                    if ci >= theta.shape[0]:
                        break
                    top = np.argsort(np.abs(theta[ci]))[-10:][::-1]
                    lines.append(f"Class '{cname}':")
                    for i in top:
                        lines.append(f"  {feature_names[i]:>20}  {theta[ci][i]:.4f}")

        elif model_name == "rf" and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            top = np.argsort(imp)[-10:][::-1]
            lines.append("Top features (global):")
            for i in top:
                lines.append(f"  {feature_names[i]:>20}  {imp[i]:.4f}")

        else:
            return None

    except Exception:
        return None

    return "\n".join(lines)

def save_feature_importance_file(blocks):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FEATURE_IMPORTANCE_FILE, "w", encoding="utf-8") as f:
        f.write("FEATURE IMPORTANCE REPORT\n")
        f.write("=" * 50 + "\n")
        for block in blocks:
            f.write(block + "\n")

def save_similar_words(word_vector_models, dataset_name):
    query_words = DATASET_QUERY_WORDS.get(dataset_name, ["good", "bad", "great"])
    lines = ["SIMILAR WORDS REPORT", "=" * 50, ""]

    for wm_name, wm_emb in word_vector_models.items():
        wv = wm_emb.get_word_vectors()
        if wv is None:
            continue
        lines.append(f"--- {wm_name.upper()} ---")
        for word in query_words:
            if word not in wv:
                lines.append(f"  '{word}' not in vocabulary")
                continue
            similar = wv.most_similar(word, topn=10)
            lines.append(f"  Query: '{word}'")
            for rank, (sw, score) in enumerate(similar, 1):
                lines.append(f"    {rank:>2}. {sw:<20} {score:.4f}")
        lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SIMILAR_WORDS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
