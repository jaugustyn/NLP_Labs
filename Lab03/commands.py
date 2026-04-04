import os
import threading
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    SENTIMENT_METHODS, NEURAL_MODELS, VALID_DATASETS,
    RESULTS_FILE, RESULTS_DIR, PLOTS_DIR, CUSTOM_LABELS,
    BINARY_LABEL_MAP, MODELS_DIR,
)
from utils import parse_params, extract_quoted_args, log_error, format_duration
from data_loader import load_dataset, add_record, get_custom_stats
from training import train_neural_model
from model_loader import list_models, find_model_for_method
from sentiment_methods import predict_sentiment
import visualizations as viz

from lab1 import data_manager, nlp_core
from lab1 import visualizer as lab1_visualizer
from lab1 import classifier
from lab2.experiment import run_experiment as run_lab2_experiment
from lab2 import visualizer as lab2_visualizer


def register_handlers(bot):
    """Register all command handlers on the Telegram bot."""

    @bot.message_handler(commands=["start", "help"])
    def cmd_help(message):
        bot.reply_to(message, HELP_TEXT)

    # ---- Lab 1 ----
    @bot.message_handler(commands=["task"])
    def cmd_task(message):
        _handle_task(bot, message)

    @bot.message_handler(commands=["full_pipeline"])
    def cmd_pipeline(message):
        _handle_full_pipeline(bot, message)

    @bot.message_handler(commands=["classifier"])
    def cmd_classifier(message):
        _handle_classifier(bot, message)

    @bot.message_handler(commands=["stats"])
    def cmd_stats(message):
        _handle_stats(bot, message)

    # ---- Lab 2 ----
    @bot.message_handler(commands=["classify"])
    def cmd_classify(message):
        _handle_classify(bot, message)

    # ---- Lab 3 ----
    @bot.message_handler(commands=["sentiment"])
    def cmd_sentiment(message):
        _handle_sentiment(bot, message)

    @bot.message_handler(commands=["train"])
    def cmd_train(message):
        _handle_train(bot, message)

    @bot.message_handler(commands=["compare"])
    def cmd_compare(message):
        _handle_compare(bot, message)

    @bot.message_handler(commands=["add_sentiment"])
    def cmd_add(message):
        _handle_add_sentiment(bot, message)

    @bot.message_handler(commands=["models"])
    def cmd_models(message):
        _handle_models(bot, message)


# =====================================================================
#  HELP
# =====================================================================

LAB2_DATASETS = {"20news_group", "imdb", "amazon", "ag_news"}
LAB2_METHODS = {"nb", "rf", "mlp", "logreg", "all"}

HELP_TEXT = (
    "NLP Bot — Lab 1 + Lab 2 + Lab 3\n\n"
    "--- Lab 1 ---\n"
    "/task <name> \"text\" \"class\"\n"
    "/full_pipeline \"text\" \"class\"\n"
    "/classifier \"text\"\n"
    "/stats\n\n"
    "Tasks: tokenize, remove_stopwords, lemmatize, stemming,\n"
    "stats, n-grams, plot_histogram, plot_wordcloud, plot_barchart\n\n"
    "--- Lab 2 ---\n"
    "/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>\n"
    "  Datasets: 20news_group, imdb, amazon, ag_news\n"
    "  Methods: nb, rf, mlp, logreg, all\n\n"
    "--- Lab 3 ---\n"
    "/sentiment method=<m> text=\"...\" [dataset=<d>]\n"
    "  Methods: rule, nb, transformer, textblob, stanza,\n"
    "  simplernn, lstm, gru\n\n"
    "/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>\n\n"
    "/compare dataset=<amazon|imdb|custom> methods=<m1,m2,...>\n\n"
    "/add_sentiment \"text\" \"label\"\n"
    "  Labels: pozytywny, neutralny, negatywny\n\n"
    "/models — list saved models\n\n"
    "Examples:\n"
    '/sentiment method=rule text="Great movie!"\n'
    '/sentiment method=lstm dataset=imdb text="Terrible product"\n'
    "/train model=lstm dataset=imdb\n"
    "/compare dataset=imdb methods=rule,nb,transformer,textblob,lstm\n"
    '/add_sentiment "Świetny film" "pozytywny"\n'
)


# =====================================================================
#  /sentiment
# =====================================================================

def _handle_sentiment(bot, message):
    try:
        params = parse_params(message.text)
        method = params.get("method")
        text = params.get("text")

        if not method and not text:
            bot.reply_to(
                message,
                "Usage: /sentiment method=<method> text=\"your text\"\n\n"
                f"Available methods: {', '.join(SENTIMENT_METHODS)}\n\n"
                "Example:\n"
                "/sentiment method=rule text=\"Great product!\""
            )
            return

        if not method:
            bot.reply_to(
                message,
                f"Missing 'method'. Available: {', '.join(SENTIMENT_METHODS)}\n"
                "Example: /sentiment method=transformer text=\"Great movie\""
            )
            return

        if method not in SENTIMENT_METHODS:
            bot.reply_to(
                message,
                f"Unknown method: '{method}'\n"
                f"Available: {', '.join(SENTIMENT_METHODS)}"
            )
            return

        if not text:
            bot.reply_to(
                message,
                "Missing 'text'. Put it in quotes.\n"
                f"Example: /sentiment method={method} text=\"Your text here\""
            )
            return

        dataset = params.get("dataset")

        # For neural/nb: ensure a model is available
        if method in NEURAL_MODELS and not dataset:
            ds = find_model_for_method(method)
            if not ds:
                bot.reply_to(
                    message,
                    f"No trained {method.upper()} model found.\n"
                    f"Train one first:\n"
                    f"  /train model={method} dataset=<amazon|imdb|custom>\n\n"
                    f"Or specify: /sentiment method={method} "
                    f'dataset=imdb text="..."'
                )
                return
            dataset = ds

        bot.reply_to(message, f"Analyzing with {method}...")
        label, confidence = predict_sentiment(method, text, dataset)

        response = ""
        if dataset and method in NEURAL_MODELS + ["nb"]:
            response += f"Dataset: {dataset}\n"
        response += (
            f"Model: {method.upper()}\n"
            f"Prediction: {label}\n"
            f"Confidence: {confidence}"
        )
        bot.send_message(message.chat.id, response)

    except FileNotFoundError as e:
        bot.reply_to(message, str(e))
    except Exception as e:
        log_error("sentiment", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /train
# =====================================================================

def _handle_train(bot, message):
    try:
        params = parse_params(message.text)
        model_type = params.get("model")
        dataset_name = params.get("dataset")

        if not model_type and not dataset_name:
            bot.reply_to(
                message,
                "Usage: /train model=<simplernn|lstm|gru> "
                "dataset=<amazon|imdb|custom>\n\n"
                "Example: /train model=lstm dataset=imdb"
            )
            return

        if not model_type or model_type not in NEURAL_MODELS:
            bot.reply_to(
                message,
                f"Invalid model. Available: {', '.join(NEURAL_MODELS)}"
            )
            return

        if not dataset_name or dataset_name not in VALID_DATASETS:
            bot.reply_to(
                message,
                f"Invalid dataset. Available: {', '.join(VALID_DATASETS)}"
            )
            return

        bot.reply_to(
            message,
            f"Starting training:\n"
            f"  Model: {model_type.upper()}\n"
            f"  Dataset: {dataset_name}\n\n"
            f"This may take several minutes..."
        )

        def _run():
            chat_id = message.chat.id
            try:
                def progress(msg):
                    bot.send_message(chat_id, f"[Training] {msg}")

                texts, labels, label_names = load_dataset(dataset_name)
                result = train_neural_model(
                    model_type, dataset_name, texts, labels, label_names,
                    progress_callback=progress,
                )

                # Training history plot
                plot_path = viz.plot_training_history(
                    result["history"], model_type, dataset_name,
                )
                if plot_path and os.path.exists(plot_path):
                    with open(plot_path, "rb") as f:
                        bot.send_photo(
                            chat_id, f,
                            caption=f"Training history: {model_type.upper()}",
                        )

                summary = (
                    f"Training complete!\n\n"
                    f"Model: {model_type.upper()}\n"
                    f"Dataset: {dataset_name}\n"
                    f"Epochs: {result['epochs_run']}\n"
                    f"Val accuracy: {result['val_accuracy']}\n"
                    f"Val loss: {result['val_loss']}\n"
                    f"Duration: {format_duration(result['duration'])}\n\n"
                    f"Saved:\n"
                    f"  {result['model_path']}\n"
                    f"  {result['tokenizer_path']}\n"
                    f"  {result['encoder_path']}"
                )
                bot.send_message(chat_id, summary)

            except Exception as e:
                log_error("train_thread", e)
                bot.send_message(
                    chat_id, f"Training failed: {type(e).__name__}: {e}"
                )

        threading.Thread(target=_run, daemon=True).start()

    except Exception as e:
        log_error("train", e)
        bot.reply_to(message, f"Error: {e}")


# =====================================================================
#  /compare
# =====================================================================

def _handle_compare(bot, message):
    try:
        params = parse_params(message.text)
        dataset_name = params.get("dataset")
        methods_str = params.get("methods")

        if not dataset_name and not methods_str:
            bot.reply_to(
                message,
                "Usage: /compare dataset=<d> methods=<m1,m2,...>\n\n"
                f"Datasets: {', '.join(VALID_DATASETS)}\n"
                f"Methods: {', '.join(SENTIMENT_METHODS)}\n\n"
                "Example:\n"
                "/compare dataset=imdb methods=rule,nb,transformer,textblob"
            )
            return

        if not dataset_name or dataset_name not in VALID_DATASETS:
            bot.reply_to(
                message,
                f"Invalid dataset. Available: {', '.join(VALID_DATASETS)}"
            )
            return

        if not methods_str:
            bot.reply_to(message, "Missing 'methods' parameter.")
            return

        methods = [m.strip().lower() for m in methods_str.split(",")]
        for m in methods:
            if m not in SENTIMENT_METHODS:
                bot.reply_to(message, f"Unknown method: '{m}'")
                return

        bot.reply_to(
            message,
            f"Comparing {len(methods)} methods on '{dataset_name}'...\n"
            f"Methods: {', '.join(methods)}\n\n"
            f"This may take a while."
        )

        def _run():
            chat_id = message.chat.id
            try:
                texts, labels, label_names = load_dataset(
                    dataset_name, max_samples=2000,
                )

                idx = np.arange(len(texts))
                train_idx, test_idx = train_test_split(
                    idx, test_size=0.3, random_state=42, stratify=labels,
                )
                test_texts = [texts[i] for i in test_idx]
                y_true = labels[test_idx]

                # Pre-train NB if needed
                _train_nb_if_needed(
                    [texts[i] for i in train_idx],
                    labels[train_idx], label_names, dataset_name,
                )

                results = []
                for method in methods:
                    bot.send_message(chat_id, f"Evaluating {method}...")
                    try:
                        y_pred = _batch_predict(
                            method, test_texts, label_names, dataset_name,
                        )
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(
                            y_true, y_pred, average="macro", zero_division=0,
                        )
                        rec = recall_score(
                            y_true, y_pred, average="macro", zero_division=0,
                        )
                        f1 = f1_score(
                            y_true, y_pred, average="macro", zero_division=0,
                        )
                        results.append({
                            "dataset": dataset_name,
                            "method": method,
                            "accuracy": round(acc, 4),
                            "precision": round(prec, 4),
                            "recall": round(rec, 4),
                            "macro_f1": round(f1, 4),
                            "model_path": _model_path_for(method, dataset_name),
                        })
                        viz.plot_confusion_matrix(
                            y_true, y_pred, label_names, method, dataset_name,
                        )
                    except Exception as e:
                        log_error(f"compare_{method}", e)
                        bot.send_message(
                            chat_id, f"Method '{method}' failed: {e}"
                        )

                if not results:
                    bot.send_message(chat_id, "All methods failed.")
                    return

                # Save CSV
                os.makedirs(RESULTS_DIR, exist_ok=True)
                df = pd.DataFrame(results)
                df.to_csv(RESULTS_FILE, index=False)

                # Comparison chart
                chart = viz.plot_comparison(df, dataset_name)

                # Word clouds per class
                for cls_idx, cls_name in enumerate(label_names):
                    cls_texts = [
                        t for t, l in zip(test_texts, y_true) if l == cls_idx
                    ]
                    if cls_texts:
                        viz.plot_wordcloud(cls_texts, cls_name, dataset_name)

                # Class distribution
                counts = Counter(y_true.tolist())
                label_counts = {label_names[k]: v for k, v in counts.items()}
                viz.plot_class_distribution(label_counts, dataset_name)

                # Send summary
                table = "Results:\n\n"
                for r in results:
                    table += (
                        f"  {r['method']:>12}: "
                        f"acc={r['accuracy']:.4f}  "
                        f"f1={r['macro_f1']:.4f}\n"
                    )
                table += f"\nSaved to: {RESULTS_FILE}"
                bot.send_message(chat_id, table)

                if chart and os.path.exists(chart):
                    with open(chart, "rb") as f:
                        bot.send_photo(
                            chat_id, f, caption="Methods comparison",
                        )

            except Exception as e:
                log_error("compare_thread", e)
                bot.send_message(chat_id, f"Comparison failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    except Exception as e:
        log_error("compare", e)
        bot.reply_to(message, f"Error: {e}")


def _train_nb_if_needed(train_texts, train_labels, label_names, dataset_name):
    """Train and save NB model for comparison if not already saved."""
    from model_loader import load_sklearn_model, save_sklearn_model

    if load_sklearn_model("nb", dataset_name) is not None:
        return

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from preprocessing import clean_text

    cleaned = [clean_text(t) for t in train_texts]
    vec = TfidfVectorizer(max_features=10000)
    X = vec.fit_transform(cleaned)
    model = MultinomialNB()
    model.fit(X, train_labels)
    save_sklearn_model("nb", dataset_name, {
        "vectorizer": vec, "model": model, "label_names": label_names,
    })


def _batch_predict(method, texts, label_names, dataset_name):
    """Predict labels for a batch of texts. Returns int label array."""
    is_binary = len(label_names) == 2
    preds = []
    for text in texts:
        try:
            label, _ = predict_sentiment(method, text, dataset_name)
            if is_binary:
                mapped = BINARY_LABEL_MAP.get(label, label)
                if mapped in label_names:
                    preds.append(label_names.index(mapped))
                else:
                    preds.append(0)
            else:
                if label in label_names:
                    preds.append(label_names.index(label))
                else:
                    preds.append(0)
        except Exception:
            preds.append(0)
    return np.array(preds)


def _model_path_for(method, dataset_name):
    """Get the model file path string for a method."""
    if method in NEURAL_MODELS:
        return os.path.join(MODELS_DIR, f"{method}_{dataset_name}.h5")
    if method == "nb":
        return os.path.join(MODELS_DIR, f"nb_{dataset_name}_sklearn.pkl")
    return ""


# =====================================================================
#  /add_sentiment
# =====================================================================

def _handle_add_sentiment(bot, message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(
                message,
                "Usage: /add_sentiment \"text\" \"label\"\n\n"
                f"Labels: {', '.join(CUSTOM_LABELS)}\n\n"
                "Example:\n"
                "/add_sentiment \"Świetny film\" \"pozytywny\""
            )
            return

        args = extract_quoted_args(parts[1], 2)
        if not args:
            bot.reply_to(
                message,
                "Put text and label in quotes.\n"
                "Example: /add_sentiment \"text\" \"label\""
            )
            return

        text, label = args
        if not text.strip():
            bot.reply_to(message, "Text cannot be empty.")
            return

        normalized = add_record(text, label)
        stats = get_custom_stats()
        bot.reply_to(
            message,
            f"Record added!\n"
            f"  Text: \"{text}\"\n"
            f"  Label: {normalized}\n\n"
            f"Dataset now has {stats['total']} records."
        )

    except ValueError as e:
        bot.reply_to(message, str(e))
    except Exception as e:
        log_error("add_sentiment", e)
        bot.reply_to(message, f"Error: {e}")


# =====================================================================
#  /models
# =====================================================================

def _handle_models(bot, message):
    try:
        models = list_models()
        if not models:
            bot.reply_to(
                message,
                "No saved models found.\n"
                "Train one: /train model=<simplernn|lstm|gru> dataset=<name>"
            )
            return

        lines = ["Saved models:\n"]
        for m in models:
            tok_str = "tokenizer" if m["tokenizer"] else "no tokenizer"
            enc_str = "encoder" if m["encoder"] else "no encoder"
            lines.append(
                f"  {m['file']}\n"
                f"    Type: {m['model_type']} | "
                f"Dataset: {m['dataset']} | "
                f"Format: {m['format']}\n"
                f"    {tok_str}, {enc_str}"
            )
        bot.reply_to(message, "\n".join(lines))

    except Exception as e:
        log_error("models", e)
        bot.reply_to(message, f"Error: {e}")


# =====================================================================
#  /classify  (Lab 2)
# =====================================================================

def _handle_classify(bot, message):
    try:
        params = parse_params(message.text)

        dataset = params.get("dataset")
        if not dataset or dataset not in LAB2_DATASETS:
            bot.reply_to(
                message,
                f"Invalid or missing dataset. Allowed: {', '.join(sorted(LAB2_DATASETS))}"
            )
            return

        method = params.get("method")
        if not method:
            bot.reply_to(message, "Missing 'method' parameter.")
            return
        methods_list = [m.strip().lower() for m in method.split(",")]
        for m in methods_list:
            if m not in LAB2_METHODS:
                bot.reply_to(
                    message,
                    f"Unknown method '{m}'. Allowed: {', '.join(sorted(LAB2_METHODS))}"
                )
                return

        gs_raw = params.get("gridsearch", "false").lower()
        if gs_raw not in ("true", "false"):
            bot.reply_to(message, "gridsearch must be 'true' or 'false'.")
            return
        gridsearch = gs_raw == "true"

        try:
            n_runs = int(params.get("run", "1"))
            if n_runs < 1 or n_runs > 3:
                raise ValueError
        except ValueError:
            bot.reply_to(message, "run must be 1, 2 or 3.")
            return

        bot.reply_to(
            message,
            f"Starting Lab 2 experiment:\n"
            f"  dataset    = {dataset}\n"
            f"  method     = {method}\n"
            f"  gridsearch = {gridsearch}\n"
            f"  runs       = {n_runs}\n\n"
            f"This may take several minutes.",
        )

        def _run():
            chat_id = message.chat.id
            try:
                def progress(msg):
                    bot.send_message(chat_id, f"[Progress] {msg}")

                summary = run_lab2_experiment(
                    dataset_name=dataset,
                    method_str=method,
                    gridsearch=gridsearch,
                    n_runs=n_runs,
                    progress_callback=progress,
                )
                bot.send_message(chat_id, summary)

                wc_path = os.path.join(
                    lab2_visualizer.PLOTS_DIR, "wordcloud_corpus.png"
                )
                if os.path.exists(wc_path):
                    with open(wc_path, "rb") as f:
                        bot.send_photo(chat_id, f, caption="Word Cloud (corpus)")

            except Exception as e:
                log_error("classify_thread", e)
                bot.send_message(
                    chat_id,
                    f"Experiment failed: {type(e).__name__}. Check server logs."
                )

        threading.Thread(target=_run, daemon=True).start()

    except Exception as e:
        log_error("classify", e)
        bot.reply_to(message, "An error occurred while parsing your command.")


# =====================================================================
#  Lab 1 handlers (inherited)
# =====================================================================

_CLASS_ALIASES = {
    "pozytywny": "pozytywny",
    "neutralny": "neutralny",
    "negatywny": "negatywny",
}


def _normalize_class(raw):
    return _CLASS_ALIASES.get(raw.strip().lower())


def _handle_task(bot, message):
    try:
        parts = message.text.split(maxsplit=2)
        if len(parts) < 3:
            bot.reply_to(message, "Usage: /task <task_name> \"text\" \"class\"")
            return

        task_name = parts[1]
        extracted = extract_quoted_args(parts[2], 2)
        if not extracted:
            bot.reply_to(message, "Put text and class in quotes.")
            return

        user_text, text_class = extracted
        if not user_text.strip():
            bot.reply_to(message, "Text is empty.")
            return

        normalized_class = _normalize_class(text_class)
        if not normalized_class:
            bot.reply_to(
                message, "Invalid class. Allowed: pozytywny, neutralny, negatywny."
            )
            return

        nlp_tasks = [
            "tokenize", "remove_stopwords", "lemmatize", "stemming",
            "stats", "n-grams",
        ]
        vis_tasks = ["plot_histogram", "plot_wordcloud", "plot_barchart"]
        plot_path = None
        response = f"Task: {task_name}\n"

        if task_name in nlp_tasks:
            result = nlp_core.run_task(task_name, user_text)
            response += f"Result:\n{result}"
        elif task_name in vis_tasks:
            tokens = nlp_core.tokenize_text(user_text)
            clean = [t for t in tokens if t not in '.,!?;:()[]"\'']
            if task_name == "plot_histogram":
                plot_path = lab1_visualizer.plot_token_length_histogram(clean)
            elif task_name == "plot_wordcloud":
                plot_path = lab1_visualizer.plot_wordcloud(user_text)
            elif task_name == "plot_barchart":
                plot_path = lab1_visualizer.plot_most_common_words(clean)
            response += "Plot generated." if plot_path else "Could not generate."
        else:
            bot.reply_to(message, f"Unknown task: {task_name}. Use /help.")
            return

        data_manager.save_record(user_text, normalized_class)
        bot.reply_to(message, response)

        if plot_path and os.path.exists(plot_path):
            with open(plot_path, "rb") as f:
                bot.send_photo(message.chat.id, f)

    except Exception as e:
        log_error("task", e)
        bot.reply_to(message, "Error processing task.")


def _handle_full_pipeline(bot, message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Usage: /full_pipeline \"text\" \"class\"")
            return

        extracted = extract_quoted_args(parts[1], 2)
        if not extracted:
            bot.reply_to(message, "Put text and class in quotes.")
            return

        user_text, text_class = extracted
        if not user_text.strip():
            bot.reply_to(message, "Text is empty.")
            return

        normalized_class = _normalize_class(text_class)
        if not normalized_class:
            bot.reply_to(
                message, "Invalid class. Allowed: pozytywny, neutralny, negatywny."
            )
            return

        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(user_text, language="polish")
        for s in sentences:
            data_manager.save_record(s, normalized_class)

        tokens = nlp_core.tokenize_text(user_text)
        clean = [t for t in tokens if t not in '.,!?;:()[]"\'']
        no_stop = nlp_core.remove_stopwords(clean)
        lemmas = nlp_core.lemmatize(user_text)
        stems = nlp_core.stemming(clean)
        bow_feat, bow_vec = nlp_core.bag_of_words(user_text)
        tfidf_feat, tfidf_vec = nlp_core.tf_idf(user_text)
        stats = nlp_core.get_stats(clean)

        response = (
            f"--- FULL PIPELINE ---\n"
            f"1. Saved {len(sentences)} sentence(s)\n"
            f"2. Tokens: {tokens[:10]}...\n"
            f"3. Without stopwords: {no_stop[:10]}...\n"
            f"4. Lemmas: {lemmas[:10]}...\n"
            f"5. Stems: {stems[:10]}...\n"
            f"6. BoW shape: {bow_vec.shape if len(bow_vec) > 0 else 'N/A'}\n"
            f"7. TF-IDF shape: {tfidf_vec.shape if len(tfidf_vec) > 0 else 'N/A'}\n"
            f"8. Stats: {stats}"
        )
        bot.reply_to(message, response)

        for p in [
            lab1_visualizer.plot_most_common_words(clean),
            lab1_visualizer.plot_token_length_histogram(clean),
            lab1_visualizer.plot_wordcloud(user_text),
        ]:
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    bot.send_photo(message.chat.id, f)

    except Exception as e:
        log_error("full_pipeline", e)
        bot.reply_to(message, "Error running pipeline.")


def _handle_classifier(bot, message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Usage: /classifier \"text\"")
            return

        extracted = extract_quoted_args(parts[1], 1)
        user_text = extracted[0] if extracted else parts[1].replace('"', "")
        if not user_text.strip():
            bot.reply_to(message, "Text is empty.")
            return

        prediction = classifier.train_and_predict(user_text)
        bot.reply_to(message, f"Prediction: {prediction}")

    except Exception as e:
        log_error("classifier", e)
        bot.reply_to(message, "Error classifying.")


def _handle_stats(bot, message):
    try:
        records = data_manager.load_records()
        if not records:
            bot.reply_to(message, "No data. Use /task or /full_pipeline first.")
            return

        all_text = " ".join(r["text"] for r in records)
        labels = [r["class"] for r in records]
        class_counts = {k: labels.count(k) for k in set(labels)}
        class_str = "\n".join(f"  {k}: {v}" for k, v in class_counts.items())

        tokens = nlp_core.tokenize_text(all_text)
        clean = [t.lower() for t in tokens if t not in '.,!?;:()[]"\'-']
        unique = set(clean)

        response = (
            f"Dataset Statistics\n\n"
            f"Classes:\n{class_str}\n\n"
            f"Unique tokens: {len(unique)}\n"
            f"Sample: {list(unique)[:10]}"
        )
        bot.reply_to(message, response)

        for p in [
            lab1_visualizer.plot_most_common_words(clean),
            lab1_visualizer.plot_token_length_histogram(clean),
            lab1_visualizer.plot_wordcloud(all_text),
        ]:
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    bot.send_photo(message.chat.id, f)

    except Exception as e:
        log_error("stats", e)
        bot.reply_to(message, "Error computing stats.")
