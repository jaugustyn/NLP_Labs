"""Lab 3 command handlers."""

from collections import Counter
import os
import threading

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from lab3 import visualizations as viz
from lab3.config import (
    BINARY_LABEL_MAP,
    CUSTOM_LABELS,
    MODELS_DIR,
    NEURAL_MODELS,
    RESULTS_DIR,
    RESULTS_FILE,
    SENTIMENT_METHODS,
    VALID_DATASETS,
)
from lab3.data_loader import add_record, get_custom_stats, load_dataset
from lab3.model_loader import find_model_for_method, list_models
from lab3.sentiment_methods import predict_sentiment
from lab3.training import train_neural_model
from utils import extract_quoted_args, format_duration, log_error, parse_params


HELP_SECTION = (
    "--- Lab 3 ---\n"
    "/sentiment method=<m> text=\"...\" [dataset=<d>]\n"
    "  Methods: rule, nb, rf, transformer, textblob, stanza,\n"
    "  simplernn, lstm, gru\n\n"
    "/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>\n"
    "/compare dataset=<amazon|imdb|custom> methods=<m1,m2,...>\n"
    "/add_sentiment \"text\" \"label\"\n"
    "  Labels: pozytywny, neutralny, negatywny\n"
    "/models - list saved models\n"
)


def register_handlers(bot):
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
                "/sentiment method=rule text=\"Great product!\"",
            )
            return

        if not method:
            bot.reply_to(
                message,
                f"Missing 'method'. Available: {', '.join(SENTIMENT_METHODS)}\n"
                "Example: /sentiment method=transformer text=\"Great movie\"",
            )
            return

        if method not in SENTIMENT_METHODS:
            bot.reply_to(
                message,
                f"Unknown method: '{method}'\n"
                f"Available: {', '.join(SENTIMENT_METHODS)}",
            )
            return

        if not text:
            bot.reply_to(
                message,
                "Missing 'text'. Put it in quotes.\n"
                f"Example: /sentiment method={method} text=\"Your text here\"",
            )
            return

        dataset = params.get("dataset")

        if method in NEURAL_MODELS and not dataset:
            ds = find_model_for_method(method)
            if not ds:
                bot.reply_to(
                    message,
                    f"No trained {method.upper()} model found.\n"
                    "Train one first:\n"
                    f"  /train model={method} dataset=<amazon|imdb|custom>\n\n"
                    f"Or specify: /sentiment method={method} "
                    "dataset=imdb text=\"...\"",
                )
                return
            dataset = ds

        bot.reply_to(message, f"Analyzing with {method}...")
        label, confidence = predict_sentiment(method, text, dataset)

        response = ""
        if dataset and method in NEURAL_MODELS + ["nb", "rf"]:
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
                "Example: /train model=lstm dataset=imdb",
            )
            return

        if not model_type or model_type not in NEURAL_MODELS:
            bot.reply_to(
                message,
                f"Invalid model. Available: {', '.join(NEURAL_MODELS)}",
            )
            return

        if not dataset_name or dataset_name not in VALID_DATASETS:
            bot.reply_to(
                message,
                f"Invalid dataset. Available: {', '.join(VALID_DATASETS)}",
            )
            return

        bot.reply_to(
            message,
            "Starting training:\n"
            f"  Model: {model_type.upper()}\n"
            f"  Dataset: {dataset_name}\n\n"
            "This may take several minutes...",
        )

        def _run():
            chat_id = message.chat.id
            try:
                def progress(msg):
                    bot.send_message(chat_id, f"[Training] {msg}")

                texts, labels, label_names = load_dataset(dataset_name)
                result = train_neural_model(
                    model_type,
                    dataset_name,
                    texts,
                    labels,
                    label_names,
                    progress_callback=progress,
                )

                plot_path = viz.plot_training_history(
                    result["history"],
                    model_type,
                    dataset_name,
                )
                if plot_path and os.path.exists(plot_path):
                    with open(plot_path, "rb") as f:
                        bot.send_photo(
                            chat_id,
                            f,
                            caption=f"Training history: {model_type.upper()}",
                        )

                summary = (
                    "Training complete!\n\n"
                    f"Model: {model_type.upper()}\n"
                    f"Dataset: {dataset_name}\n"
                    f"Epochs: {result['epochs_run']}\n"
                    f"Val accuracy: {result['val_accuracy']}\n"
                    f"Val loss: {result['val_loss']}\n"
                    f"Duration: {format_duration(result['duration'])}\n\n"
                    "Saved:\n"
                    f"  {result['model_path']}\n"
                    f"  {result['tokenizer_path']}\n"
                    f"  {result['encoder_path']}"
                )
                bot.send_message(chat_id, summary)

            except Exception as e:
                log_error("train_thread", e)
                bot.send_message(chat_id, f"Training failed: {type(e).__name__}: {e}")

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
                "/compare dataset=imdb methods=rule,nb,transformer,textblob",
            )
            return

        if not dataset_name or dataset_name not in VALID_DATASETS:
            bot.reply_to(
                message,
                f"Invalid dataset. Available: {', '.join(VALID_DATASETS)}",
            )
            return

        if not methods_str:
            bot.reply_to(message, "Missing 'methods' parameter.")
            return

        methods = [m.strip().lower() for m in methods_str.split(",")]
        for method in methods:
            if method not in SENTIMENT_METHODS:
                bot.reply_to(message, f"Unknown method: '{method}'")
                return

        bot.reply_to(
            message,
            f"Comparing {len(methods)} methods on '{dataset_name}'...\n"
            f"Methods: {', '.join(methods)}\n\n"
            "This may take a while.",
        )

        def _run():
            chat_id = message.chat.id
            try:
                texts, labels, label_names = load_dataset(
                    dataset_name,
                    max_samples=2000,
                )

                idx = np.arange(len(texts))
                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=0.3,
                    random_state=42,
                    stratify=labels,
                )
                test_texts = [texts[i] for i in test_idx]
                y_true = labels[test_idx]

                _train_ml_if_needed(
                    [texts[i] for i in train_idx],
                    labels[train_idx],
                    label_names,
                    dataset_name,
                )

                results = []
                for method in methods:
                    bot.send_message(chat_id, f"Evaluating {method}...")
                    try:
                        y_pred = _batch_predict(
                            method,
                            test_texts,
                            label_names,
                            dataset_name,
                        )
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                        rec = recall_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                        f1 = f1_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                        results.append(
                            {
                                "dataset": dataset_name,
                                "method": method,
                                "accuracy": round(acc, 4),
                                "precision": round(prec, 4),
                                "recall": round(rec, 4),
                                "macro_f1": round(f1, 4),
                                "model_path": _model_path_for(method, dataset_name),
                            }
                        )
                        viz.plot_confusion_matrix(
                            y_true,
                            y_pred,
                            label_names,
                            method,
                            dataset_name,
                        )
                    except Exception as e:
                        log_error(f"compare_{method}", e)
                        bot.send_message(chat_id, f"Method '{method}' failed: {e}")

                if not results:
                    bot.send_message(chat_id, "All methods failed.")
                    return

                os.makedirs(RESULTS_DIR, exist_ok=True)
                df = pd.DataFrame(results)
                df.to_csv(RESULTS_FILE, index=False)

                chart = viz.plot_comparison(df, dataset_name)

                for cls_idx, cls_name in enumerate(label_names):
                    cls_texts = [
                        text
                        for text, label in zip(test_texts, y_true)
                        if label == cls_idx
                    ]
                    if cls_texts:
                        viz.plot_wordcloud(cls_texts, cls_name, dataset_name)

                counts = Counter(y_true.tolist())
                label_counts = {label_names[k]: v for k, v in counts.items()}
                viz.plot_class_distribution(label_counts, dataset_name)

                table = "Results:\n\n"
                for result in results:
                    table += (
                        f"  {result['method']:>12}: "
                        f"acc={result['accuracy']:.4f}  "
                        f"f1={result['macro_f1']:.4f}\n"
                    )
                table += f"\nSaved to: {RESULTS_FILE}"
                bot.send_message(chat_id, table)

                if chart and os.path.exists(chart):
                    with open(chart, "rb") as f:
                        bot.send_photo(chat_id, f, caption="Methods comparison")

            except Exception as e:
                log_error("compare_thread", e)
                bot.send_message(chat_id, f"Comparison failed: {e}")

        threading.Thread(target=_run, daemon=True).start()

    except Exception as e:
        log_error("compare", e)
        bot.reply_to(message, f"Error: {e}")


def _train_ml_if_needed(train_texts, train_labels, label_names, dataset_name):
    """Train and save NB + RF models for comparison if not already saved."""
    from lab3.model_loader import load_sklearn_model, save_sklearn_model
    from lab3.preprocessing import clean_text
    from sklearn.feature_extraction.text import TfidfVectorizer

    cleaned = [clean_text(t) for t in train_texts]

    for model_name in ["nb", "rf"]:
        if load_sklearn_model(model_name, dataset_name) is not None:
            continue

        if model_name == "nb":
            from sklearn.naive_bayes import MultinomialNB

            clf = MultinomialNB()
        else:
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )

        vec = TfidfVectorizer(max_features=10000)
        X = vec.fit_transform(cleaned)
        clf.fit(X, train_labels)
        save_sklearn_model(
            model_name,
            dataset_name,
            {
                "vectorizer": vec,
                "model": clf,
                "label_names": label_names,
            },
        )


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
    if method in NEURAL_MODELS:
        return os.path.join(MODELS_DIR, f"{method}_{dataset_name}.h5")
    if method in ("nb", "rf"):
        return os.path.join(MODELS_DIR, f"{method}_{dataset_name}_sklearn.pkl")
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
                "/add_sentiment \"Swietny film\" \"pozytywny\"",
            )
            return

        args = extract_quoted_args(parts[1], 2)
        if not args:
            bot.reply_to(
                message,
                "Put text and label in quotes.\n"
                "Example: /add_sentiment \"text\" \"label\"",
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
            "Record added!\n"
            f"  Text: \"{text}\"\n"
            f"  Label: {normalized}\n\n"
            f"Dataset now has {stats['total']} records.",
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
                "Train one: /train model=<simplernn|lstm|gru> dataset=<name>",
            )
            return

        lines = ["Saved models:\n"]
        for model in models:
            tok_str = "tokenizer" if model["tokenizer"] else "no tokenizer"
            enc_str = "encoder" if model["encoder"] else "no encoder"
            lines.append(
                f"  {model['file']}\n"
                f"    Type: {model['model_type']} | "
                f"Dataset: {model['dataset']} | "
                f"Format: {model['format']}\n"
                f"    {tok_str}, {enc_str}"
            )
        bot.reply_to(message, "\n".join(lines))

    except Exception as e:
        log_error("models", e)
        bot.reply_to(message, f"Error: {e}")


__all__ = ["HELP_SECTION", "register_handlers"]
