"""Lab 1 command handlers."""

import os

from lab1 import classifier
from lab1 import data_manager
from lab1 import nlp_core
from lab1 import visualizer as lab1_visualizer
from utils import extract_quoted_args, log_error


HELP_SECTION = (
    "--- Lab 1 ---\n"
    "/task <name> \"text\" \"class\"\n"
    "/full_pipeline \"text\" \"class\"\n"
    "/classifier \"text\"\n"
    "/stats\n\n"
    "Tasks: tokenize, remove_stopwords, lemmatize, stemming,\n"
    "stats, n-grams, plot_histogram, plot_wordcloud, plot_barchart\n"
)

_CLASS_ALIASES = {
    "pozytywny": "pozytywny",
    "neutralny": "neutralny",
    "negatywny": "negatywny",
}


def register_handlers(bot):
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
        for sentence in sentences:
            data_manager.save_record(sentence, normalized_class)

        tokens = nlp_core.tokenize_text(user_text)
        clean = [t for t in tokens if t not in '.,!?;:()[]"\'']
        no_stop = nlp_core.remove_stopwords(clean)
        lemmas = nlp_core.lemmatize(user_text)
        stems = nlp_core.stemming(clean)
        _, bow_vec = nlp_core.bag_of_words(user_text)
        _, tfidf_vec = nlp_core.tf_idf(user_text)
        stats = nlp_core.get_stats(clean)

        response = (
            "--- FULL PIPELINE ---\n"
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

        for plot_path in [
            lab1_visualizer.plot_most_common_words(clean),
            lab1_visualizer.plot_token_length_histogram(clean),
            lab1_visualizer.plot_wordcloud(user_text),
        ]:
            if plot_path and os.path.exists(plot_path):
                with open(plot_path, "rb") as f:
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

        all_text = " ".join(record["text"] for record in records)
        labels = [record["class"] for record in records]
        class_counts = {label: labels.count(label) for label in set(labels)}
        class_str = "\n".join(f"  {key}: {value}" for key, value in class_counts.items())

        tokens = nlp_core.tokenize_text(all_text)
        clean = [t.lower() for t in tokens if t not in '.,!?;:()[]"\'-']
        unique = set(clean)

        response = (
            "Dataset Statistics\n\n"
            f"Classes:\n{class_str}\n\n"
            f"Unique tokens: {len(unique)}\n"
            f"Sample: {list(unique)[:10]}"
        )
        bot.reply_to(message, response)

        for plot_path in [
            lab1_visualizer.plot_most_common_words(clean),
            lab1_visualizer.plot_token_length_histogram(clean),
            lab1_visualizer.plot_wordcloud(all_text),
        ]:
            if plot_path and os.path.exists(plot_path):
                with open(plot_path, "rb") as f:
                    bot.send_photo(message.chat.id, f)

    except Exception as e:
        log_error("stats", e)
        bot.reply_to(message, "Error computing stats.")


__all__ = ["HELP_SECTION", "register_handlers"]
