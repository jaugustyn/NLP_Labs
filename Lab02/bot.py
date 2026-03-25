import os
import re
import threading
import traceback

import telebot
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

from lab1 import data_manager
from lab1 import nlp_core
from lab1 import visualizer as lab1_visualizer
from lab1 import classifier
from experiment import run_experiment
import visualizer as viz

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("Missing token! Make sure .env contains TELEGRAM_BOT_TOKEN.")

bot = telebot.TeleBot(TOKEN)

VALID_DATASETS = {"20news_group", "imdb", "amazon", "ag_news"}
VALID_METHODS = {"nb", "rf", "mlp", "logreg", "all"}

CLASS_ALIASES = {
    "pozytywny": "pozytywny",
    "neutralny": "neutralny",
    "negatywny": "negatywny",
}


def parse_params(text):
    params = {}
    for part in text.split():
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().lower()] = value.strip()
    return params

def extract_args(text, expected_args_count):
    matches = re.findall(r'"([^"]*)"', text)
    if len(matches) == expected_args_count:
        return matches
    return None

def normalize_and_validate_class(raw_class):
    normalized = raw_class.strip().lower()
    return CLASS_ALIASES.get(normalized)

def log_error(context, error):
    print(f"[ERROR] {context}: {type(error).__name__}: {error}")
    print(traceback.format_exc())


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    help_text = (
        "NLP Bot\n\n"
        "--- Lab 1 commands ---\n"
        "/task <task_name> \"text\" \"class\"\n"
        "/full_pipeline \"text\" \"class\"\n"
        "/classifier \"text\"\n"
        "/stats\n\n"
        "Tasks for /task:\n"
        "tokenize, remove_stopwords, lemmatize, stemming, stats, n-grams, "
        "plot_histogram, plot_wordcloud, plot_barchart\n\n"
        "--- Lab 2 commands ---\n"
        "/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>\n\n"
        "Parameters:\n"
        "  dataset   : 20news_group | imdb | amazon | ag_news\n"
        "  method    : nb | rf | mlp | logreg | all  (comma-separated OK)\n"
        "  gridsearch: true | false\n"
        "  run       : 1-3  (number of runs with different seeds)\n\n"
        "Examples:\n"
        "/classify dataset=20news_group method=all gridsearch=false run=1\n"
        "/classify dataset=imdb method=logreg gridsearch=true run=2"
    )
    bot.reply_to(message, help_text)


# --- Lab 1 commands ---

@bot.message_handler(commands=["task"])
def handle_task(message):
    try:
        parts = message.text.split(maxsplit=2)
        if len(parts) < 3:
            bot.reply_to(message, "Invalid format! Use: /task <task_name> \"text\" \"class\"")
            return

        task_name = parts[1]
        args_text = parts[2]

        extracted = extract_args(args_text, 2)
        if not extracted:
            bot.reply_to(message, "Could not parse arguments. Put text and class in quotes.")
            return

        user_text, text_class = extracted[0], extracted[1]

        if not user_text.strip():
            bot.reply_to(message, "Provided message text is empty!")
            return

        normalized_class = normalize_and_validate_class(text_class)
        if not normalized_class:
            bot.reply_to(message, "Invalid class. Allowed: pozytywny, neutralny, negatywny.")
            return

        response_msg = f"Running task: {task_name}\n"
        visualizations = ["plot_histogram", "plot_wordcloud", "plot_barchart"]
        nlp_tasks = ["tokenize", "remove_stopwords", "lemmatize", "stemming", "stats", "n-grams"]
        plot_path = None

        if task_name in nlp_tasks:
            result = nlp_core.run_task(task_name, user_text)
            response_msg += f"Result:\n{result}"
        elif task_name in visualizations:
            tokens = nlp_core.tokenize_text(user_text)
            clean_tokens = [t for t in tokens if t not in '.,!?;:()[]"\'']
            if task_name == "plot_histogram":
                plot_path = lab1_visualizer.plot_token_length_histogram(clean_tokens)
                response_msg += "Histogram was generated." if plot_path else "Could not generate."
            elif task_name == "plot_wordcloud":
                plot_path = lab1_visualizer.plot_wordcloud(user_text)
                response_msg += "Word cloud was generated." if plot_path else "Could not generate."
            elif task_name == "plot_barchart":
                plot_path = lab1_visualizer.plot_most_common_words(clean_tokens)
                response_msg += "Bar chart was generated." if plot_path else "Could not generate."
        else:
            bot.reply_to(message, f"Unknown task: {task_name}. Use /help to see available tasks.")
            return

        data_manager.save_record(user_text, normalized_class)
        bot.reply_to(message, response_msg)

        if plot_path and os.path.exists(plot_path):
            with open(plot_path, "rb") as photo:
                bot.send_photo(message.chat.id, photo)

    except Exception as e:
        log_error("task", e)
        bot.reply_to(message, "An error occurred while processing the task.")


@bot.message_handler(commands=["full_pipeline"])
def handle_full_pipeline(message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Invalid format! Use: /full_pipeline \"text\" \"class\"")
            return

        extracted = extract_args(parts[1], 2)
        if not extracted:
            bot.reply_to(message, "Invalid quote format. Example: /full_pipeline \"Sample text\" \"pozytywny\"")
            return

        user_text, text_class = extracted[0], extracted[1]

        if not user_text.strip():
            bot.reply_to(message, "Provided message text is empty!")
            return

        normalized_class = normalize_and_validate_class(text_class)
        if not normalized_class:
            bot.reply_to(message, "Invalid class. Allowed: pozytywny, neutralny, negatywny.")
            return

        sentences = sent_tokenize(user_text, language="polish")
        for s in sentences:
            data_manager.save_record(s, normalized_class)

        tokens = nlp_core.tokenize_text(user_text)
        clean_tokens = [t for t in tokens if t not in '.,!?;:()[]"\'']

        no_stop = nlp_core.remove_stopwords(clean_tokens)
        lemmas = nlp_core.lemmatize(user_text)
        stems = nlp_core.stemming(clean_tokens)
        bow_feat, bow_vec = nlp_core.bag_of_words(user_text)
        tfidf_feat, tfidf_vec = nlp_core.tf_idf(user_text)
        stats = nlp_core.get_stats(clean_tokens)

        response = (
            f"--- FULL PIPELINE ---\n"
            f"1. Saved entries (sentences: {len(sentences)})\n"
            f"2. Tokens: {tokens[:10]}...\n"
            f"3. Without stopwords (Top10): {no_stop[:10]}...\n"
            f"4. Lemmatization (Top10): {lemmas[:10]}...\n"
            f"5. Stemming (Top10): {stems[:10]}...\n"
            f"6. Bag of words: shape {bow_vec.shape if len(bow_vec)>0 else 'None'}\n"
            f"7. TF-IDF: shape {tfidf_vec.shape if len(tfidf_vec)>0 else 'None'}\n"
            f"8. Statistics: {stats}"
        )
        bot.reply_to(message, response)

        p1 = lab1_visualizer.plot_most_common_words(clean_tokens)
        p2 = lab1_visualizer.plot_token_length_histogram(clean_tokens)
        p3 = lab1_visualizer.plot_wordcloud(user_text)

        for p in [p1, p2, p3]:
            if p and os.path.exists(p):
                with open(p, "rb") as photo:
                    bot.send_photo(message.chat.id, photo)

    except Exception as e:
        log_error("full_pipeline", e)
        bot.reply_to(message, "An error occurred while running the pipeline.")


@bot.message_handler(commands=["classifier"])
def handle_classifier(message):
    try:
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            bot.reply_to(message, "Use: /classifier \"text_to_classify\"")
            return

        extracted = extract_args(parts[1], 1)
        if not extracted:
            user_text = parts[1].replace('"', '')
        else:
            user_text = extracted[0]

        if not user_text.strip():
            bot.reply_to(message, "Provided message text is empty!")
            return

        prediction = classifier.train_and_predict(user_text)
        bot.reply_to(message, f"Classifier prediction:\n{prediction}")

    except Exception as e:
        log_error("classifier", e)
        bot.reply_to(message, "An error occurred while classifying.")


@bot.message_handler(commands=["stats"])
def handle_stats(message):
    try:
        records = data_manager.load_records()
        if not records:
            bot.reply_to(message, "No data in dataset. Use /task or /full_pipeline first.")
            return

        all_text = " ".join([r["text"] for r in records])
        labels = [r["class"] for r in records]

        class_counts = {k: labels.count(k) for k in set(labels)}
        class_stats_str = "\n".join([f"- {k}: {v}" for k, v in class_counts.items()])

        tokens = nlp_core.tokenize_text(all_text)
        clean_tokens = [t.lower() for t in tokens if t not in '.,!?;:()[]"\'-']
        unique_tokens = set(clean_tokens)

        bigrams = list(set(nlp_core.get_ngrams(clean_tokens, 2)))
        trigrams = list(set(nlp_core.get_ngrams(clean_tokens, 3)))

        response = (
            f"**DATASET STATISTICS**\n\n"
            f"Class distribution:\n{class_stats_str}\n\n"
            f"Unique tokens (Top 10 of {len(unique_tokens)}): {list(unique_tokens)[:10]}...\n\n"
            f"Unique 2-grams (Top 5): {bigrams[:5]}...\n"
            f"Unique 3-grams (Top 5): {trigrams[:5]}..."
        )
        bot.reply_to(message, response)

        p1 = lab1_visualizer.plot_most_common_words(clean_tokens)
        p2 = lab1_visualizer.plot_token_length_histogram(clean_tokens)
        p3 = lab1_visualizer.plot_wordcloud(all_text)

        for p in [p1, p2, p3]:
            if p and os.path.exists(p):
                with open(p, "rb") as photo:
                    bot.send_photo(message.chat.id, photo)

    except Exception as e:
        log_error("stats", e)
        bot.reply_to(message, "An error occurred while computing stats.")


# --- Lab 2 commands ---

@bot.message_handler(commands=["classify"])
def handle_classify(message):
    try:
        params = parse_params(message.text)

        dataset = params.get("dataset")
        if not dataset or dataset not in VALID_DATASETS:
            bot.reply_to(message,
                         f"Invalid or missing dataset. Allowed: {', '.join(sorted(VALID_DATASETS))}")
            return

        method = params.get("method")
        if not method:
            bot.reply_to(message, "Missing 'method' parameter.")
            return
        methods_list = [m.strip().lower() for m in method.split(",")]
        for m in methods_list:
            if m not in VALID_METHODS:
                bot.reply_to(message,
                             f"Unknown method '{m}'. Allowed: {', '.join(sorted(VALID_METHODS))}")
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
            f"Starting experiment:\n"
            f"  dataset    = {dataset}\n"
            f"  method     = {method}\n"
            f"  gridsearch = {gridsearch}\n"
            f"  runs       = {n_runs}\n\n"
            f"This may take several minutes. I will send progress updates.",
        )

        def _run():
            chat_id = message.chat.id
            try:
                def progress(msg):
                    bot.send_message(chat_id, f"[Progress] {msg}")

                summary = run_experiment(
                    dataset_name=dataset,
                    method_str=method,
                    gridsearch=gridsearch,
                    n_runs=n_runs,
                    progress_callback=progress,
                )
                bot.send_message(chat_id, summary)

                # Send word cloud as a preview
                wc_path = os.path.join(viz.PLOTS_DIR, "wordcloud_corpus.png")
                if os.path.exists(wc_path):
                    with open(wc_path, "rb") as f:
                        bot.send_photo(chat_id, f, caption="Word Cloud (corpus)")

            except Exception as e:
                log_error("classify_thread", e)
                bot.send_message(chat_id,
                                 f"Experiment failed: {type(e).__name__}. Check server logs.")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    except Exception as e:
        log_error("classify", e)
        bot.reply_to(message, "An error occurred while parsing your command.")


if __name__ == "__main__":
    print("Bot (Lab 1 + Lab 2) is starting...")
    bot.infinity_polling()
