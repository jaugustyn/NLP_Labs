import os
import threading
import traceback

import telebot
from dotenv import load_dotenv

from experiment import run_experiment

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    raise ValueError("Missing token! Make sure .env contains TELEGRAM_BOT_TOKEN.")

bot = telebot.TeleBot(TOKEN)

VALID_DATASETS = {"20news_group", "imdb", "amazon", "ag_news"}
VALID_METHODS = {"nb", "rf", "mlp", "logreg", "all"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_params(text):
    """Parse key=value pairs from the command text."""
    params = {}
    for part in text.split():
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip().lower()] = value.strip()
    return params


def _log_error(context, error):
    print(f"[ERROR] {context}: {type(error).__name__}: {error}")
    print(traceback.format_exc())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    help_text = (
        "NLP Classification Bot (Lab 2)\n\n"
        "Command:\n"
        "/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>\n\n"
        "Parameters:\n"
        "  dataset   : 20news_group | imdb | amazon | ag_news\n"
        "  method    : nb | rf | mlp | logreg | all  (comma-separated OK)\n"
        "  gridsearch: true | false\n"
        "  run       : 1-3  (number of runs with different seeds)\n\n"
        "Examples:\n"
        "/classify dataset=20news_group method=all gridsearch=false run=1\n"
        "/classify dataset=imdb method=logreg gridsearch=true run=2\n"
        "/classify dataset=ag_news method=rf,nb gridsearch=false run=3"
    )
    bot.reply_to(message, help_text)


@bot.message_handler(commands=["classify"])
def handle_classify(message):
    try:
        params = _parse_params(message.text)

        # --- validate dataset ---
        dataset = params.get("dataset")
        if not dataset or dataset not in VALID_DATASETS:
            bot.reply_to(message,
                         f"Invalid or missing dataset. Allowed: {', '.join(sorted(VALID_DATASETS))}")
            return

        # --- validate method ---
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

        # --- validate gridsearch ---
        gs_raw = params.get("gridsearch", "false").lower()
        if gs_raw not in ("true", "false"):
            bot.reply_to(message, "gridsearch must be 'true' or 'false'.")
            return
        gridsearch = gs_raw == "true"

        # --- validate run ---
        try:
            n_runs = int(params.get("run", "1"))
            if n_runs < 1 or n_runs > 3:
                raise ValueError
        except ValueError:
            bot.reply_to(message, "run must be 1, 2 or 3.")
            return

        # --- acknowledge & start in background ---
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
                wc_path = os.path.join("lab2plots", "wordcloud_corpus.png")
                if os.path.exists(wc_path):
                    with open(wc_path, "rb") as f:
                        bot.send_photo(chat_id, f, caption="Word Cloud (corpus)")

            except Exception as e:
                _log_error("classify_thread", e)
                bot.send_message(chat_id,
                                 f"Experiment failed: {type(e).__name__}. Check server logs.")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    except Exception as e:
        _log_error("classify", e)
        bot.reply_to(message, "An error occurred while parsing your command.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Bot (Lab 2) is starting...")
    bot.infinity_polling()
