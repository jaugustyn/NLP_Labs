"""Lab 2 command handlers."""

import os
import threading

from lab2 import visualizer as lab2_visualizer
from lab2.experiment import run_experiment as run_lab2_experiment
from utils import log_error, parse_params


LAB2_DATASETS = {"20news_group", "imdb", "amazon", "ag_news"}
LAB2_METHODS = {"nb", "rf", "mlp", "logreg", "all"}

HELP_SECTION = (
    "--- Lab 2 ---\n"
    "/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>\n"
    "  Datasets: 20news_group, imdb, amazon, ag_news\n"
    "  Methods: nb, rf, mlp, logreg, all\n"
)


def register_handlers(bot):
    @bot.message_handler(commands=["classify"])
    def cmd_classify(message):
        _handle_classify(bot, message)


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
        for method_name in methods_list:
            if method_name not in LAB2_METHODS:
                bot.reply_to(
                    message,
                    f"Unknown method '{method_name}'. Allowed: {', '.join(sorted(LAB2_METHODS))}"
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
            "This may take several minutes.",
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
                    lab2_visualizer.PLOTS_DIR,
                    "wordcloud_corpus.png",
                )
                if os.path.exists(wc_path):
                    with open(wc_path, "rb") as f:
                        bot.send_photo(chat_id, f, caption="Word Cloud (corpus)")

            except Exception as e:
                log_error("classify_thread", e)
                bot.send_message(
                    chat_id,
                    f"Experiment failed: {type(e).__name__}. Check server logs.",
                )

        threading.Thread(target=_run, daemon=True).start()

    except Exception as e:
        log_error("classify", e)
        bot.reply_to(message, "An error occurred while parsing your command.")


__all__ = ["HELP_SECTION", "register_handlers"]
