"""Offline smoke test for Lab 1 core behavior."""

import json
import os
import tempfile

from lab1 import classifier
from lab1 import data_manager
from lab1 import nlp_core
from lab1 import visualizer


class _Chat:
    id = 1


class _Message:
    def __init__(self, text):
        self.text = text
        self.chat = _Chat()


class _Bot:
    def __init__(self):
        self.replies = []
        self.photos = []

    def reply_to(self, message, text):
        self.replies.append(text)

    def send_photo(self, chat_id, file_obj):
        self.photos.append(getattr(file_obj, "name", ""))


def _set_temp_paths(temp_dir):
    data_manager.DATA_FILE = os.path.join(temp_dir, "sentences.json")
    visualizer.PLOTS_DIR = os.path.join(temp_dir, "plots")


def _assert_record_shape(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    assert rows
    assert all(set(row) == {"text", "class"} for row in rows)


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        _set_temp_paths(temp_dir)

        data_manager.save_record("kocham ten film", "pozytywny")
        data_manager.save_record("fantastyczna historia", "pozytywny")
        data_manager.save_record("to byl okropny film", "negatywny")
        data_manager.save_record("nienawidze tego", "negatywny")

        rows = data_manager.load_records()
        assert len(rows) == 4
        _assert_record_shape(data_manager.DATA_FILE)

        prediction = classifier.train_and_predict("fantastyczny film")
        assert prediction in {"pozytywny", "neutralny", "negatywny"}

        text = "System dziala szybko, ale interfejs wymaga poprawy."
        cleaned = nlp_core.clean_text(text)
        tokens = nlp_core.clean_tokens(nlp_core.tokenize_text(cleaned))
        assert cleaned
        assert tokens
        assert nlp_core.get_ngrams(tokens, 2)
        assert nlp_core.get_ngrams(tokens, 3)

        for path in (
            visualizer.plot_most_common_words(tokens),
            visualizer.plot_token_length_histogram(tokens),
            visualizer.plot_wordcloud(cleaned),
        ):
            assert path and os.path.exists(path) and path.endswith(".png")

        from lab1 import commands

        data_manager.DATA_FILE = os.path.join(temp_dir, "handler_sentences.json")
        bot = _Bot()
        commands._handle_full_pipeline(
            bot,
            _Message('/full_pipeline "Pierwsze zdanie. Drugie zdanie." "neutralny"'),
        )
        assert len(data_manager.load_records()) == 2
        assert len(bot.photos) == 3

        stats_bot = _Bot()
        commands._handle_stats(stats_bot, _Message("/stats"))
        assert "Unique 2-grams" in stats_bot.replies[-1]
        assert "Unique 3-grams" in stats_bot.replies[-1]

    print("Lab1 smoke OK")


if __name__ == "__main__":
    main()
