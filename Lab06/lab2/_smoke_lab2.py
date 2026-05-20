"""Offline smoke test for Lab 2 experiment flow."""

import csv
import os
import tempfile

import numpy as np

from lab2 import experiment
from lab2 import visualizer
from lab2.models import resolve_methods


def _toy_dataset(_name):
    texts = [
        "great movie with excellent story",
        "good acting and strong plot",
        "excellent film good music",
        "great story and good cast",
        "bad movie with weak story",
        "terrible acting and boring plot",
        "awful film bad music",
        "weak story and terrible cast",
        "good excellent enjoyable movie",
        "bad awful boring movie",
    ]
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    return texts, labels, ["negative", "positive"]


def main():
    try:
        resolve_methods("all,nb")
        raise AssertionError("resolve_methods should reject mixed 'all' usage")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as temp_dir:
        old_loader = experiment.load_dataset
        old_embeddings = experiment.EMBEDDING_NAMES
        old_results_dir = experiment.RESULTS_DIR
        old_results_file = experiment.RESULTS_FILE
        old_similar_words_file = experiment.SIMILAR_WORDS_FILE
        old_feature_importance_file = experiment.FEATURE_IMPORTANCE_FILE
        old_plots_dir = visualizer.PLOTS_DIR

        try:
            experiment.load_dataset = _toy_dataset
            experiment.EMBEDDING_NAMES = ("bow", "tfidf")
            experiment.RESULTS_DIR = os.path.join(temp_dir, "results")
            experiment.RESULTS_FILE = os.path.join(
                experiment.RESULTS_DIR,
                "lab2results.csv",
            )
            experiment.SIMILAR_WORDS_FILE = os.path.join(
                experiment.RESULTS_DIR,
                "lab2_similar_words.txt",
            )
            experiment.FEATURE_IMPORTANCE_FILE = os.path.join(
                experiment.RESULTS_DIR,
                "lab2_feature_importance.txt",
            )
            visualizer.PLOTS_DIR = os.path.join(temp_dir, "lab2plots")

            summary = experiment.run_experiment(
                dataset_name="toy",
                method_str="nb,logreg",
                gridsearch=False,
                n_runs=1,
            )
            assert "EXPERIMENT RESULTS" in summary
            assert os.path.exists(experiment.RESULTS_FILE)
            assert os.path.exists(
                os.path.join(visualizer.PLOTS_DIR, "wordcloud_corpus.png")
            )
            assert os.path.exists(
                os.path.join(visualizer.PLOTS_DIR, "confusion_bow_nb.png")
            )

            with open(experiment.RESULTS_FILE, "r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            assert len(rows) == 4
            assert set(rows[0]) == {
                "embedding",
                "model",
                "accuracy",
                "macro_f1",
                "seed",
            }
        finally:
            experiment.load_dataset = old_loader
            experiment.EMBEDDING_NAMES = old_embeddings
            experiment.RESULTS_DIR = old_results_dir
            experiment.RESULTS_FILE = old_results_file
            experiment.SIMILAR_WORDS_FILE = old_similar_words_file
            experiment.FEATURE_IMPORTANCE_FILE = old_feature_importance_file
            visualizer.PLOTS_DIR = old_plots_dir

    print("Lab2 smoke OK")


if __name__ == "__main__":
    main()
