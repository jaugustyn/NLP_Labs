"""Offline smoke test for Lab 3 core behavior."""

import os
import tempfile

import numpy as np
import pandas as pd

from lab3 import commands
from lab3 import data_loader
from lab3 import model_loader
from lab3 import visualizations
from lab3.sentiment_methods import predict_rule, train_sklearn_sentiment_model


def main():
    assert commands._parse_methods("rule,nb,rule,,rf") == ["rule", "nb", "rf"]

    label, confidence = predict_rule("Bardzo nie polecam tego filmu")
    assert label == "negatywny"
    assert confidence >= 0.5

    label, _ = predict_rule("Obsługa była świetna i bardzo pomocna")
    assert label == "pozytywny"

    with tempfile.TemporaryDirectory() as temp_dir:
        old_dataset_file = data_loader.CUSTOM_DATASET_FILE
        old_plots_dir = visualizations.PLOTS_DIR
        old_models_dir = model_loader.MODELS_DIR

        try:
            data_loader.CUSTOM_DATASET_FILE = os.path.join(
                temp_dir,
                "sentiment_dataset.csv",
            )
            visualizations.PLOTS_DIR = os.path.join(temp_dir, "lab3plots")
            model_loader.MODELS_DIR = os.path.join(temp_dir, "models")

            data_loader.add_record("Świetny produkt", "positive")
            data_loader.add_record("Zwykły dzień", "neutralny")
            data_loader.add_record("Nie polecam", "negatywny")

            texts, labels, label_names = data_loader.load_dataset("custom")
            assert len(texts) == 3
            assert label_names == ["pozytywny", "neutralny", "negatywny"]
            assert set(labels.tolist()) == {0, 1, 2}

            train_texts = [
                "great movie excellent acting",
                "good useful product",
                "bad broken product",
                "terrible boring movie",
            ]
            train_labels = np.array([1, 1, 0, 0])
            train_label_names = ["negative", "positive"]
            artifact = train_sklearn_sentiment_model(
                "nb",
                "custom",
                texts=train_texts,
                labels=train_labels,
                label_names=train_label_names,
                save=False,
            )
            preds = commands._batch_predict(
                "nb",
                ["excellent product", "broken movie"],
                train_label_names,
                "custom",
                {"nb": artifact},
            )
            assert preds.shape == (2,)

            results = pd.DataFrame(
                [
                    {
                        "dataset": "custom",
                        "method": "rule",
                        "accuracy": 0.75,
                        "precision": 0.7,
                        "recall": 0.8,
                        "macro_f1": 0.74,
                        "model_path": "",
                    }
                ]
            )
            chart = visualizations.plot_comparison(results, "custom")
            assert chart and os.path.exists(chart)

            os.makedirs(model_loader.MODELS_DIR, exist_ok=True)
            open(os.path.join(model_loader.MODELS_DIR, "lstm_custom.h5"), "wb").close()
            open(
                os.path.join(model_loader.MODELS_DIR, "lstm_custom_tokenizer.pkl"),
                "wb",
            ).close()
            open(
                os.path.join(model_loader.MODELS_DIR, "lstm_custom_label_encoder.pkl"),
                "wb",
            ).close()
            models = model_loader.list_models()
            assert models and models[0]["dataset"] == "custom"
            assert model_loader.find_model_for_method("lstm") == "custom"
        finally:
            data_loader.CUSTOM_DATASET_FILE = old_dataset_file
            visualizations.PLOTS_DIR = old_plots_dir
            model_loader.MODELS_DIR = old_models_dir

    print("Lab3 smoke OK")


if __name__ == "__main__":
    main()
