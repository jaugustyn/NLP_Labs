import glob
import os
import pickle

from lab3.config import MODELS_DIR


def _load_trusted_pickle(path):
    """Load a pickle artifact generated locally by this project."""
    with open(path, "rb") as file:
        return pickle.load(file)


def _save_pickle(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load_neural_model(model_type, dataset_name):
    """Load a locally trained neural model and trusted pickle artifacts."""
    prefix = f"{model_type}_{dataset_name}"
    model_path = os.path.join(MODELS_DIR, f"{prefix}.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {prefix}.h5\n"
            f"Train it first: /train model={model_type} dataset={dataset_name}"
        )

    tokenizer_path = os.path.join(MODELS_DIR, f"{prefix}_tokenizer.pkl")
    encoder_path = os.path.join(MODELS_DIR, f"{prefix}_label_encoder.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{prefix}_meta.pkl")
    for required_path in (tokenizer_path, encoder_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"Missing artifact for {prefix}: {os.path.basename(required_path)}\n"
                f"Train it again: /train model={model_type} dataset={dataset_name}"
            )

    from tensorflow.keras.models import load_model

    model = load_model(model_path)

    tokenizer = _load_trusted_pickle(tokenizer_path)
    label_encoder = _load_trusted_pickle(encoder_path)

    meta = {}
    if os.path.exists(meta_path):
        meta = _load_trusted_pickle(meta_path)

    return model, tokenizer, label_encoder, meta


def load_sklearn_model(model_name, dataset_name):
    """Load a locally trained sklearn artifact.

    Pickle is intentionally limited to artifacts created by this project.
    Do not place untrusted files in the models directory.
    """
    path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}_sklearn.pkl")
    if not os.path.exists(path):
        return None
    return _load_trusted_pickle(path)


def save_sklearn_model(model_name, dataset_name, data):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}_sklearn.pkl")
    _save_pickle(path, data)


def list_models():
    if not os.path.exists(MODELS_DIR):
        return []

    models = []

    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5"))):
        name = os.path.splitext(os.path.basename(path))[0]
        parts = name.split("_", 1)
        model_type = parts[0] if parts else "unknown"
        dataset = parts[1] if len(parts) > 1 else "unknown"
        tokenizer = os.path.exists(os.path.join(MODELS_DIR, f"{name}_tokenizer.pkl"))
        encoder = os.path.exists(os.path.join(MODELS_DIR, f"{name}_label_encoder.pkl"))
        models.append(
            {
                "name": name,
                "file": os.path.basename(path),
                "model_type": model_type,
                "dataset": dataset,
                "format": "h5",
                "tokenizer": tokenizer,
                "encoder": encoder,
            }
        )

    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*_sklearn.pkl"))):
        name = os.path.basename(path).replace("_sklearn.pkl", "")
        parts = name.split("_", 1)
        model_type = parts[0] if parts else "unknown"
        dataset = parts[1] if len(parts) > 1 else "unknown"
        models.append(
            {
                "name": name,
                "file": os.path.basename(path),
                "model_type": model_type,
                "dataset": dataset,
                "format": "pkl",
                "tokenizer": True,
                "encoder": True,
            }
        )

    return models


def find_model_for_method(method, preferred_dataset=None):
    models = list_models()
    matching = [
        model
        for model in models
        if model["model_type"] == method and model["format"] == "h5"
    ]
    if not matching:
        return None
    if preferred_dataset:
        for m in matching:
            if m["dataset"] == preferred_dataset:
                return m["dataset"]
    return matching[0]["dataset"]
