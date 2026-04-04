import os
import glob
import pickle

from config import MODELS_DIR


def load_neural_model(model_type, dataset_name):
    """Load a Keras model + tokenizer + label encoder from disk."""
    prefix = f"{model_type}_{dataset_name}"
    model_path = os.path.join(MODELS_DIR, f"{prefix}.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {prefix}.h5\n"
            f"Train it first: /train model={model_type} dataset={dataset_name}"
        )

    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    tok_path = os.path.join(MODELS_DIR, f"{prefix}_tokenizer.pkl")
    enc_path = os.path.join(MODELS_DIR, f"{prefix}_label_encoder.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{prefix}_meta.pkl")

    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(enc_path, "rb") as f:
        label_encoder = pickle.load(f)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    return model, tokenizer, label_encoder, meta


def load_sklearn_model(model_name, dataset_name):
    """Load a saved sklearn model dict from disk. Returns None if missing."""
    path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}_sklearn.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_sklearn_model(model_name, dataset_name, data):
    """Save sklearn model dict (vectorizer + model + label_names) as pickle."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{model_name}_{dataset_name}_sklearn.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def list_models():
    """List all saved models in models/ directory."""
    if not os.path.exists(MODELS_DIR):
        return []

    models = []

    # Neural models (.h5)
    for f in sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5"))):
        name = os.path.splitext(os.path.basename(f))[0]
        parts = name.split("_", 1)
        mtype = parts[0] if parts else "unknown"
        dset = parts[1] if len(parts) > 1 else "unknown"
        tok = os.path.exists(os.path.join(MODELS_DIR, f"{name}_tokenizer.pkl"))
        enc = os.path.exists(os.path.join(MODELS_DIR, f"{name}_label_encoder.pkl"))
        models.append({
            "name": name, "file": os.path.basename(f),
            "model_type": mtype, "dataset": dset, "format": "h5",
            "tokenizer": tok, "encoder": enc,
        })

    # Sklearn models (_sklearn.pkl)
    for f in sorted(glob.glob(os.path.join(MODELS_DIR, "*_sklearn.pkl"))):
        name = os.path.basename(f).replace("_sklearn.pkl", "")
        parts = name.split("_", 1)
        mtype = parts[0] if parts else "unknown"
        dset = parts[1] if len(parts) > 1 else "unknown"
        models.append({
            "name": name, "file": os.path.basename(f),
            "model_type": mtype, "dataset": dset, "format": "pkl",
            "tokenizer": True, "encoder": True,
        })

    return models


def find_model_for_method(method, preferred_dataset=None):
    """Find an available saved model for the given method. Returns dataset name or None."""
    models = list_models()
    matching = [m for m in models if m["model_type"] == method]
    if not matching:
        return None
    if preferred_dataset:
        for m in matching:
            if m["dataset"] == preferred_dataset:
                return m["dataset"]
    return matching[0]["dataset"]
