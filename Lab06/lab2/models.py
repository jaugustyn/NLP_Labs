"""Classifier factory helpers for Lab 2."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier


ALL_MODELS = ("nb", "rf", "mlp", "logreg")

MODEL_DISPLAY_NAMES = {
    "nb": "Naive Bayes",
    "rf": "Random Forest",
    "mlp": "MLP",
    "logreg": "Logistic Regression",
}

GRID_PARAMS = {
    "nb_multinomial": {"alpha": [0.1, 0.5, 1.0]},
    "nb_gaussian": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    "rf": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
    "mlp": {"hidden_layer_sizes": [(128,), (256, 128)]},
    "logreg": {"C": [0.1, 1, 10]},
}


def get_model(name, embedding_type="bow", seed=42):
    if name == "nb":
        if embedding_type in ("word2vec", "glove"):
            return GaussianNB()
        return MultinomialNB()
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=300,
            random_state=seed,
            early_stopping=True,
        )
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(f"Unknown model: {name}. Available: {ALL_MODELS}")


def get_grid_params(name, embedding_type="bow"):
    if name == "nb":
        if embedding_type in ("word2vec", "glove"):
            return GRID_PARAMS["nb_gaussian"]
        return GRID_PARAMS["nb_multinomial"]
    return GRID_PARAMS.get(name, {})


def resolve_methods(method_str):
    raw_methods = [method.strip().lower() for method in method_str.split(",")]
    methods = [method for method in raw_methods if method]
    if not methods:
        raise ValueError("No model selected.")

    if "all" in methods:
        if len(methods) > 1:
            raise ValueError("'all' cannot be combined with explicit model names.")
        return list(ALL_MODELS)

    unique_methods = list(dict.fromkeys(methods))
    for method in unique_methods:
        if method not in ALL_MODELS:
            raise ValueError(f"Unknown method '{method}'. Available: {ALL_MODELS}")
    return unique_methods
