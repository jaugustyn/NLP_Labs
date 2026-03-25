from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

ALL_MODELS = ["nb", "rf", "mlp", "logreg"]

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
    elif name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    elif name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)
    elif name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {name}. Available: {ALL_MODELS}")

def get_grid_params(name, embedding_type="bow"):
    if name == "nb":
        if embedding_type in ("word2vec", "glove"):
            return GRID_PARAMS["nb_gaussian"]
        return GRID_PARAMS["nb_multinomial"]
    return GRID_PARAMS.get(name, {})

def resolve_methods(method_str):
    if method_str.strip().lower() == "all":
        return list(ALL_MODELS)
    methods = [m.strip().lower() for m in method_str.split(",")]
    for m in methods:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown method '{m}'. Available: {ALL_MODELS}")
    return methods
