import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "lab3plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Files
RESULTS_FILE = os.path.join(RESULTS_DIR, "lab3results.csv")
CUSTOM_DATASET_FILE = os.path.join(BASE_DIR, "sentiment_dataset.csv")

# Neural model parameters
EMBEDDING_DIM = 100
MAX_VOCAB_SIZE = 20000
BATCH_SIZE = 32
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
MAX_LEN_OPTIONS = [100, 150, 200, 300]
DEFAULT_MAX_LEN = 200
MAX_SAMPLES = 5000

# Sentiment methods
SENTIMENT_METHODS = [
    "rule", "nb", "transformer", "textblob", "stanza",
    "simplernn", "lstm", "gru",
]
NEURAL_MODELS = ["simplernn", "lstm", "gru"]
ML_MODELS = ["nb"]

# Datasets
VALID_DATASETS = ["amazon", "imdb", "custom"]

# Custom dataset labels
CUSTOM_LABELS = ["pozytywny", "neutralny", "negatywny"]
LABEL_ALIASES = {
    "pozytywny": "pozytywny", "pozytywna": "pozytywny", "positive": "pozytywny",
    "neutralny": "neutralny", "neutralna": "neutralny", "neutral": "neutralny",
    "negatywny": "negatywny", "negatywna": "negatywny", "negative": "negatywny",
}

# Binary label mapping (for methods returning PL labels on EN datasets)
BINARY_LABEL_MAP = {
    "pozytywny": "positive", "positive": "positive",
    "negatywny": "negative", "negative": "negative",
    "neutralny": "negative",  # conservative mapping for binary datasets
}
