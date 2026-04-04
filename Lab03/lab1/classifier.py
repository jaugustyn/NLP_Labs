from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lab1 import data_manager

CLASS_TO_LABEL = {
    "pozytywny": 1,
    "neutralny": 0,
    "negatywny": -1,
}

LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}

def train_and_predict(new_text):
    records = data_manager.load_records()
    if not records:
        return "No data available. Use /task or /full_pipeline first to build the dataset."

    texts = []
    labels = []

    for r in records:
        text = r.get("text")
        label = str(r.get("class", "")).strip().lower()

        if label not in CLASS_TO_LABEL:
            return (
                f"Unknown label: {label}. "
                "Allowed: pozytywny, neutralny, negatywny."
            )

        texts.append(text)
        labels.append(CLASS_TO_LABEL[label])

    if len(set(labels)) < 2:
        return "Too few unique classes in dataset (at least 2 required). Add more examples with a different class."

    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(texts, labels)

    prediction = model.predict([new_text])[0]
    return LABEL_TO_CLASS.get(prediction, f"unknown_class_{prediction}")
