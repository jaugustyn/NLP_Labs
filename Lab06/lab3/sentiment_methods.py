"""Sentiment prediction methods used by Lab 3."""

import re
import unicodedata

import numpy as np

from lab3.model_loader import (
    find_model_for_method,
    load_neural_model,
    load_sklearn_model,
    save_sklearn_model,
)
from lab3.preprocessing import clean_text, texts_to_padded


TOKEN_RE = re.compile(r"(?u)\b\w+\b")

POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "brilliant",
    "cudowny",
    "delightful",
    "dobra",
    "dobre",
    "dobry",
    "doskonala",
    "doskonale",
    "doskonaly",
    "excellent",
    "fantastic",
    "fantastyczna",
    "fantastyczne",
    "fantastyczny",
    "favorite",
    "genialny",
    "good",
    "great",
    "happy",
    "idealny",
    "impressive",
    "kocham",
    "love",
    "najlepszy",
    "outstanding",
    "perfect",
    "perfekcyjny",
    "pieknie",
    "piekna",
    "piekne",
    "piekny",
    "pleased",
    "pomocna",
    "pomocny",
    "polecam",
    "recommend",
    "rewelacyjny",
    "super",
    "superb",
    "swietna",
    "swietne",
    "swietnie",
    "swietny",
    "uwielbiam",
    "wspaniale",
    "wspanialy",
    "zadowolony",
    "znakomity",
}

NEGATIVE_WORDS = {
    "angry",
    "annoying",
    "awful",
    "bad",
    "beznadziejny",
    "bezuzyteczny",
    "boring",
    "broken",
    "damaged",
    "disappointing",
    "dreadful",
    "disgusting",
    "fatalnie",
    "fatalny",
    "frustrated",
    "garbage",
    "hate",
    "horrible",
    "kiepski",
    "marny",
    "mediocre",
    "najgorszy",
    "nienawidze",
    "niezadowolony",
    "nudny",
    "okropnie",
    "okropny",
    "pathetic",
    "poor",
    "rozczarowany",
    "rubbish",
    "skandaliczny",
    "slaby",
    "straszny",
    "terrible",
    "tragiczna",
    "tragiczne",
    "tragiczny",
    "useless",
    "uszkodzony",
    "waste",
    "weak",
    "worst",
    "zalosny",
    "zly",
}

NEGATION_WORDS = {"nie", "not", "never", "no", "bez"}


def _normalize(value):
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return value.lower()


def _tokens(text):
    return TOKEN_RE.findall(_normalize(text))


def predict_rule(text):
    tokens = _tokens(text)
    if not tokens:
        return "neutralny", 0.5

    positive = 0
    negative = 0
    for index, token in enumerate(tokens):
        is_positive = token in POSITIVE_WORDS
        is_negative = token in NEGATIVE_WORDS
        if not is_positive and not is_negative:
            continue

        window = tokens[max(0, index - 2) : index]
        negated = any(word in NEGATION_WORDS for word in window)
        if is_positive and not negated:
            positive += 1
        elif is_positive and negated:
            negative += 1
        elif is_negative and not negated:
            negative += 1
        else:
            positive += 1

    score = positive - negative
    total_hits = positive + negative
    confidence = round(min(1.0, 0.5 + abs(score) / max(2, total_hits * 2)), 4)
    if score > 0:
        return "pozytywny", confidence
    if score < 0:
        return "negatywny", confidence
    return "neutralny", 0.5


def train_sklearn_sentiment_model(
    model_name,
    dataset_name,
    texts=None,
    labels=None,
    label_names=None,
    save=True,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    from lab3.data_loader import load_dataset

    if texts is None or labels is None or label_names is None:
        texts, labels, label_names = load_dataset(dataset_name)

    cleaned = [clean_text(text) for text in texts]
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned)

    if model_name == "nb":
        model = MultinomialNB()
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported sklearn sentiment model: {model_name}")

    model.fit(X, labels)
    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "label_names": list(label_names),
    }
    if save:
        save_sklearn_model(model_name, dataset_name, artifact)
    return artifact


def _predict_sklearn(text, artifact):
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    label_names = artifact["label_names"]
    X_new = vectorizer.transform([clean_text(text)])
    prediction = int(model.predict(X_new)[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X_new).max())
    label = label_names[prediction] if prediction < len(label_names) else str(prediction)
    return label, round(confidence, 4)


def predict_nb(text, dataset_name="imdb"):
    saved = load_sklearn_model("nb", dataset_name)
    if saved is None:
        saved = train_sklearn_sentiment_model("nb", dataset_name)
    return _predict_sklearn(text, saved)


def predict_rf(text, dataset_name="imdb"):
    saved = load_sklearn_model("rf", dataset_name)
    if saved is None:
        saved = train_sklearn_sentiment_model("rf", dataset_name)
    return _predict_sklearn(text, saved)


_transformer_pipe = None


def predict_transformer(text):
    global _transformer_pipe
    if _transformer_pipe is None:
        from transformers import pipeline

        _transformer_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    result = _transformer_pipe(text[:512])[0]
    label = result["label"].lower()
    score = round(float(result["score"]), 4)
    if label == "positive":
        return "pozytywny", score
    return "negatywny", score


def predict_textblob(text):
    from textblob import TextBlob

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "pozytywny", round(abs(polarity), 4)
    if polarity < -0.1:
        return "negatywny", round(abs(polarity), 4)
    return "neutralny", round(1.0 - abs(polarity), 4)


_stanza_pipe = None


def predict_stanza(text):
    global _stanza_pipe
    if _stanza_pipe is None:
        import stanza

        try:
            _stanza_pipe = stanza.Pipeline(
                "en",
                processors="tokenize,sentiment",
                verbose=False,
            )
        except Exception:
            stanza.download("en", processors="tokenize,sentiment", verbose=False)
            _stanza_pipe = stanza.Pipeline(
                "en",
                processors="tokenize,sentiment",
                verbose=False,
            )

    doc = _stanza_pipe(text[:1000])
    sentiments = [sentence.sentiment for sentence in doc.sentences]
    if not sentiments:
        return "neutralny", 0.5

    average = float(np.mean(sentiments))
    if average > 1.5:
        return "pozytywny", round(min(1.0, average - 1.0), 4)
    if average < 0.5:
        return "negatywny", round(min(1.0, 1.0 - average), 4)
    return "neutralny", round(1.0 - abs(average - 1.0), 4)


def predict_neural(text, model_type, dataset_name):
    model, tokenizer, label_encoder, meta = load_neural_model(model_type, dataset_name)
    max_len = meta.get("max_len", 200)
    num_classes = meta.get("num_classes", 2)

    cleaned = clean_text(text)
    X = texts_to_padded(tokenizer, [cleaned], max_len)
    prediction = model.predict(X, verbose=0)

    if num_classes == 2:
        probability = float(prediction[0][0])
        label_index = 1 if probability > 0.5 else 0
        confidence = probability if probability > 0.5 else 1.0 - probability
    else:
        label_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][label_index])

    label = label_encoder.inverse_transform([label_index])[0]
    return label, round(confidence, 4)


def predict_sentiment(method, text, dataset_name=None):
    if method == "rule":
        return predict_rule(text)
    if method == "nb":
        return predict_nb(text, dataset_name or "imdb")
    if method == "rf":
        return predict_rf(text, dataset_name or "imdb")
    if method == "transformer":
        return predict_transformer(text)
    if method == "textblob":
        return predict_textblob(text)
    if method == "stanza":
        return predict_stanza(text)
    if method in ("simplernn", "lstm", "gru"):
        if not dataset_name:
            dataset_name = find_model_for_method(method)
            if not dataset_name:
                raise FileNotFoundError(
                    f"No trained {method.upper()} model found.\n"
                    f"Train one first: /train model={method} dataset=<amazon|imdb|custom>"
                )
        return predict_neural(text, method, dataset_name)
    raise ValueError(f"Unknown method: {method}")
