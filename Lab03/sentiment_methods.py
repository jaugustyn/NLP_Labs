import numpy as np

from preprocessing import clean_text, texts_to_padded
from model_loader import (
    load_neural_model, load_sklearn_model, save_sklearn_model,
    find_model_for_method,
)

# ===================== 1. Rule-based =====================

_POS_EN = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "best", "awesome", "perfect", "beautiful", "happy", "enjoy",
    "recommend", "outstanding", "brilliant", "superb", "pleased",
    "impressive", "favorite", "delightful", "incredible",
}
_NEG_EN = {
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "disappointing", "useless", "waste", "broken", "damaged", "angry",
    "disgusting", "boring", "mediocre", "dreadful", "pathetic",
    "annoying", "frustrated", "garbage", "rubbish",
}
_POS_PL = {
    "świetny", "super", "doskonały", "wspaniały", "cudowny", "fantastyczny",
    "kocham", "najlepszy", "piękny", "szczęśliwy", "polecam", "rewelacyjny",
    "genialny", "zadowolony", "uwielbiam", "idealny", "perfekcyjny",
    "znakomity", "wspaniale", "świetnie", "doskonale",
}
_NEG_PL = {
    "zły", "okropny", "fatalny", "straszny", "najgorszy", "nienawidzę",
    "słaby", "rozczarowany", "bezużyteczny", "uszkodzony", "nudny",
    "beznadziejny", "tragiczny", "niezadowolony", "kiepski", "marny",
    "skandaliczny", "żałosny", "fatalnie", "okropnie",
}
_ALL_POS = _POS_EN | _POS_PL
_ALL_NEG = _NEG_EN | _NEG_PL


def predict_rule(text):
    words = set(text.lower().split())
    pos = len(words & _ALL_POS)
    neg = len(words & _ALL_NEG)
    score = pos - neg
    if score > 0:
        return "pozytywny", round(min(1.0, score / 5), 4)
    elif score < 0:
        return "negatywny", round(min(1.0, abs(score) / 5), 4)
    return "neutralny", 0.5


# ===================== 2. Naive Bayes =====================

def predict_nb(text, dataset_name="imdb"):
    """Predict with Naive Bayes. Loads saved model or trains on-the-fly."""
    saved = load_sklearn_model("nb", dataset_name)
    if saved:
        vec = saved["vectorizer"]
        model = saved["model"]
        label_names = saved["label_names"]
    else:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from data_loader import load_dataset

        texts, labels, label_names = load_dataset(dataset_name)
        vec = TfidfVectorizer(max_features=10000)
        X = vec.fit_transform(texts)
        model = MultinomialNB()
        model.fit(X, labels)
        save_sklearn_model("nb", dataset_name, {
            "vectorizer": vec, "model": model, "label_names": label_names,
        })

    X_new = vec.transform([clean_text(text)])
    pred = model.predict(X_new)[0]
    proba = float(model.predict_proba(X_new).max())
    return label_names[pred], round(proba, 4)


# ===================== 2b. Random Forest =====================

def predict_rf(text, dataset_name="imdb"):
    """Predict with Random Forest. Loads saved model or trains on-the-fly."""
    saved = load_sklearn_model("rf", dataset_name)
    if saved:
        vec = saved["vectorizer"]
        model = saved["model"]
        label_names = saved["label_names"]
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from data_loader import load_dataset

        texts, labels, label_names = load_dataset(dataset_name)
        vec = TfidfVectorizer(max_features=10000)
        X = vec.fit_transform(texts)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, labels)
        save_sklearn_model("rf", dataset_name, {
            "vectorizer": vec, "model": model, "label_names": label_names,
        })

    X_new = vec.transform([clean_text(text)])
    pred = model.predict(X_new)[0]
    proba = float(model.predict_proba(X_new).max())
    return label_names[pred], round(proba, 4)


# ===================== 3. Transformer =====================

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
    score = round(result["score"], 4)
    if label == "positive":
        return "pozytywny", score
    return "negatywny", score


# ===================== 4. TextBlob =====================

def predict_textblob(text):
    from textblob import TextBlob

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "pozytywny", round(abs(polarity), 4)
    elif polarity < -0.1:
        return "negatywny", round(abs(polarity), 4)
    return "neutralny", round(1.0 - abs(polarity), 4)


# ===================== 5. Stanza =====================

_stanza_pipe = None


def predict_stanza(text):
    global _stanza_pipe
    if _stanza_pipe is None:
        import stanza
        try:
            _stanza_pipe = stanza.Pipeline(
                "en", processors="tokenize,sentiment", verbose=False,
            )
        except Exception:
            stanza.download("en", processors="tokenize,sentiment", verbose=False)
            _stanza_pipe = stanza.Pipeline(
                "en", processors="tokenize,sentiment", verbose=False,
            )

    doc = _stanza_pipe(text[:1000])
    sentiments = [s.sentiment for s in doc.sentences]
    if not sentiments:
        return "neutralny", 0.5

    avg = np.mean(sentiments)  # 0=neg, 1=neutral, 2=pos
    if avg > 1.5:
        return "pozytywny", round(min(1.0, avg - 1.0), 4)
    elif avg < 0.5:
        return "negatywny", round(min(1.0, 1.0 - avg), 4)
    return "neutralny", round(1.0 - abs(avg - 1.0), 4)


# ===================== 6-8. Neural (SimpleRNN / LSTM / GRU) =====================

def predict_neural(text, model_type, dataset_name):
    """Predict with a saved Keras model."""
    model, tokenizer, le, meta = load_neural_model(model_type, dataset_name)
    max_len = meta.get("max_len", 200)
    num_classes = meta.get("num_classes", 2)

    cleaned = clean_text(text)
    X = texts_to_padded(tokenizer, [cleaned], max_len)
    pred = model.predict(X, verbose=0)

    if num_classes == 2:
        prob = float(pred[0][0])
        idx = 1 if prob > 0.5 else 0
        confidence = prob if prob > 0.5 else 1.0 - prob
    else:
        idx = int(np.argmax(pred[0]))
        confidence = float(pred[0][idx])

    label = le.inverse_transform([idx])[0]
    return label, round(confidence, 4)


# ===================== Dispatcher =====================

def predict_sentiment(method, text, dataset_name=None):
    """Route prediction to the correct sentiment method."""
    if method == "rule":
        return predict_rule(text)
    elif method == "nb":
        return predict_nb(text, dataset_name or "imdb")
    elif method == "rf":
        return predict_rf(text, dataset_name or "imdb")
    elif method == "transformer":
        return predict_transformer(text)
    elif method == "textblob":
        return predict_textblob(text)
    elif method == "stanza":
        return predict_stanza(text)
    elif method in ("simplernn", "lstm", "gru"):
        if not dataset_name:
            dataset_name = find_model_for_method(method)
            if not dataset_name:
                raise FileNotFoundError(
                    f"No trained {method.upper()} model found.\n"
                    f"Train one first: /train model={method} dataset=<amazon|imdb|custom>"
                )
        return predict_neural(text, method, dataset_name)
    else:
        raise ValueError(f"Unknown method: {method}")
