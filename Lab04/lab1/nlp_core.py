import string
import os
from collections import Counter

import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_NLP = None


STEM_SUFFIXES = sorted([
    "owego","owej","owym","owie",
    "anie","enie","aniu","eniu",
    "ami","ach","ego","emu","owi",
    "owa","owe","ych","ymi",
    "nia","nie","cie",
    "ów","om","ie",
    "ą","ę","a","e","i","y","u"
], key=len, reverse=True)

PUNCT = set(string.punctuation)

def get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("pl_core_news_sm", disable=["parser", "ner"])
        except OSError:
            # Keep bot functional even when full model is missing
            _NLP = spacy.blank("pl")
    return _NLP


_DIR = os.path.dirname(os.path.abspath(__file__))

def load_stopwords():
    filepath = os.path.join(_DIR, "stopwords-pl.txt")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


PL_STOPWORDS = load_stopwords()


def tokenize_text(text):
    doc = get_nlp()(text)
    return [token.text for token in doc if not token.is_space]


def remove_stopwords(tokens):
    return [
        t for t in tokens
        if t.lower() not in PL_STOPWORDS and t not in PUNCT
    ]


def lemmatize(text):
    doc = get_nlp()(text)
    lemmas = []
    for token in doc:
        if token.is_punct:
            continue
        lemma = token.lemma_.strip() if token.lemma_ else ""
        lemmas.append(lemma if lemma else token.text.lower())
    return lemmas


def stem_token(token):
    value = token.lower().strip()
    if not value or value in string.punctuation:
        return value

    for suffix in STEM_SUFFIXES:
        if len(value) - len(suffix) >= 3 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def stemming(tokens):
    return [stem_token(t) for t in tokens if t not in PUNCT]


def get_stats(tokens):
    return Counter(tokens).most_common(10)


def get_ngrams(tokens, n=2):
    return list(nltk.ngrams(tokens, n))


def bag_of_words(text):
    try:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out(), X.toarray()
    except ValueError:
        return [], []


def tf_idf(text):
    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out(), X.toarray()
    except ValueError:
        return [], []


def run_task(task_name, text):
    tokens = tokenize_text(text)

    if task_name == "tokenize":
        return tokens

    tokens = remove_stopwords(tokens)

    if task_name == "remove_stopwords":
        return tokens
    elif task_name == "lemmatize":
        return lemmatize(text)
    elif task_name == "stemming":
        return stemming(tokens)
    elif task_name == "stats":
        return get_stats(tokens)
    elif task_name == "n-grams":
        return get_ngrams(tokens, 2)
    else:
        return None