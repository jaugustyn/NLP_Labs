"""Text representation methods used by Lab 2."""

import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


class EmbeddingUnavailable(RuntimeError):
    """Raised when an optional embedding backend cannot be loaded."""


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


class BowEmbedding:
    def __init__(self, max_features=10000):
        self.vectorizer = CountVectorizer(max_features=max_features)

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def get_word_vectors(self):
        return None


class TfidfEmbedding:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def get_word_vectors(self):
        return None


class Word2VecEmbedding:
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    @staticmethod
    def _tokenize(texts):
        return [tokenize(text) for text in texts]

    def fit(self, texts):
        try:
            from gensim.models import Word2Vec
        except Exception as exc:
            raise EmbeddingUnavailable(f"word2vec backend unavailable: {exc}") from exc

        tokenized = self._tokenize(texts)
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=42,
        )
        return self

    def transform(self, texts):
        if self.model is None:
            raise ValueError("Word2VecEmbedding must be fitted before transform().")

        results = []
        for tokens in self._tokenize(texts):
            vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if vectors:
                results.append(np.mean(vectors, axis=0))
            else:
                results.append(np.zeros(self.vector_size))
        return np.array(results)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return [f"w2v_dim_{i}" for i in range(self.vector_size)]

    def get_word_vectors(self):
        return self.model.wv if self.model else None


class GloveEmbedding:
    _cached_model = None

    def __init__(self, model_name="glove-wiki-gigaword-100"):
        self.model_name = model_name
        self.model = None
        self.vector_size = None

    def fit(self, texts):
        try:
            if GloveEmbedding._cached_model is None:
                import gensim.downloader as api

                GloveEmbedding._cached_model = api.load(self.model_name)
            self.model = GloveEmbedding._cached_model
        except Exception as exc:
            raise EmbeddingUnavailable(f"GloVe model unavailable: {exc}") from exc

        self.vector_size = self.model.vector_size
        return self

    def transform(self, texts):
        if self.model is None:
            raise ValueError("GloveEmbedding must be fitted before transform().")

        results = []
        for text in texts:
            vectors = [self.model[word] for word in tokenize(text) if word in self.model]
            if vectors:
                results.append(np.mean(vectors, axis=0))
            else:
                results.append(np.zeros(self.vector_size))
        return np.array(results)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        if self.vector_size is None:
            return None
        return [f"glove_dim_{i}" for i in range(self.vector_size)]

    def get_word_vectors(self):
        return self.model


EMBEDDING_CLASSES = {
    "bow": BowEmbedding,
    "tfidf": TfidfEmbedding,
    "word2vec": Word2VecEmbedding,
    "glove": GloveEmbedding,
}

EMBEDDING_NAMES = tuple(EMBEDDING_CLASSES)


def get_embedding(name):
    cls = EMBEDDING_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown embedding: {name}. Available: {EMBEDDING_NAMES}")
    return cls()
