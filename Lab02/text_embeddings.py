import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
        self.vectorizer = TfidfVectorizer(max_features=max_features)

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
    # Document vector = mean of word vectors.

    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    @staticmethod
    def _tokenize(texts):
        return [text.lower().split() for text in texts]

    def fit(self, texts):
        from gensim.models import Word2Vec
        tokenized = self._tokenize(texts)
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            seed=42,
        )
        return self

    def transform(self, texts):
        tokenized = self._tokenize(texts)
        results = []
        for tokens in tokenized:
            vecs = [self.model.wv[w] for w in tokens if w in self.model.wv]
            if vecs:
                results.append(np.mean(vecs, axis=0))
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
    # Pre-trained GloVe via gensim. Document vector = mean of word vectors.

    _cached_model = None

    def __init__(self, model_name="glove-wiki-gigaword-100"):
        self.model_name = model_name
        self.model = None
        self.vector_size = None

    def fit(self, texts):
        if GloveEmbedding._cached_model is None:
            import gensim.downloader as api
            print(f"Downloading GloVe model '{self.model_name}' (first time only)...")
            GloveEmbedding._cached_model = api.load(self.model_name)
        self.model = GloveEmbedding._cached_model
        self.vector_size = self.model.vector_size
        return self

    def transform(self, texts):
        results = []
        for text in texts:
            words = text.lower().split()
            vecs = [self.model[w] for w in words if w in self.model]
            if vecs:
                results.append(np.mean(vecs, axis=0))
            else:
                results.append(np.zeros(self.vector_size))
        return np.array(results)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return [f"glove_dim_{i}" for i in range(self.vector_size)]

    def get_word_vectors(self):
        return self.model


EMBEDDING_CLASSES = {
    "bow": BowEmbedding,
    "tfidf": TfidfEmbedding,
    "word2vec": Word2VecEmbedding,
    "glove": GloveEmbedding,
}

EMBEDDING_NAMES = list(EMBEDDING_CLASSES.keys())

def get_embedding(name):
    cls = EMBEDDING_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown embedding: {name}. Available: {EMBEDDING_NAMES}")
    return cls()
