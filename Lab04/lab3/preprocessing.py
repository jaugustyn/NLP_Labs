import re


def clean_text(text):
    """Clean text: remove HTML tags, URLs, normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def build_tokenizer(texts, max_vocab_size=20000):
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded(tokenizer, texts, max_len=200):
    """Convert texts to padded integer sequences."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
