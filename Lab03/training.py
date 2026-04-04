import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    MODELS_DIR, EMBEDDING_DIM, MAX_VOCAB_SIZE, BATCH_SIZE,
    EPOCHS, EARLY_STOPPING_PATIENCE, DEFAULT_MAX_LEN,
)
from preprocessing import clean_text, build_tokenizer, texts_to_padded


def _build_model(model_type, vocab_size, embedding_dim, max_len, num_classes):
    """Build a Keras sequential model for text classification."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout,
    )

    rnn_layers = {"simplernn": SimpleRNN, "lstm": LSTM, "gru": GRU}
    if model_type not in rnn_layers:
        raise ValueError(f"Unknown model type: {model_type}")

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        rnn_layers[model_type](64, dropout=0.2),
        Dense(32, activation="relu"),
        Dropout(0.3),
    ])

    if num_classes == 2:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
    else:
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    return model


def train_neural_model(model_type, dataset_name, texts, labels, label_names,
                       max_len=DEFAULT_MAX_LEN, progress_callback=None):
    """Train a neural model and save all artifacts. Returns result dict."""
    from tensorflow.keras.callbacks import EarlyStopping

    os.makedirs(MODELS_DIR, exist_ok=True)

    def progress(msg):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass
        print(msg)

    start = time.time()

    # Preprocess
    progress("Cleaning texts...")
    cleaned = [clean_text(t) for t in texts]

    # Encode labels
    le = LabelEncoder()
    le.fit(label_names)
    y = le.transform([label_names[l] for l in labels])
    num_classes = len(label_names)

    # Tokenizer
    progress("Building tokenizer...")
    tokenizer = build_tokenizer(cleaned, MAX_VOCAB_SIZE)
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)

    # Pad
    X = texts_to_padded(tokenizer, cleaned, max_len)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Build
    progress(
        f"Building {model_type.upper()} "
        f"(max_len={max_len}, vocab={vocab_size}, classes={num_classes})..."
    )
    model = _build_model(model_type, vocab_size, EMBEDDING_DIM, max_len,
                         num_classes)

    # Train
    es = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    progress(f"Training for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0,
    )

    duration = time.time() - start

    # Save
    prefix = f"{model_type}_{dataset_name}"
    paths = _save_artifacts(prefix, model, tokenizer, le, max_len, num_classes)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    result = {
        "model_type": model_type,
        "dataset": dataset_name,
        "epochs_run": len(history.history["loss"]),
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "duration": duration,
        "max_len": max_len,
        "history": history.history,
        **paths,
    }

    progress(
        f"Training complete!\n"
        f"  Model: {model_type.upper()}\n"
        f"  Dataset: {dataset_name}\n"
        f"  Epochs: {result['epochs_run']}/{EPOCHS}\n"
        f"  Val accuracy: {result['val_accuracy']}\n"
        f"  Val loss: {result['val_loss']}\n"
        f"  Duration: {duration:.1f}s\n"
        f"  Saved: {paths['model_path']}"
    )

    return result


def _save_artifacts(prefix, model, tokenizer, label_encoder, max_len,
                    num_classes):
    """Save model .h5, tokenizer .pkl, label_encoder .pkl, meta .pkl."""
    model_path = os.path.join(MODELS_DIR, f"{prefix}.h5")
    tok_path = os.path.join(MODELS_DIR, f"{prefix}_tokenizer.pkl")
    enc_path = os.path.join(MODELS_DIR, f"{prefix}_label_encoder.pkl")
    meta_path = os.path.join(MODELS_DIR, f"{prefix}_meta.pkl")

    model.save(model_path)
    with open(tok_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(enc_path, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(meta_path, "wb") as f:
        pickle.dump({"max_len": max_len, "num_classes": num_classes}, f)

    return {
        "model_path": model_path,
        "tokenizer_path": tok_path,
        "encoder_path": enc_path,
    }
