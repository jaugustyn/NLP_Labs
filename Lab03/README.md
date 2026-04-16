# NLP Bot — Laboratory 3 (Sentiment Analysis & Sequential Models)

Extension of Lab 1 + Lab 2 bot with sentiment analysis capabilities:

- All Lab 1 NLP commands (tokenize, lemmatize, stemming, etc.)
- All Lab 2 classification experiments (BoW, TF-IDF, Word2Vec, GloVe + NB, RF, MLP, LogReg)
- 8 sentiment analysis methods (rule-based, ML, transformer, neural)
- Sequential model training (SimpleRNN, LSTM, GRU)
- Custom dataset management (sentiment_dataset.csv)
- Method comparison with metrics and visualizations

## Commands

### Lab 3 (New)

| Command                                    | Description                          |
| ------------------------------------------ | ------------------------------------ |
| `/sentiment method=<m> text="..."`         | Analyze sentiment with chosen method |
| `/train model=<type> dataset=<name>`       | Train a neural model                 |
| `/compare dataset=<d> methods=<m1,m2,...>` | Compare methods on a dataset         |
| `/add_sentiment "text" "label"`            | Add record to custom dataset         |
| `/models`                                  | List saved models                    |
| `/help`                                    | Show all commands                    |

### Lab 1 (Inherited)

| Command                         | Description        |
| ------------------------------- | ------------------ |
| `/task <name> "text" "class"`   | Run NLP task       |
| `/full_pipeline "text" "class"` | Full NLP pipeline  |
| `/classifier "text"`            | Classify text      |
| `/stats`                        | Dataset statistics |

Tasks for `/task`: `tokenize`, `remove_stopwords`, `lemmatize`, `stemming`, `stats`, `n-grams`, `plot_histogram`, `plot_wordcloud`, `plot_barchart`

### Lab 2 (Inherited)

| Command                                                                   | Description                   |
| ------------------------------------------------------------------------- | ----------------------------- |
| `/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>` | Run classification experiment |

| Parameter  | Values                                            | Description                      |
| ---------- | ------------------------------------------------- | -------------------------------- |
| dataset    | `20news_group`, `imdb`, `amazon`, `ag_news`       | Dataset to use                   |
| method     | `nb`, `rf`, `mlp`, `logreg`, `all` (comma-sep OK) | Classifier(s) to train           |
| gridsearch | `true` / `false`                                  | Hyper-parameter tuning           |
| run        | `1`–`3`                                           | Number of runs (different seeds) |

## Sentiment Methods

| Method        | Description                          | Notes                                 |
| ------------- | ------------------------------------ | ------------------------------------- |
| `rule`        | Rule-based (PL/EN sentiment lexicon) | No training needed                    |
| `nb`          | Naive Bayes + TF-IDF                 | Auto-trains on first use              |
| `transformer` | DistilBERT (HuggingFace)             | English, no training                  |
| `textblob`    | TextBlob polarity                    | English, no training                  |
| `stanza`      | Stanza sentiment                     | English, downloads model on first use |
| `simplernn`   | SimpleRNN (Keras)                    | Requires `/train` first               |
| `lstm`        | LSTM (Keras)                         | Requires `/train` first               |
| `gru`         | GRU (Keras)                          | Requires `/train` first               |

## Neural Model Architecture

All sequential models follow the same pattern:

```
Embedding(vocab_size, 100, input_length=max_len)
→ SimpleRNN / LSTM / GRU (64 units, dropout=0.2)
→ Dense(32, relu)
→ Dropout(0.3)
→ Dense(output, sigmoid/softmax)
```

### Parameters

| Parameter                 | Default | Description                |
| ------------------------- | ------- | -------------------------- |
| `EMBEDDING_DIM`           | 100     | Embedding layer dimension  |
| `DEFAULT_MAX_LEN`         | 200     | Sequence padding length    |
| `MAX_VOCAB_SIZE`          | 20,000  | Tokenizer vocabulary limit |
| `BATCH_SIZE`              | 32      | Training batch size        |
| `EPOCHS`                  | 10      | Maximum epochs             |
| `EARLY_STOPPING_PATIENCE` | 3       | Early stopping patience    |

### Experimenting with max_len

The `max_len` parameter controls input sequence length. To experiment, modify `DEFAULT_MAX_LEN` in `config.py`:

| max_len | Characteristics                                   |
| ------- | ------------------------------------------------- |
| 100     | Faster training, may lose context on longer texts |
| 150     | Good balance for short reviews                    |
| **200** | Default — works well for most datasets            |
| 300     | Better for longer texts (IMDB), slower training   |

The `MAX_LEN_OPTIONS` list in `config.py` defines the range `[100, 150, 200, 300]` for batch experiments.

## Datasets

| Name     | Description            | Classes                         | Language |
| -------- | ---------------------- | ------------------------------- | -------- |
| `amazon` | Amazon product reviews | negative, positive              | English  |
| `imdb`   | IMDB movie reviews     | negative, positive              | English  |
| `custom` | User-managed CSV file  | pozytywny, neutralny, negatywny | Polish   |

### Custom Dataset

The `/add_sentiment` command appends records to `sentiment_dataset.csv`. Multi-sentence texts are stored as a single record (not split into sentences).

Minimum format:

```csv
text,label
"Uwielbiam ten film",pozytywny
"To był zwykły dzień",neutralny
"Ten produkt jest fatalny",negatywny
```

A starter dataset with 15 records (5 per class) is included.

## Output Files

| Output             | Path                                           |
| ------------------ | ---------------------------------------------- |
| Neural models      | `models/<type>_<dataset>.h5`                   |
| Tokenizers         | `models/<type>_<dataset>_tokenizer.pkl`        |
| Label encoders     | `models/<type>_<dataset>_label_encoder.pkl`    |
| Sklearn models     | `models/<method>_<dataset>_sklearn.pkl`        |
| Comparison CSV     | `results/lab3results.csv`                      |
| Training plots     | `lab3plots/train_history_<type>_<dataset>.png` |
| Confusion matrices | `lab3plots/confusion_<method>_<dataset>.png`   |
| Comparison chart   | `lab3plots/compare_methods_<dataset>.png`      |
| Word clouds        | `lab3plots/wordcloud_<label>.png`              |
| Class distribution | `lab3plots/class_distribution_<dataset>.png`   |

## Project Structure

```
Lab03/
├── bot.py                 — Entry point
├── commands.py            — All command handlers (Lab 1 + Lab 2 + Lab 3)
├── config.py              — Configuration & constants
├── utils.py               — Parsing, formatting, logging
├── preprocessing.py       — Text cleaning & Keras preprocessing
├── data_loader.py         — Dataset loading & custom CSV management
├── training.py            — Neural model building & training
├── model_loader.py        — Model loading, saving & listing
├── sentiment_methods.py   — 8 sentiment analysis methods
├── visualizations.py      — All plots & charts
├── sentiment_dataset.csv  — Custom dataset (starter, 15 records)
├── requirements.txt       — Python dependencies
├── .gitignore
├── .env                   — Telegram bot token (not in repo)
├── lab1/                  — Lab 1 modules (inherited)
│   ├── nlp_core.py        — Tokenization, stopwords, lemmatization, stemming
│   ├── data_manager.py    — Save/load sentences.json
│   ├── classifier.py      — Single-message classifier
│   ├── visualizer.py      — Lab 1 charts
│   ├── stopwords-pl.txt   — Polish stopwords
│   └── sentences.json     — Labeled sentence dataset
├── lab2/                  — Lab 2 modules (inherited)
│   ├── experiment.py      — Lab 2 experiment orchestrator
│   ├── dataset_loader.py  — Dataset loading (sklearn + HuggingFace)
│   ├── text_embeddings.py — BoW, TF-IDF, Word2Vec, GloVe wrappers
│   ├── models.py          — Classifier factory + GridSearch config
│   └── visualizer.py      — Lab 2 visualisation functions
├── models/                — Saved models (generated, gitignored)
├── lab3plots/             — Visualizations (generated, gitignored)
└── results/               — Comparison results (generated, gitignored)
```

## Installation

1. Clone the repo and switch to the `Lab03` branch:

   ```bash
   git clone https://github.com/jaugustyn/NLP_Labs.git
   cd NLP_Labs
   git checkout Lab03
   cd Lab03
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m spacy download pl_core_news_sm
   python -m textblob.download_corpora
   ```

4. Create `.env` in `Lab03/`:

   ```
   TELEGRAM_BOT_TOKEN=<YOUR_TOKEN>
   ```

5. Start the bot:

   ```bash
   python bot.py
   ```

> **Note:** First use of `transformer`, `stanza`, and dataset loading may download models/data — allow a few minutes.

## Usage Examples

```
/sentiment method=rule text="To był naprawdę świetny film"
/sentiment method=transformer text="This product is terrible and I want a refund"
/sentiment method=textblob text="The movie was okay, nothing special"
/train model=lstm dataset=imdb
/train model=simplernn dataset=amazon
/train model=gru dataset=custom
/sentiment method=lstm dataset=imdb text="Amazing experience, highly recommend!"
/compare dataset=imdb methods=rule,nb,transformer,textblob,stanza,lstm
/compare dataset=custom methods=rule,textblob,nb
/add_sentiment "Obsługa była poprawna ale niczym mnie nie zachwyciła" "neutralny"
/models
```

## System Requirements

- Python 3.10+
- Telegram Bot API token
- ~2 GB disk space (transformer + stanza models)
- GPU recommended for neural model training (CPU works but slower)
