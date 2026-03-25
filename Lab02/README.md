# NLP Bot - Laboratory 2 (Text Classification Experiments)

Extension of the Lab 1 bot. Adds full classification experiments on standard NLP datasets using multiple text representations and classifiers. Lab 1 single-message commands remain available.

## Lab 1 Commands (inherited)

| Command                         | Description                                         |
| ------------------------------- | --------------------------------------------------- |
| `/task <name> "text" "class"`   | Run a single NLP task on given text                 |
| `/full_pipeline "text" "class"` | Full NLP pipeline (tokenize → BoW → TF-IDF → plots) |
| `/classifier "text"`            | Classify text using saved dataset                   |
| `/stats`                        | Dataset statistics + charts                         |

Tasks for `/task`: `tokenize`, `remove_stopwords`, `lemmatize`, `stemming`, `stats`, `n-grams`, `plot_histogram`, `plot_wordcloud`, `plot_barchart`

## Lab 2 Command

```
/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>
```

### Parameters

| Parameter  | Values                                            | Description                      |
| ---------- | ------------------------------------------------- | -------------------------------- |
| dataset    | `20news_group`, `imdb`, `amazon`, `ag_news`       | Dataset to use                   |
| method     | `nb`, `rf`, `mlp`, `logreg`, `all` (comma-sep OK) | Classifier(s) to train           |
| gridsearch | `true` / `false`                                  | Hyper-parameter tuning           |
| run        | `1`–`3`                                           | Number of runs (different seeds) |

### Examples

```
/classify dataset=20news_group method=all gridsearch=false run=1
/classify dataset=imdb method=logreg gridsearch=true run=2
/classify dataset=ag_news method=rf,nb gridsearch=false run=3
```

## What the experiment produces

| Output                     | Path / file                                                |
| -------------------------- | ---------------------------------------------------------- |
| Classification results CSV | `lab2results/results.csv`                                  |
| Word cloud (corpus)        | `lab2plots/wordcloud_corpus.png`                           |
| Word cloud (per class)     | `lab2plots/wordcloud_class_<class>.png`                    |
| Confusion matrices         | `lab2plots/confusion_<emb>_<model>.png`                    |
| Embedding visualisations   | `lab2plots/<dataset>_<model>_<emb>_<method>_embedding.png` |
| Word embedding viz         | `lab2plots/word_embedding_pca.png`, `…_tsne.png`           |
| Similar words              | `lab2results/similar_words.txt`                            |
| Feature importance         | `lab2results/feature_importance.txt`                       |

## Text representations

| Embedding | Description                       |
| --------- | --------------------------------- |
| bow       | Bag of Words (`CountVectorizer`)  |
| tfidf     | TF-IDF (`TfidfVectorizer`)        |
| word2vec  | Word2Vec trained on the corpus    |
| glove     | Pre-trained GloVe (100-d, gensim) |

## Classifiers

| Key    | Model               | Grid Search params                             |
| ------ | ------------------- | ---------------------------------------------- |
| nb     | Naive Bayes         | alpha=[0.1, 0.5, 1.0]                          |
| rf     | Random Forest       | n_estimators=[100,300], max_depth=[None,10,20] |
| mlp    | MLP                 | hidden_layer_sizes=[(128,), (256,128)]         |
| logreg | Logistic Regression | C=[0.1, 1, 10]                                 |

## System Requirements

- Python 3.12+
- Telegram Bot API token

## Installation and Run

1. Clone the repo and switch to the `lab02` branch.
2. Navigate to the `Lab02/` directory.
3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m spacy download pl_core_news_sm
   ```

5. Create `.env` in `Lab02/`:

   ```
   TELEGRAM_BOT_TOKEN=<YOUR_TOKEN>
   ```

6. Start the bot:

   ```bash
   python bot.py
   ```

> **Note:** The first run with `word2vec` / `glove` may download models and datasets — this can take a few minutes.

## Project Structure

```
Lab02/
  bot.py              — Telegram bot (Lab 1 + Lab 2 commands)
  experiment.py       — Experiment orchestrator (pipeline)
  dataset_loader.py   — Dataset loading (sklearn + HuggingFace)
  text_embeddings.py  — BoW, TF-IDF, Word2Vec, GloVe wrappers
  models.py           — Classifier factory + GridSearch config
  visualizer.py       — Lab 2 visualisation functions
  requirements.txt    — Python dependencies
  lab1/
    nlp_core.py       — Tokenization, stopwords, lemmatization, stemming
    data_manager.py   — Save/load sentences.json
    classifier.py     — Single-message classifier (Lab 1)
    visualizer.py     — Lab 1 charts (histogram, wordcloud, barchart)
    stopwords-pl.txt  — Polish stopwords list
    sentences.json    — Labeled sentence dataset
```
