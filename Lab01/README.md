# NLP Bot – Laboratory 1
### Natural Language Processing

A simple Telegram bot for processing and classifying single text messages. The project demonstrates core NLP operations and basic machine learning classification trained on user-provided examples.

## Features

The bot supports four main commands:

1. **/task `<task_name>` `"text"` `"class"`** - Run a single NLP task and save the labeled example to JSON (e.g. `tokenize`, `remove_stopwords`, `lemmatize`, `stemming`, `stats`, `n-grams`, `plot_histogram`, `plot_wordcloud`, `plot_barchart`).
2. **/full_pipeline `"text"` `"class"`** - Run the full processing pipeline including tokenization, stopword removal, lemmatization, stemming, BoW, TF-IDF, statistics, and chart generation.
3. **/classifier `"text"`** - Predict a label for a new message using a machine learning model trained on examples stored in `sentences.json`.
4. **/stats** - Aggregate global dataset statistics (class distribution, unique tokens, n-grams) and generate charts.

## System Requirements

- Python 3.12+
- Valid Telegram Bot API token from [BotFather](https://core.telegram.org/bots#botfather)

## Installation and Run

1. Clone the repository:

```bash
git clone <repository_url>
cd NLP_Labs
```

2. Switch to the Lab01 branch and move to the lab directory:
```
git checkout Lab01
cd Lab01
```

3. Create and activate a virtual environment

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
   ```

5. Download the Polish spaCy language model:

   ```bash
   python -m spacy download pl_core_news_sm
   ```

6. Create `.env` in the same directory as `bot.py`:

   ```text
   TELEGRAM_BOT_TOKEN=<TOKEN>
   ```

7. Download required NLTK resources:

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

8. Start the bot:

   ```bash
   python bot.py
   ```

## Usage Examples

```text
/task tokenize "To był bardzo zły dzień." "negatywny"
/task plot_barchart "To jest świetna i bardzo duża chmura wyrazów z bardzo pozytywnym wydźwiękiem" "pozytywny"
/full_pipeline "System działa szybko, ale interfejs wymaga poprawy." "neutralny"
/classifier "To był fantastyczny film"
/stats
```

Class labels used by the project:

- `pozytywny` = `1`
- `neutralny` = `0`
- `negatywny` = `-1`

## Project Structure

- `bot.py` - Main Telegram command handlers and runtime loop.
- `nlp_core.py` - NLP processing utilities.
- `classifier.py` - Text classifier based on `scikit-learn`.
- `visualizer.py` - Chart generation with `matplotlib` and `wordcloud`.
- `data_manager.py` - Read/write operations for `sentences.json`.
- `stopwords-pl.txt` - Polish stopword list.
- `sentences.json` - Collected labeled examples used for training.
- `plots/` - Output directory for generated charts.
- `requirements.txt` - Python dependencies required to run the bot.
