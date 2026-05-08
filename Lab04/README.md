# NLP Bot — Laboratory 4 (NER, NEL, NED, Translation, Summarization)

Extension of Lab 1 + Lab 2 + Lab 3 bot with named-entity processing,
machine translation, summarization and a knowledge graph view:

- All Lab 1 NLP commands (tokenize, lemmatize, stemming, etc.)
- All Lab 2 classification experiments (BoW, TF-IDF, Word2Vec, GloVe + NB, RF, MLP, LogReg)
- All Lab 3 sentiment analysis methods (rule, NB, RF, transformer, TextBlob, Stanza, SimpleRNN, LSTM, GRU)
- Automatic language detection (langdetect)
- Named Entity Recognition (spaCy + Stanza)
- Named Entity Linking to Wikidata with a local JSON cache
- Named Entity Disambiguation by context-token overlap
- Full pipeline command (`/analyze_entities`) running NER + NEL + NED
- Machine translation (Helsinki-NLP/Opus-MT, lazy-loaded per language pair)
- Text summarization via a local LLM (Ollama, `qwen2.5:1.5b`)
- Co-occurrence knowledge graph (networkx + matplotlib)

## Commands

### Lab 4 (New)

| Command                                                                                                                     | Description                                       |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `/language_detect text="..."`                                                                                               | Detect text language with confidence              |
| `/ner method=<spacy\|stanza> text="..." [language=auto\|en\|pl]`                                                            | Named Entity Recognition                          |
| `/nel text="..." [language=auto\|en\|pl]`                                                                                   | NER + Wikidata candidate search                   |
| `/ned entity="..." context="..." [language=auto\|en\|pl]`                                                                   | Disambiguate one entity in a context              |
| `/analyze_entities text="..." [link=true\|false] [language=auto\|en\|pl]`                                                   | Full pipeline: spaCy + Stanza + optional linking  |
| `/translate text="..." target_lang=<en\|pl\|de\|fr\|es> [source_lang=auto]`                                                 | Machine translation (Opus-MT)                     |
| `/summarize text="..." [summary_type=abstractive\|extractive\|bullets\|custom] [length=short\|medium\|long] [prompt="..."]` | Summarize via local LLM                           |
| `/knowledge_graph text="..." [language=auto\|en\|pl]`                                                                       | Build & render entity co-occurrence graph         |
| `/help`                                                                                                                     | Show all commands (Lab 1 + Lab 2 + Lab 3 + Lab 4) |

> Aliases accepted for backward compatibility: `lang=` (== `language=`),
> `src=` (== `source_lang=`), `tgt=` (== `target_lang=`),
> `type=` (== `summary_type=`).

### Lab 3 (Inherited)

| Command                                    | Description                          |
| ------------------------------------------ | ------------------------------------ |
| `/sentiment method=<m> text="..."`         | Analyze sentiment with chosen method |
| `/train model=<type> dataset=<name>`       | Train a neural model                 |
| `/compare dataset=<d> methods=<m1,m2,...>` | Compare methods on a dataset         |
| `/add_sentiment "text" "label"`            | Add record to custom dataset         |
| `/models`                                  | List saved models                    |

### Lab 2 (Inherited)

| Command                                                                   | Description                   |
| ------------------------------------------------------------------------- | ----------------------------- |
| `/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>` | Run classification experiment |

### Lab 1 (Inherited)

| Command                         | Description        |
| ------------------------------- | ------------------ |
| `/task <name> "text" "class"`   | Run NLP task       |
| `/full_pipeline "text" "class"` | Full NLP pipeline  |
| `/classifier "text"`            | Classify text      |
| `/stats`                        | Dataset statistics |

## Lab 4 Components

### Language Detection

`langdetect` with a deterministic seed (42). Returns ISO 639-1 code and a
confidence score. Used internally whenever `lang=auto` or `src=auto` is
passed to other commands.

### Named Entity Recognition

| Method   | Backend                             | Languages         | Notes                              |
| -------- | ----------------------------------- | ----------------- | ---------------------------------- |
| `spacy`  | `pl_core_news_sm`, `en_core_web_sm` | PL, EN            | Default — fast, must be downloaded |
| `stanza` | Stanza NER models                   | PL, EN, DE, FR, … | Auto-downloads on first use        |

`/analyze_entities` runs both backends and merges the entity sets,
de-duplicating by `(text, label)`.

### Named Entity Linking

Searches Wikidata via the `wbsearchentities` API and returns the top
candidate (QID, label, description). Results are cached in
`cache/nel_cache.json` (key: `<lang>::<lower-cased-name>`) to limit
network calls.

### Named Entity Disambiguation

Each candidate is scored by token overlap between the candidate's label
and description and the surrounding context (the entity name itself is
removed from the context tokens). The highest-scoring candidate is
returned; if its score is below `NEL_MIN_CONFIDENCE`, the result is
flagged with `low_confidence=True`.

### Machine Translation

Uses `Helsinki-NLP/opus-mt-<src>-<tgt>` models via `transformers`.
Pipelines are lazy-loaded per language pair and cached in process
memory. Supported pairs:

| Source | Targets                |
| ------ | ---------------------- |
| `en`   | `pl`, `de`, `fr`, `es` |
| `pl`   | `en`, `de`, `fr`       |
| `de`   | `en`, `pl`             |
| `fr`   | `en`, `pl`             |
| `es`   | `en`                   |

Each model occupies roughly 300 MB and is downloaded on first use.

### Summarization

Uses Ollama running locally (`http://localhost:11434`) with the
`qwen2.5:1.5b` model (multilingual, ~1.5 B parameters).

| Type          | Description                                             |
| ------------- | ------------------------------------------------------- |
| `extractive`  | Keep key fragments from the original text               |
| `abstractive` | Rewrite in own words (default)                          |
| `bullets`     | 3–7 bullet points                                       |
| `custom`      | Use the user-supplied `prompt="..."` as the instruction |

Length controls `num_predict` (target tokens):

| Length   | Tokens |
| -------- | ------ |
| `short`  | 80     |
| `medium` | 200    |
| `long`   | 400    |

### Knowledge Graph

The input text is split into sentences and entities are extracted per
sentence (spaCy). Each entity becomes a node (size scaled by occurrence
count, color by label); entities that appear in the same sentence get
an edge with a weight equal to the co-occurrence count. The graph is
rendered with `networkx` + `matplotlib` and saved as PNG.

Reported statistics: `nodes`, `edges`, `connected components`, `density`.

## Output Files

| Output                | Path                                                            |
| --------------------- | --------------------------------------------------------------- |
| Knowledge graph plots | `lab4plots/knowledge_graph_<timestamp>.png`                     |
| Saved summaries       | `lab4results/summaries/summary_<type>_<length>_<timestamp>.txt` |
| NEL cache             | `cache/nel_cache.json`                                          |
| Lab 3 artifacts       | `lab3/models/`, `lab3/lab3plots/`, `lab3/results/`              |

## Project Structure

```
Lab04/
├── bot.py                 — Entry point
├── commands.py            — Lab 4 command handlers + delegation to Lab 1+2+3
├── config.py              — Configuration & constants (Lab 4)
├── utils.py               — Parsing, formatting, logging, truncation
├── language_detect.py     — langdetect wrapper
├── ner.py                 — spaCy + Stanza NER, merge & grouping helpers
├── nel.py                 — Wikidata search + JSON cache
├── ned.py                 — Candidate ranking & disambiguation
├── translation.py         — Helsinki-NLP/Opus-MT pipelines (lazy)
├── summarization.py       — Ollama HTTP client + prompt templates
├── knowledge_graph.py     — networkx graph builder + matplotlib plot
├── requirements.txt       — Python dependencies
├── .gitignore
├── .env                   — Telegram bot token (not in repo)
├── lab1/                  — Lab 1 modules (inherited as a subpackage)
├── lab2/                  — Lab 2 modules (inherited as a subpackage)
├── lab3/                  — Lab 3 modules (inherited as a subpackage)
├── cache/                 — NEL cache (generated, gitignored)
├── lab4plots/             — Knowledge-graph PNGs (generated, gitignored)
└── lab4results/           — Saved summaries (generated, gitignored)
```

## Installation

1. Clone the repo and switch to the `Lab04` branch:

   ```bash
   git clone https://github.com/jaugustyn/NLP_Labs.git
   cd NLP_Labs
   git checkout Lab04
   cd Lab04
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
   python -m spacy download en_core_web_sm
   python -m textblob.download_corpora
   ```

4. Install and start Ollama (https://ollama.com/download):

   ```bash
   ollama serve
   ollama pull qwen2.5:1.5b
   ```

5. Create `.env` in `Lab04/`:

   ```
   TELEGRAM_BOT_TOKEN=<YOUR_TOKEN>
   ```

6. Start the bot:

   ```bash
   python bot.py
   ```

> **Note:** First use of `/translate`, `/ner method=stanza`, and the
> `transformer` / `stanza` sentiment methods will download models — allow
> a few minutes per backend.

## Usage Examples

```text
# Language detection
/language_detect text="Warszawa to stolica Polski"
/language_detect text="The quick brown fox jumps over the lazy dog"

# Named Entity Recognition (per-entity offsets in output)
/ner method=spacy text="Apple was founded by Steve Jobs in California"
/ner method=stanza text="Frédéric Chopin urodził się w Polsce" language=pl
/ner method=spacy text="Barack Obama was the 44th President of the United States"

# Named Entity Linking — Wikidata candidates with Wikipedia URL & confidence
/nel text="Elon Musk founded SpaceX in 2002" language=en
/nel text="Maria Skłodowska-Curie odkryła rad i polon" language=pl

# Disambiguation of a single ambiguous name
/ned entity="Paris" context="Paris is the capital of France"
/ned entity="Paris" context="Paris Hilton attended the gala in New York"
/ned entity="Apple" context="Apple released a new iPhone yesterday"

# Full pipeline (spaCy + Stanza merged + Wikidata + Wikipedia URLs)
/analyze_entities text="Barack Obama was the 44th President of the United States"
/analyze_entities text="Elon Musk founded SpaceX in 2002" link=true
/analyze_entities text="Steve Jobs founded Apple" link=false

# Machine translation (auto-detect source unless given)
/translate text="Dziś jest ładny dzień" target_lang=en
/translate text="Bonjour le monde" target_lang=en
/translate text="Berlin ist eine schöne Stadt" target_lang=pl
/translate text="The cat sits on the mat" source_lang=en target_lang=es

# Summarization (requires Ollama running on localhost:11434)
/summarize text="Long article text..." summary_type=bullets length=short
/summarize text="..." summary_type=abstractive length=medium
/summarize text="..." summary_type=custom prompt="List only company names from the text"

# Knowledge graph (multi-sentence input → co-occurrence graph)
/knowledge_graph text="Elon Musk founded SpaceX. Steve Jobs founded Apple. Both met in California."
/knowledge_graph text="Warszawa to stolica Polski. Kraków leży w południowej Polsce." language=pl
```

### Known limitations

- **Translation `en → pl`**: the original `Helsinki-NLP/opus-mt-en-pl` is
  gated on Hugging Face and replaced internally with the multi-target
  `Helsinki-NLP/opus-mt-en-sla` (with a `>>pol<<` prefix). Quality is
  comparable for general text.
- **NER on short text**: `en_core_web_sm` may miss product names like
  `SpaceX` or `iPhone`. For better recall use `method=stanza` or run
  `/analyze_entities` (which merges both backends).
- **`langdetect` on very short text**: e.g. `"Hello world"` may be
  detected as `nl`. `/translate` automatically falls back to `en` as the
  source when the detected language has no model for the target.
- **Wikidata rate-limits**: heavy usage may return errors; results are
  cached in `cache/nel_cache.json` once successfully fetched.

## System Requirements

- Python 3.10+
- Telegram Bot API token
- Ollama installed locally with `qwen2.5:1.5b` pulled
- ~3 GB disk space (Opus-MT + Stanza + spaCy + transformer models)
- Internet access for first-time model downloads and Wikidata queries
