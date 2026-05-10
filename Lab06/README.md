# NLP Bot (Lab06)

Unified Telegram bot for all course labs (Lab01-Lab06).

This version includes:

- Lab 1 text processing and classic NLP tasks
- Lab 2 text classification experiments (BoW, TF-IDF, Word2Vec, GloVe)
- Lab 3 sentiment analysis and neural training commands
- Lab 4 NER/NEL/NED, translation, summarization, knowledge graph
- Lab 5 tool-calling agent (including vision)
- Lab 6 content moderation pipeline with CSV audit logs

## Command Architecture (Current)

Command registration is split by lab modules and aggregated centrally:

- Global command router: `commands.py`
- Global handlers only: `/start`, `/help`
- Per-lab handlers:
    - `lab1/commands.py`
    - `lab2/commands.py`
    - `lab3/commands.py`
    - `lab4/commands.py`
    - `lab5/commands.py`
    - `lab6/commands.py`

This means each lab owns its own commands, while root `commands.py` only composes help and registers all labs.

## Requirements

- Python 3.10+ (3.10/3.11 recommended for best dependency compatibility)
- Telegram bot token
- Optional but strongly recommended for Lab4/Lab5 features: Ollama

Install Python dependencies from:

- `requirements.txt`

## Installation

1. Enter the Lab06 directory.

```bash
cd Lab06
```

2. Create and activate a virtual environment.

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Download required language resources.

```bash
python -m spacy download pl_core_news_sm
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

5. (Optional, but needed for local LLM features) setup Ollama.

```bash
ollama serve
ollama pull qwen2.5:1.5b
ollama pull qwen2.5vl:3b
```

6. Create `.env` in `Lab06/`:

```env
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
```

7. Run the bot.

```bash
python bot.py
```

## Core Runtime Config

Important settings are in `config.py`, including:

- `OLLAMA_MODEL`
- `VISION_MODEL`
- `MAX_AGENT_ITERATIONS`
- weather/web API endpoints
- output/cache directories

Lab06 moderation-specific settings live in:

- `lab6/config.py`

## Command Reference

### Global

- `/start`
- `/help`

### Lab 1

- `/task <name> "text" "class"`
- `/full_pipeline "text" "class"`
- `/classifier "text"`
- `/stats`

Task names include:

- `tokenize`
- `remove_stopwords`
- `lemmatize`
- `stemming`
- `stats`
- `n-grams`
- `plot_histogram`
- `plot_wordcloud`
- `plot_barchart`

### Lab 2

- `/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>`

Supported datasets:

- `20news_group`
- `imdb`
- `amazon`
- `ag_news`

Supported methods:

- `nb`
- `rf`
- `mlp`
- `logreg`
- `all`

### Lab 3

- `/sentiment method=<m> text="..." [dataset=<d>]`
- `/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>`
- `/compare dataset=<amazon|imdb|custom> methods=<m1,m2,...>`
- `/add_sentiment "text" "label"`
- `/models`

Sentiment methods include:

- `rule`
- `nb`
- `rf`
- `transformer`
- `textblob`
- `stanza`
- `simplernn`
- `lstm`
- `gru`

### Lab 4

- `/language_detect text="..."`
- `/ner method=<spacy|stanza> text="..." [language=auto|en|pl]`
- `/nel text="..." [language=auto|en|pl]`
- `/ned entity="..." context="..." [language=auto|en|pl]`
- `/analyze_entities text="..." [link=true|false] [language=auto|en|pl]`
- `/translate text="..." target_lang=<en|pl|de|fr|es> [source_lang=auto]`
- `/summarize text="..." [summary_type=abstractive|extractive|bullets|custom] [length=short|medium|long] [prompt="..."]`
- `/knowledge_graph text="..." [language=auto|en|pl]`

### Lab 5 (Agent + Tools)

- `/agent text="..."` (also supports photo + caption)
- `/chat`
- `/chat_reset`
- `/agent_history [n=5]`
- `/tools`

Direct tool commands:

- `/tool_calc expression="2+2*sqrt(16)"`
- `/tool_weather city="Warsaw"`
- `/tool_search query="OpenAI" [language=en|pl]`
- `/tool_local_kb query="Paris"`
- `/tool_datetime [city=Tokyo|tz=Europe/Warsaw]`
- `/tool_nlp operation=<translate|summarize|extract_entities|classify_sentiment> text="..." [target_language=...] [language=...]`
- `/tool_vision` (send photo with this caption)

### Lab 6 (Moderation)

- `/moderate "text" [user_id=...] [username=...]`
- `/mod_policy_check "text"`
- `/mod_status <content_id>`
- `/mod_history <user_id>`
- `/mod_analytics`
- `/mod_add_feedback <content_id> "comment" "APPROVE|REJECT|FLAG_FOR_REVIEW"`
- `/mod_watchlist`
- `/mod_train_on_feedback`
- `/mod_help`

`/mod_add_feedback` also accepts decision aliases in params:

- `decision`
- `override`
- `correct_decision`
- `correct`
- `poprawna_decyzja`

## Moderation Pipeline (Lab 6)

Main flow in `lab6/moderation/pipeline.py`:

1. PII detection
2. Bielik-Guard style classification
3. Qwen-Guard style risk recommendation
4. Sentiment context
5. Entity extraction context
6. Ensemble decision (`APPROVE`, `REJECT`, `FLAG_FOR_REVIEW`)
7. Action execution through tool layer
8. CSV persistence and analytics update

Moderation action tools include:

- `approve_content`
- `reject_content`
- `flag_for_human_review`
- `shadow_ban_user`
- `add_to_watchlist`
- `get_user_moderation_history`
- `find_similar_violations`

## Outputs and Data Files

Generated artifacts are saved to:

- `cache/sessions/` - agent chat sessions and run traces
- `cache/nel_cache.json` - NEL cache
- `lab4plots/` - Lab4 plots (including knowledge graph PNG)
- `lab4results/summaries/` - saved summaries
- `results/lab2results.csv` - Lab2 experiment results
- `lab3/results/lab3results.csv` - Lab3 comparison/training results
- `moderation_data/moderation_log.csv`
- `moderation_data/moderation_actions.csv`
- `moderation_data/user_moderation_history.csv`
- `moderation_data/feedback_log.csv`
- `moderation_data/train_data.csv`
- `moderation_data/watchlist.csv`

## Project Structure (High Level)

```text
Lab06/
├── bot.py
├── commands.py
├── config.py
├── instructions.md
├── requirements.txt
├── utils.py
├── lab1/
│   ├── commands.py
│   └── ...
├── lab2/
│   ├── commands.py
│   └── ...
├── lab3/
│   ├── commands.py
│   └── ...
├── lab4/
│   ├── commands.py
│   └── ...
├── lab5/
│   ├── commands.py
│   ├── agent.py
│   └── tools/
├── lab6/
│   ├── commands.py
│   └── moderation/
└── moderation_data/
```

## Troubleshooting

### Missing token

If startup fails with missing token, verify `.env` contains:

- `TELEGRAM_BOT_TOKEN=...`

### Ollama unavailable

If agent/summarization fails with Ollama connection errors:

- start service: `ollama serve`
- ensure model exists: `ollama pull qwen2.5:1.5b`
- for vision: `ollama pull qwen2.5vl:3b`

### First-run model downloads take time

Initial runs of some commands can be slow because models are downloaded on first use.

### Use project venv for checks

Always run checks and imports using `Lab06/venv` Python interpreter to avoid false negatives from system Python.
