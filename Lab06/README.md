# NLP Bot — Laboratory 6 (Content Moderation Pipeline)

Extension of Lab 1-5 bot with a local multi-stage content moderation
pipeline. Lab06 keeps the Lab05 tool-calling agent and adds policy
enforcement, PII detection, toxicity classification, sentiment context,
NER context, moderation actions, feedback logging and analytics.

- All Lab 1 NLP commands (tokenize, lemmatize, stemming, etc.)
- All Lab 2 classification experiments (BoW, TF-IDF, Word2Vec, GloVe + NB, RF, MLP, LogReg)
- All Lab 3 sentiment analysis methods (rule, NB, RF, transformer, TextBlob, Stanza, SimpleRNN, LSTM, GRU)
- All Lab 4 commands (NER, NEL, NED, translation, summarization, knowledge graph)
- Tool / function calling via Ollama (`qwen2.5:1.5b`, native tool calls)
- Multimodal vision tool (`qwen2.5vl:3b`)
- Web search tool (Wikipedia REST + DuckDuckGo fallback)
- Weather tool (Open-Meteo, no API key)
- Custom calculator tool (AST-safe math evaluator)
- Local knowledge tool over Lab 4 cache and summaries
- Datetime tool (`zoneinfo`)
- NLP tool wrapping Lab 1-4 (translate / summarize / NER / sentiment)
- Multi-step reasoning loop (max 5 iterations, multiple tool calls per step)
- Per-chat persisted history and tool-call traces
- Conversational `/chat` mode and one-shot `/agent` mode
- Lab06 moderation pipeline: PII -> Bielik Guard-style classifier ->
  Qwen Guard-style decision -> sentiment -> NER -> action tools
- CSV audit logs for moderation decisions, actions, feedback,
  watchlist and user history

## Commands

### Lab 6 (New)

| Command                                                             | Description                                  |
| ------------------------------------------------------------------- | -------------------------------------------- |
| `/moderate "tekst" [user_id=...] [username=...]`                    | Full moderation pipeline with action logging |
| `/mod_policy_check "tekst"`                                         | Dry-run policy check without saving action   |
| `/mod_status <content_id>`                                          | Show saved moderation decision               |
| `/mod_history <user_id>`                                            | Show user moderation history and risk score  |
| `/mod_analytics`                                                    | Moderator BI report                          |
| `/mod_add_feedback <content_id> "comment" "APPROVE|REJECT|FLAG..."` | Save human moderator feedback                |
| `/mod_watchlist`                                                    | List users on moderation watchlist           |
| `/mod_train_on_feedback`                                            | Prepare feedback summary for future training |
| `/mod_help`                                                         | Show all commands                            |

### Lab 5 (Inherited)

| Command                                                                                                                          | Description                                               |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `/agent text="..."`                                                                                                              | One-shot agent run; can also receive a photo with caption |
| `/chat`                                                                                                                          | Toggle conversational mode (history kept across turns)    |
| `/chat_reset`                                                                                                                    | Clear conversation history for the current chat           |
| `/agent_history [n=5]`                                                                                                           | Last agent runs with their tool traces                    |
| `/tools`                                                                                                                         | List registered tools with descriptions                   |
| `/tool_calc expression="..."`                                                                                                    | Direct call: AST-safe calculator                          |
| `/tool_weather city="..."`                                                                                                       | Direct call: current weather (Open-Meteo)                 |
| `/tool_search query="..." [language=en\|pl]`                                                                                     | Direct call: web search                                   |
| `/tool_local_kb query="..."`                                                                                                     | Direct call: local knowledge base                         |
| `/tool_datetime [city=Tokyo\|tz=Europe/Warsaw]`                                                                                   | Direct call: current date/time                            |
| `/tool_nlp operation=<translate\|summarize\|extract_entities\|classify_sentiment> text="..." [target_language=..] [language=..]` | Direct call: Lab 1-4 NLP wrapper                          |
| `/tool_vision`                                                                                                                   | Send a photo with this caption to run the vision tool     |
| `/help`                                                                                                                          | Show all commands (Lab 1 + Lab 2 + Lab 3 + Lab 4 + Lab 5 + Lab 6) |

## Lab 6 Components

### Moderation pipeline

`moderation.pipeline.moderate_content(text, user_id, username)` runs:

1. `detect_private_info()` — local PII detector compatible with the
   `openai/privacy-filter` interface.
2. `classify_bielik_guard()` — local fallback for categories:
   `toxic`, `spam`, `hate_speech`, `self_harm`, `violence`, `sexual`,
   `clean`.
3. `classify_qwen_guard()` — Qwen Guard-style structured risk decision.
4. `analyze_sentiment_for_moderation()` — sentiment context from Lab03
   wrapper. Negative sentiment alone does not reject content.
5. `extract_moderation_entities()` — usernames, URLs, emails, phones and
   Lab04 NER context.
6. Ensemble decision and action tool execution:
   `approve_content`, `reject_content`, `flag_for_human_review`,
   `shadow_ban_user`, `add_to_watchlist`.

The Hugging Face models named in the brief are represented by adapters.
The default implementation is deterministic and local, so the lab works
offline and remains testable on a small machine.

### Moderation data

| Output              | Path                                            |
| ------------------- | ----------------------------------------------- |
| Decisions           | `moderation_data/moderation_log.csv`            |
| User history        | `moderation_data/user_moderation_history.csv`   |
| Human feedback      | `moderation_data/feedback_log.csv`              |
| Feedback train data | `moderation_data/train_data.csv`                |
| Action audit trail  | `moderation_data/moderation_actions.csv`        |
| Watchlist           | `moderation_data/watchlist.csv`                 |

### Offline smoke test

```bash
python _smoke_moderation.py
```

### Lab 4 (Inherited)

| Command                                                                                                                     | Description                                      |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `/language_detect text="..."`                                                                                               | Detect text language with confidence             |
| `/ner method=<spacy\|stanza> text="..." [language=auto\|en\|pl]`                                                            | Named Entity Recognition                         |
| `/nel text="..." [language=auto\|en\|pl]`                                                                                   | NER + Wikidata candidate search                  |
| `/ned entity="..." context="..." [language=auto\|en\|pl]`                                                                   | Disambiguate one entity in a context             |
| `/analyze_entities text="..." [link=true\|false] [language=auto\|en\|pl]`                                                   | Full pipeline: spaCy + Stanza + optional linking |
| `/translate text="..." target_lang=<en\|pl\|de\|fr\|es> [source_lang=auto]`                                                 | Machine translation (Opus-MT)                    |
| `/summarize text="..." [summary_type=abstractive\|extractive\|bullets\|custom] [length=short\|medium\|long] [prompt="..."]` | Summarize via local LLM                          |
| `/knowledge_graph text="..." [language=auto\|en\|pl]`                                                                       | Build & render entity co-occurrence graph        |

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

## Lab 5 Components Kept In Lab06

### Agent loop

`agent.Agent.run(user_text, images=None, history=None)` orchestrates
tool calling against a local Ollama model:

1. Build messages: system prompt + recent history + user turn. If a
   photo is attached, run `analyze_image` first and add its text result.
2. Call `/api/chat` with the full set of registered tool schemas.
3. If the model returns `tool_calls`, execute each call via
   `tools.call_tool(name, arguments)` and append the result as a
   `{role: "tool", name, content: <json>}` message.
4. Loop up to `MAX_AGENT_ITERATIONS = 5`. Stop as soon as the model
   answers without further tool calls.
5. Return `{answer, tool_trace, iterations, messages}`.

The controller model is `qwen2.5:1.5b` — the smallest Qwen with
reliable native tool calling. Bump to `qwen2.5:7b` in `config.py`
for richer reasoning.

### Tool registry

| #   | Tool            | Function                            | Backend                                                 |
| --- | --------------- | ----------------------------------- | ------------------------------------------------------- |
| 1   | Web Search      | `web_search(query, language)`       | Wikipedia REST + DuckDuckGo fallback                    |
| 2   | Vision          | `analyze_image(image_path, prompt)` | Ollama `qwen2.5vl:3b` (multimodal)                      |
| 3   | Calculator      | `calculator(expression)`            | AST-safe math (`sqrt`, `sin`, `pi`, …)                  |
| 4   | Local Knowledge | `local_knowledge(query, max_hits)`  | Lab 4 NEL cache + Lab 4 summaries + Lab 1 sentences     |
| 5   | Weather         | `get_weather(city)`                 | Open-Meteo (geocoding + forecast, no API key)           |
| 6   | Datetime        | `datetime_now(city, tz)`            | Open-Meteo geocoding + Python `zoneinfo`                |
| 7   | NLP             | `nlp_tools(operation, text, …)`     | Lab 1-4: detect / translate / summary / sentiment / NER |
| 8   | Moderation      | `approve_content(...)`, `reject_content(...)`, ... | Lab06 action tools and CSV audit log |

Each tool advertises an OpenAI-style JSON schema in `lab5/tools/<name>.SCHEMA`
and is picked up automatically by `get_tools_payload()`.

### Adding a new tool

1. Drop `lab5/tools/my_tool.py` exposing a callable and a `SCHEMA` dict.
2. Register it in `lab5/tools/__init__.py` `TOOL_REGISTRY`.

No agent-side changes are required.

### Session store

Per-chat history and tool traces are persisted as JSON in
`cache/sessions/<chat_id>.json`. `get_history(max_turns=10)` is used
by the chat mode; `append_run` records each agent run with a trimmed
tool trace (capped at 50 entries per chat).

### Vision pipeline

`/tool_vision` and `/agent` (with photo) download the highest-res
photo from Telegram. `/tool_vision` calls `analyze_image` directly;
`/agent` first runs the same vision tool and then feeds the textual
image description into the agent loop. The controller model can then
reason over that description and chain additional tools in follow-up
iterations.

## Output Files

| Output                | Path                                                            |
| --------------------- | --------------------------------------------------------------- |
| Per-chat sessions     | `cache/sessions/<chat_id>.json`                                 |
| NEL cache (Lab 4)     | `cache/nel_cache.json`                                          |
| Knowledge graph plots | `lab4plots/knowledge_graph_<timestamp>.png`                     |
| Saved summaries       | `lab4results/summaries/summary_<type>_<length>_<timestamp>.txt` |
| Moderation decisions  | `moderation_data/moderation_log.csv`                            |
| Moderation feedback   | `moderation_data/feedback_log.csv`                              |
| Moderation train data | `moderation_data/train_data.csv`                                |
| Moderation watchlist  | `moderation_data/watchlist.csv`                                 |

## Project Structure

```
Lab06/
├── bot.py                 — Entry point (Telegram polling)
├── commands.py            — Lab 6 handlers + Lab 5 agent + Lab 1-4 delegation
├── config.py              — Configuration & constants (Lab 5/6)
├── utils.py               — Parsing, formatting, logging, truncation
├── moderation/
│   ├── models.py          — PII, guard, sentiment and entity adapters
│   ├── pipeline.py        — Multi-stage moderation decision flow
│   ├── actions.py         — Moderation action tools
│   └── storage.py         — CSV logs and analytics
├── lab5/
│   ├── agent.py           — Multi-step tool-calling loop
│   ├── ollama_client.py   — Ollama /api/chat wrapper (tools + images)
│   ├── session_store.py   — Per-chat history & tool-trace persistence
│   └── tools/
│       ├── __init__.py    — TOOL_REGISTRY, get_tools_payload, call_tool
│       ├── web_search.py  — Wikipedia REST + DuckDuckGo
│       ├── vision.py      — Ollama vision call
│       ├── calculator.py  — AST-safe math evaluator
│       ├── local_kb.py    — Local knowledge search
│       ├── weather.py     — Open-Meteo client
│       ├── datetime_tool.py — zoneinfo wrapper
│       ├── nlp_tools.py   — Lab 1-4 NLP wrapper
│       └── moderation_tools.py — Lab06 action schemas
├── lab4/                  — Lab 4 modules (NER, NEL, NED, translation, …)
│   ├── commands.py
│   ├── ner.py / nel.py / ned.py
│   ├── translation.py / summarization.py
│   ├── language_detect.py / knowledge_graph.py
│   └── __init__.py
├── requirements.txt       — Python dependencies
├── .gitignore
├── .env                   — Telegram bot token (not in repo)
└── cache/
    ├── sessions/          — Per-chat agent history (gitignored)
    └── nel_cache.json     — NEL cache (gitignored)
```

`lab1/`, `lab2/` and `lab3/` are inherited from the Lab 4 tree and
imported as subpackages by `lab4/commands.py`.

## Installation

1. Clone the repo and switch to the `Lab06` branch:

   ```bash
   git clone https://github.com/jaugustyn/NLP_Labs.git
   cd NLP_Labs
   git checkout Lab06
   cd Lab06
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
   ollama pull qwen2.5:1.5b      # controller (native tool calling)
   ollama pull qwen2.5vl:3b      # vision tool (~3.2 GB)
   ```

5. Create `.env` in `Lab06/`:

   ```
   TELEGRAM_BOT_TOKEN=<YOUR_TOKEN>
   ```

6. Start the bot:

   ```bash
   python bot.py
   ```

> **Note:** First use of `/translate`, `/ner method=stanza`, the
> `transformer` / `stanza` sentiment methods and the vision model
> will download models — allow a few minutes per backend.

## Usage Examples

```text
# Automatic approve
/moderate "Uwielbiam ten produkt, najlepszy zakup!" user_id=user_123

# Automatic reject: toxic/self-harm
/moderate "Jesteś głupi i powinieneś się zabić" user_id=user_456

# Automatic reject: personal data
/moderate "Mój email to john@example.com i numer +48 123 456 789" user_id=user_789

# Human review: political/opinion edge case
/moderate "Ci politykanci to wszystko złodzieje!" user_id=user_321

# Dry-run policy check without saving an action
/mod_policy_check "Kup teraz na http://spam.example i wygraj darmowe krypto"

# Inspect moderation state
/mod_status mod_abc123
/mod_history user_456
/mod_watchlist
/mod_analytics

# Human moderator feedback loop
/mod_add_feedback mod_abc123 "To legalne zaproszenie, nie spam" "APPROVE"
/mod_train_on_feedback
/mod_help
```

### Known limitations

- **Safety models**: Lab06 exposes interfaces matching
  `openai/privacy-filter`, `speakleash/Bielik-Guard-0.1B-v1.0` and
  Qwen Guard, but the default backend is a deterministic local fallback.
  This keeps the lab runnable offline on a small machine.
- **Moderation policy**: rule-based fallbacks are intentionally
  conservative. Ambiguous content is sent to `FLAG_FOR_REVIEW` instead
  of being forcefully accepted or rejected.
- **Feedback training**: `/mod_train_on_feedback` prepares and reports
  saved feedback examples. It does not automatically fine-tune a model.

## System Requirements

- Python 3.10+
- Telegram Bot API token
- Lab06 moderation core runs locally and does not require an external API
- Optional inherited Lab05 agent features use Ollama with `qwen2.5:1.5b`
  and `qwen2.5vl:3b`
- Internet access is only needed for first-time dependency/model downloads
  and inherited web/weather tools
