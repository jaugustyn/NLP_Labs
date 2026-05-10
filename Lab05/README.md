# NLP Bot — Laboratory 5 (Tool / Function Calling Agent)

Extension of Lab 1 + Lab 2 + Lab 3 + Lab 4 bot with a tool-calling
agent running on a local LLM (Ollama). The model autonomously decides
whether to call a tool, picks the right one, passes structured JSON
arguments, integrates results from multiple tools and produces a final
natural-language answer in the user's language.

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

## Commands

### Lab 5 (New)

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
| `/help`                                                                                                                          | Show all commands (Lab 1 + Lab 2 + Lab 3 + Lab 4 + Lab 5) |

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

## Lab 5 Components

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

Each tool advertises an OpenAI-style JSON schema in `tools/<name>.SCHEMA`
and is picked up automatically by `get_tools_payload()`.

### Adding a new tool

1. Drop `tools/my_tool.py` exposing a callable and a `SCHEMA` dict.
2. Register it in `tools/__init__.py` `TOOL_REGISTRY`.

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

## Project Structure

```
Lab05/
├── bot.py                 — Entry point (Telegram polling)
├── commands.py            — Lab 5 handlers + delegation to Lab 1-4
├── config.py              — Configuration & constants (Lab 5)
├── utils.py               — Parsing, formatting, logging, truncation
├── ollama_client.py       — Ollama /api/chat wrapper (tools + images)
├── agent.py               — Multi-step tool-calling loop
├── session_store.py       — Per-chat history & tool-trace persistence
├── tools/
│   ├── __init__.py        — TOOL_REGISTRY, get_tools_payload, call_tool
│   ├── web_search.py      — Wikipedia REST + DuckDuckGo
│   ├── vision.py          — Ollama vision call
│   ├── calculator.py      — AST-safe math evaluator
│   ├── local_kb.py        — Local knowledge search
│   ├── weather.py         — Open-Meteo client
│   ├── datetime_tool.py   — zoneinfo wrapper
│   └── nlp_tools.py       — Lab 1-4 NLP wrapper
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

1. Clone the repo and switch to the `Lab05` branch:

   ```bash
   git clone https://github.com/jaugustyn/NLP_Labs.git
   cd NLP_Labs
   git checkout Lab05
   cd Lab05
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

5. Create `.env` in `Lab05/`:

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
# One-shot agent — model picks the tool
/agent text="Kto jest CEO Tesli?"
/agent text="Jaka jest pogoda w Warszawie?"
/agent text="Czy dziś jest dobra pogoda na spacer w Warszawie?"
/agent text="Porównaj pogodę w Warszawie i Paryżu"
/agent text="Czy pogoda w Warszawie jest typowa dla maja?"
/agent text="Ile to (12 + 7) * sqrt(81)?"
/agent text="Co wiemy lokalnie o Apple?"
/agent text="Która godzina jest teraz w Tokio?"

# Conversational mode — history is kept until /chat_reset
/chat
Cześć, kim jest Maria Skłodowska-Curie?
A gdzie się urodziła?
Podsumuj to w jednym zdaniu po angielsku.
/chat_reset
/chat

# Vision — attach a photo with caption
/agent text="Opisz co widać i odczytaj tekst"   (photo attached)
/tool_vision                                    (photo + caption)

# Direct tool calls — for testing & debugging
/tool_calc expression="2 + 2 * sqrt(16)"
/tool_calc expression="sin(pi/2) + log(e)"
/tool_weather city="Warsaw"
/tool_weather city="Paryż"
/tool_search query="Maria Skłodowska-Curie" language=pl
/tool_search query="OpenAI" language=en
/tool_local_kb query="Apple"
/tool_local_kb query="Paris Hilton"
/tool_datetime
/tool_datetime city="Barcelona"
/tool_datetime tz="Asia/Tokyo"
/tool_nlp operation=translate text="Dziś jest ładny dzień" target_language=en
/tool_nlp operation=summarize text="Long article text..." length=short
/tool_nlp operation=extract_entities text="Elon Musk founded SpaceX in 2002"
/tool_nlp operation=classify_sentiment text="Bardzo polecam ten film!"

# Inherited Lab 4 commands work unchanged
/language_detect text="Warszawa to stolica Polski"
/ner method=stanza text="Frédéric Chopin urodził się w Polsce" language=pl
/ned entity="Paris" context="Paris Hilton attended the gala in New York"
/translate text="Bonjour le monde" target_lang=en
/summarize text="..." summary_type=bullets length=short
/knowledge_graph text="Elon Musk founded SpaceX. Steve Jobs founded Apple."

# Inspect agent activity
/tools
/agent_history n=10
```

### Known limitations

- **Tool-calling model size**: `qwen2.5:1.5b` occasionally emits a
  malformed JSON argument or skips a tool call on the first iteration.
  The agent loop allows up to 5 iterations and surfaces tool errors
  back to the model, which usually self-corrects on the next turn.
  For higher reliability bump `AGENT_MODEL` to `qwen2.5:7b`.
- **Vision model size**: `qwen2.5vl:3b` is ~3.2 GB; first inference
  is slow on CPU. The image cap is 4 MB (`VISION_MAX_IMAGE_BYTES`);
  larger photos are rejected with a clear error.
- **Web search**: `web_search` uses Wikipedia REST + DuckDuckGo
  Instant Answer; it intentionally avoids HTML scraping. Queries with
  no Wikipedia article and no DuckDuckGo abstract return an empty
  result and the model falls back to its own knowledge.
- **Open-Meteo**: city geocoding occasionally returns an unexpected
  match for ambiguous names (e.g. multiple "Springfield"). The tool
  returns `country` and coordinates so the model can confirm.
- **Local knowledge base**: indexes only Lab 4 NEL cache, Lab 4
  saved summaries, and Lab 1 `sentences.json`. Drop more `.txt`
  files into `lab4results/summaries/` to enrich it.
- **History**: only the last `AGENT_HISTORY_TURNS = 10` turns are
  fed back to the model; older turns remain on disk for
  `/agent_history` but are not part of the prompt.

## System Requirements

- Python 3.10+
- Telegram Bot API token
- Ollama installed locally with `qwen2.5:1.5b` and `qwen2.5vl:3b` pulled
- ~6 GB disk space (Lab 4 models + agent + vision model)
- Internet access for first-time model downloads, Wikidata, Wikipedia, DuckDuckGo and Open-Meteo
