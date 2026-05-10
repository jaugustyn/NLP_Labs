# NLP Bot — Laboratory 5 (Tool / Function Calling Agent)

Extension of **Lab 1 + Lab 2 + Lab 3 + Lab 4** with a **tool-calling agent**
running on a local LLM (Ollama). The model autonomously decides **whether
and which tool to call**, passes structured JSON arguments, integrates
multiple tool results and produces a final natural-language answer in
the user's language. Multi-step reasoning across tools is supported, and
all interactions are persisted per chat.

Spec: <https://github.com/mkrzywda/WSEI_NLP/blob/main/Lab05.md>

---

## Architecture

```
Lab05/
├── bot.py              # Telegram entry point
├── commands.py         # Lab 5 handlers (/agent, /chat, /tool_*, ...)
├── config.py           # central config (Ollama URLs, model names, paths)
├── agent.py            # tool-calling loop (max 5 iterations)
├── ollama_client.py    # /api/chat wrapper with tools + multimodal images
├── session_store.py    # per-chat history & tool-trace persistence
├── utils.py
├── tools/              # 7 tools, each with SCHEMA + callable
│   ├── __init__.py     #   TOOL_REGISTRY, get_tools_payload, call_tool
│   ├── web_search.py
│   ├── vision.py
│   ├── calculator.py
│   ├── local_kb.py
│   ├── weather.py
│   ├── datetime_tool.py   (bonus)
│   └── nlp_tools.py       (bonus — wraps Lab 1-4)
├── lab4/               # Lab 4 modules (NER, NEL, NED, translation, ...)
└── cache/sessions/     # <chat_id>.json (history + tool runs)
```

`lab1/`, `lab2/`, `lab3/` live in the original Lab04 tree (Lab 5 only
adds the agent layer on top — Lab 4 logic was moved into `Lab05/lab4/`
so that Lab 5 sources sit at the package root and Lab 4 still works as
before via the `lab4.commands.register_handlers` adapter).

### Agent loop

`agent.Agent.run(user_text, images=None, history=None)`:

1. Build messages: system prompt → recent history → user turn (with
   base64-encoded images attached, if any).
2. Call `ollama_client.chat(messages, tools=<all schemas>)`.
3. If the model returns `tool_calls`, execute each via
   `tools.call_tool(name, arguments)` and append the result as
   `{role: "tool", name, content: <json>}`.
4. Loop up to `MAX_AGENT_ITERATIONS = 5`. Stop as soon as the model
   answers without further tool calls.
5. Return `{answer, tool_trace, iterations, messages}`.

### Adding a new tool

1. Drop `tools/my_tool.py` exposing a callable and a JSON `SCHEMA`
   (OpenAI function-calling shape).
2. Register it in `tools/__init__.py` `TOOL_REGISTRY`.

That's it — no agent-side changes needed.

---

## Required tools (spec ✅)

| #   | Tool             | Function                  | Source                                                  |
| --- | ---------------- | ------------------------- | ------------------------------------------------------- |
| 1   | Web Search       | `web_search(query, lang)` | Wikipedia REST + DuckDuckGo fallback                    |
| 2   | Vision           | `analyze_image(path,...)` | Ollama `qwen2.5vl:3b` (multimodal)                      |
| 3   | Custom (calc)    | `calculator(expression)`  | AST-safe math (`sqrt`, `sin`, `pi`, …)                  |
| 4   | Local Knowledge  | `local_knowledge(query)`  | NEL cache + Lab 4 summaries + sentences                 |
| 5   | Weather          | `get_weather(city)`       | Open-Meteo (no API key)                                 |
| 6   | Datetime (bonus) | `datetime_now(tz)`        | Python `zoneinfo`                                       |
| 7   | NLP (bonus)      | `nlp_tools(operation,…)`  | Lab 1-4: detect / translate / summary / sentiment / NER |

All tools advertise an OpenAI-style JSON schema in
`tools/<name>.SCHEMA` and are picked up automatically by
`get_tools_payload()`.

---

## Telegram commands

### Lab 5 — agent layer

| Command                    | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| `/agent <text>`            | One-shot agent run — model decides which tools to call |
| `/chat`                    | Toggle conversational mode (every message → agent)     |
| `/chat_reset`              | Clear conversational history for the current chat      |
| `/agent_history`           | Show the last few agent runs with their tool traces    |
| `/tools`                   | List registered tools with descriptions                |
| `/tool_calc <expr>`        | Direct call: calculator                                |
| `/tool_weather <city>`     | Direct call: weather                                   |
| `/tool_search <query>`     | Direct call: web search                                |
| `/tool_local_kb <query>`   | Direct call: local knowledge base                      |
| `/tool_datetime [tz]`      | Direct call: current date/time                         |
| `/tool_nlp <op> \| <text>` | Direct call: NLP (translate / summarize / …)           |
| `/tool_vision`             | Reply to a photo with this caption to run vision       |

### From earlier labs (still available)

`/help`, `/start`, `/sentiment`, `/wordcount`, `/preprocess`, `/zipf`,
`/ner`, `/nel`, `/ned`, `/translate`, `/summarize`, `/detect_lang`,
`/knowledge_graph`, …

`/help` lists everything.

---

## Multi-tool scenarios (spec §"Rozszerzone scenariusze")

```
/agent Kto jest CEO Tesli?
   → web_search

/agent Jaka jest pogoda w Warszawie?
   → get_weather("Warsaw")

/agent Czy dziś jest dobra pogoda na spacer w Warszawie?
   → get_weather  → model interprets

/agent Porównaj pogodę w Warszawie i Paryżu
   → get_weather("Warsaw") + get_weather("Paris")  → comparison

/agent Czy pogoda w Warszawie jest typowa dla maja?
   → get_weather + web_search("average weather Warsaw May")  → reasoning

/agent Ile to (12 + 7) * sqrt(81)?
   → calculator

/agent Co wiemy lokalnie o Apple?
   → local_knowledge

/tool_vision  (reply to photo)  Opisz co widać i odczytaj tekst
   → analyze_image
```

---

## Setup

```powershell
# 1) Python deps
pip install -r requirements.txt

# 2) Ollama — required for both Lab 4 (summarization) and Lab 5 (agent)
ollama serve
ollama pull qwen2.5:1.5b      # controller — native tool calling
ollama pull qwen2.5vl:3b      # vision tool (~3.2 GB)

# 3) .env
#   TELEGRAM_TOKEN=...

# 4) Run
python bot.py
```

---

## Spec compliance — Lab05.md

| Wymaganie                                           |            Status             |
| --------------------------------------------------- | :---------------------------: |
| ≥ 5 tools (web / vision / custom / local / weather) |             ✅ 7              |
| Function calling via Ollama                         |              ✅               |
| JSON schema per tool                                |              ✅               |
| Multimodal (Vision) model                           |       ✅ `qwen2.5vl:3b`       |
| Multi-tool reasoning (combine tool outputs)         |   ✅ agent loop, max 5 iter   |
| Local LLM only (no external LLM API)                |              ✅               |
| Multi-step reasoning                                |              ✅               |
| Interaction history persisted                       |     ✅ `cache/sessions/`      |
| Easy extension                                      |      ✅ `TOOL_REGISTRY`       |
| Separation of LLM logic vs tools                    | ✅ `agent.py` vs `tools/*.py` |
| Scenarios 5–8 (Weather, +reasoning, multi-tool)     |              ✅               |

---

## Notes / limitations

- `qwen2.5:1.5b` is the smallest Qwen with reliable native tool
  calling; for richer reasoning bump to `qwen2.5:7b` in `config.py`.
- The vision model (`qwen2.5vl:3b`) is large — pull it once and reuse.
- `web_search` uses Wikipedia REST + DuckDuckGo Instant Answer; it
  intentionally avoids HTML scraping to keep the tool deterministic.
- `local_knowledge` indexes Lab 4 NEL cache + Lab 4 summary outputs +
  Lab 1 `sentences.json` — drop more `.txt` files into
  `Lab04/lab4results/summaries/` to enrich the KB.
