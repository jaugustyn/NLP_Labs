"""Smoke tests for Lab 5 tools that don't need network/Ollama."""
import json
import sys

import tools as t


def show(name, result):
    print(f"=== {name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:800])
    print()


# 1. Calculator
show("calc 2+2", t.call_tool("calculator", {"expression": "2+2"}))
show("calc sqrt", t.call_tool("calculator", {"expression": "2 + 2 * sqrt(16)"}))
show("calc math.pi", t.call_tool("calculator", {"expression": "math.pi * 2"}))
show("calc bad", t.call_tool("calculator", {"expression": "__import__('os').system('echo hax')"}))
show("calc huge pow", t.call_tool("calculator", {"expression": "9**999"}))

# 2. Datetime
show("datetime", t.call_tool("datetime_now", {}))
show("datetime tz", t.call_tool("datetime_now", {"tz": "Europe/Warsaw"}))

# 3. Local KB
show("local_kb paris", t.call_tool("local_knowledge", {"query": "Paris"}))

# 4. NLP sentiment
show(
    "sentiment negation",
    t.call_tool("nlp_tools", {
        "operation": "classify_sentiment",
        "text": "Bardzo nie polecam tego filmu!",
    }),
)
show(
    "sentiment adjective",
    t.call_tool("nlp_tools", {
        "operation": "classify_sentiment",
        "text": "Ten seans był niesamowicie głupi!",
    }),
)

# 5. Tool list
print("Registered tools:", t.list_tool_names())
print("Total schemas:", len(t.get_tools_payload()))

# 6. Schema sanity check
for s in t.get_tools_payload():
    assert s["type"] == "function"
    fn = s["function"]
    assert "name" in fn and "parameters" in fn, fn
print("Schemas OK")
