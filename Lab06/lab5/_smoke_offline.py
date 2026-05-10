"""Smoke tests for Lab 5 tools that do not need network or Ollama."""

import json

from lab5 import tools


def show(name, result):
    print(f"=== {name} ===")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:800])
    print()


def main():
    show("calc 2+2", tools.call_tool("calculator", {"expression": "2+2"}))
    show(
        "calc sqrt",
        tools.call_tool("calculator", {"expression": "2 + 2 * sqrt(16)"}),
    )
    show(
        "calc math.pi",
        tools.call_tool("calculator", {"expression": "math.pi * 2"}),
    )
    show(
        "calc bad",
        tools.call_tool(
            "calculator",
            {"expression": "__import__('os').system('echo hax')"},
        ),
    )
    show("calc huge pow", tools.call_tool("calculator", {"expression": "9**999"}))

    show("datetime", tools.call_tool("datetime_now", {}))
    show("datetime tz", tools.call_tool("datetime_now", {"tz": "Europe/Warsaw"}))
    show("local_kb paris", tools.call_tool("local_knowledge", {"query": "Paris"}))
    show(
        "sentiment negation",
        tools.call_tool(
            "nlp_tools",
            {
                "operation": "classify_sentiment",
                "text": "Bardzo nie polecam tego filmu!",
            },
        ),
    )
    show(
        "sentiment adjective",
        tools.call_tool(
            "nlp_tools",
            {
                "operation": "classify_sentiment",
                "text": "Ten seans był niesamowicie głupi!",
            },
        ),
    )

    print("Registered tools:", tools.list_tool_names())
    print("Total schemas:", len(tools.get_tools_payload()))

    for schema in tools.get_tools_payload():
        assert schema["type"] == "function"
        function_schema = schema["function"]
        assert "name" in function_schema and "parameters" in function_schema
    print("Schemas OK")


if __name__ == "__main__":
    main()
