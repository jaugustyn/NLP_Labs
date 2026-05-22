"""Offline smoke test for Lab 5 tool-calling safety nets."""

import json
import os
import tempfile

from lab5 import agent as agent_mod
from lab5 import commands as commands_mod
from lab5 import ollama_client
from lab5 import session_store
from lab5 import tool_planner
from lab5 import tools as tools_mod
from lab5.tools import local_kb


def _fake_schema(name, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in required},
                "required": required,
            },
        },
    }


def _fake_weather(city):
    folded = (city or "").lower()
    if "wars" in folded:
        city_name = "Warszawa"
    elif "par" in folded:
        city_name = "Paryż"
    else:
        city_name = city
    return {
        "city": city_name,
        "country": "Testland",
        "temperature_c": 21.5,
        "wind_kmh": 5,
        "description": "clear sky",
        "observed_at": "2026-05-21T12:00",
    }


def main():
    original_chat = ollama_client.chat
    original_registry = dict(tools_mod.TOOL_REGISTRY)
    original_sessions_dir = session_store.SESSIONS_DIR
    original_lab1_candidates = list(local_kb._LAB1_SENTENCES_CANDIDATES)

    try:
        ollama_client.chat = lambda **kwargs: {"message": {"content": "Nie wiem."}}
        tools_mod.TOOL_REGISTRY["get_weather"] = (
            _fake_weather,
            _fake_schema("get_weather", ["city"]),
        )
        tools_mod.TOOL_REGISTRY["web_search"] = (
            lambda query, language="en": {
                "query": query,
                "language": language,
                "summary": "Average climate weather information for the requested month.",
            },
            _fake_schema("web_search", ["query"]),
        )
        tools_mod.TOOL_REGISTRY["datetime_now"] = (
            lambda city=None, tz=None: {
                "city": city,
                "country": "Japan" if city else None,
                "timezone": "Asia/Tokyo" if city else tz,
                "time": "19:00:00",
                "date": "2026-05-21",
                "weekday": "Thursday",
            },
            {
                "type": "function",
                "function": {
                    "name": "datetime_now",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "tz": {"type": "string"},
                        },
                    },
                },
            },
        )

        agent = agent_mod.Agent(max_iters=1)
        weather = agent.run("Porównaj pogodę w Warszawie i Paryżu")
        assert [call["tool"] for call in weather["tool_trace"]] == [
            "get_weather",
            "get_weather",
        ]
        assert "Warszawa" in weather["answer"]
        assert "Paryż" in weather["answer"]

        missing_weather = agent.run("Jaka jest pogoda?")
        assert not missing_weather["tool_trace"]
        assert "Podaj miasto" in missing_weather["answer"]

        typical = agent.run("Czy pogoda w Warszawie jest typowa dla marca?")
        assert [call["tool"] for call in typical["tool_trace"]] == [
            "get_weather",
            "web_search",
        ]
        assert "marca" in typical["tool_trace"][1]["arguments"]["query"]
        assert typical["tool_trace"][1]["arguments"]["language"] == "pl"

        missing_translation = agent.run("Przetłumacz na niemiecki dobry wieczór")
        assert not missing_translation["tool_trace"]
        assert "kod ISO" in missing_translation["answer"]
        missing_target = tools_mod.call_tool(
            "nlp_tools",
            {"operation": "translate", "text": "dobry wieczór"},
        )
        assert "target_language is required" in missing_target["error"]
        assert tool_planner.extract_translation_target(
            "Przetłumacz na de dobry wieczór"
        ) == "de"

        duplicate_responses = iter([
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Warsaw"},
                            }
                        },
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            }
                        },
                    ],
                }
            },
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Warszawie"},
                            }
                        },
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paryżu"},
                            }
                        },
                    ],
                }
            },
        ])
        ollama_client.chat = lambda **kwargs: next(duplicate_responses)
        weather = agent_mod.Agent(max_iters=3).run(
            "Porównaj pogodę w Warszawie i Paryżu"
        )
        assert len(weather["tool_trace"]) == 2
        assert weather["answer"].count("Warszawa") == 1
        assert weather["answer"].count("Paryż") == 1
        ollama_client.chat = lambda **kwargs: {"message": {"content": "Nie wiem."}}

        calc = agent.run("Ile to 2+2*sqrt(16)?")
        assert calc["tool_trace"][0]["tool"] == "calculator"
        assert "10" in calc["answer"]

        dt = agent.run("Która godzina jest teraz w Tokio?")
        assert dt["tool_trace"][0]["tool"] == "datetime_now"
        assert dt["tool_trace"][0]["arguments"]["city"] == "Tokio"

        parsed = commands_mod._extract_agent_text(
            r'text="Jaki jest sentyment tekstu \"Bardzo nie polecam\"?"'
        )
        assert parsed == 'Jaki jest sentyment tekstu "Bardzo nie polecam"?'
        sentiment = agent.run(parsed)
        assert sentiment["tool_trace"][0]["tool"] == "nlp_tools"
        assert sentiment["tool_trace"][0]["arguments"]["text"] == "Bardzo nie polecam"
        assert "negatywny" in sentiment["answer"]

        bad = tools_mod.call_tool("calculator", {"unexpected": "2+2"})
        assert "Missing required" in bad["error"]

        with tempfile.TemporaryDirectory() as temp_dir:
            session_store.SESSIONS_DIR = temp_dir
            session_store.append_run(123, "test", calc)
            runs = session_store.list_runs(123)
            assert runs[0]["tool_trace"][0].get("result")

            lab1_path = os.path.join(temp_dir, "sentences.json")
            with open(lab1_path, "w", encoding="utf-8") as file:
                json.dump(
                    [{"text": "Steve Jobs founded Apple", "class": "pozytywny"}],
                    file,
            )
            local_kb._LAB1_SENTENCES_CANDIDATES = [lab1_path]
            kb = local_kb.local_knowledge("Steve Jobs", max_hits=20)
            assert any(hit.get("source") == "lab1_sentences" for hit in kb["hits"])
    finally:
        ollama_client.chat = original_chat
        tools_mod.TOOL_REGISTRY.clear()
        tools_mod.TOOL_REGISTRY.update(original_registry)
        session_store.SESSIONS_DIR = original_sessions_dir
        local_kb._LAB1_SENTENCES_CANDIDATES = original_lab1_candidates

    print("Lab5 smoke OK")


if __name__ == "__main__":
    main()
