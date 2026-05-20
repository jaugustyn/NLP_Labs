"""Offline smoke test for Lab 5 tool-calling safety nets."""

import json
import os
import tempfile

from lab5 import agent as agent_mod
from lab5 import ollama_client
from lab5 import session_store
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


def main():
    original_chat = ollama_client.chat
    original_registry = dict(tools_mod.TOOL_REGISTRY)
    original_sessions_dir = session_store.SESSIONS_DIR
    original_lab1_candidates = list(local_kb._LAB1_SENTENCES_CANDIDATES)

    try:
        ollama_client.chat = lambda **kwargs: {"message": {"content": "Nie wiem."}}
        tools_mod.TOOL_REGISTRY["get_weather"] = (
            lambda city: {
                "city": city,
                "country": "Testland",
                "temperature_c": 21.5,
                "wind_kmh": 5,
                "description": "clear sky",
                "observed_at": "2026-05-21T12:00",
            },
            _fake_schema("get_weather", ["city"]),
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
        assert "Warszawie" in weather["answer"]
        assert "Paryżu" in weather["answer"]

        calc = agent.run("Ile to 2+2*sqrt(16)?")
        assert calc["tool_trace"][0]["tool"] == "calculator"
        assert "10" in calc["answer"]

        dt = agent.run("Która godzina jest teraz w Tokio?")
        assert dt["tool_trace"][0]["tool"] == "datetime_now"
        assert dt["tool_trace"][0]["arguments"]["city"] == "Tokio"

        sentiment = agent.run('Jaki jest sentyment tekstu "Bardzo nie polecam"?')
        assert sentiment["tool_trace"][0]["tool"] == "nlp_tools"
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
            kb = local_kb.local_knowledge("Steve Jobs")
            assert kb["total_hits"] == 1
    finally:
        ollama_client.chat = original_chat
        tools_mod.TOOL_REGISTRY.clear()
        tools_mod.TOOL_REGISTRY.update(original_registry)
        session_store.SESSIONS_DIR = original_sessions_dir
        local_kb._LAB1_SENTENCES_CANDIDATES = original_lab1_candidates

    print("Lab5 smoke OK")


if __name__ == "__main__":
    main()
