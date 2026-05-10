"""Per-chat conversation history + agent run log persisted as JSON."""
import json
import os
import time

from config import SESSIONS_DIR, AGENT_HISTORY_TURNS


def _path(chat_id):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    return os.path.join(SESSIONS_DIR, f"{chat_id}.json")


def _empty():
    return {"history": [], "runs": []}


def load_session(chat_id):
    path = _path(chat_id)
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty()
        data.setdefault("history", [])
        data.setdefault("runs", [])
        return data
    except Exception:
        return _empty()


def _save(chat_id, data):
    path = _path(chat_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_history(chat_id, max_turns=AGENT_HISTORY_TURNS):
    """Return the last N user/assistant message pairs."""
    data = load_session(chat_id)
    hist = data.get("history") or []
    # Each turn = 2 messages (user + assistant). Keep last 2*N.
    return hist[-2 * max_turns:]


def append_turn(chat_id, user_text, assistant_text):
    data = load_session(chat_id)
    data["history"].append({"role": "user", "content": user_text})
    data["history"].append({"role": "assistant", "content": assistant_text})
    # Truncate
    data["history"] = data["history"][-2 * AGENT_HISTORY_TURNS:]
    _save(chat_id, data)


def append_run(chat_id, user_text, result):
    """Persist an /agent or /chat run with its tool trace."""
    data = load_session(chat_id)
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_text,
        "answer": result.get("answer"),
        "iterations": result.get("iterations"),
        "tool_trace": [
            {"tool": t.get("tool"), "arguments": t.get("arguments")}
            for t in (result.get("tool_trace") or [])
        ],
    }
    data["runs"].append(entry)
    data["runs"] = data["runs"][-50:]
    _save(chat_id, data)


def reset_session(chat_id):
    path = _path(chat_id)
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except Exception:
            return False
    return True


def list_runs(chat_id, n=5):
    data = load_session(chat_id)
    return (data.get("runs") or [])[-n:]
