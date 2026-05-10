"""Thin wrapper around Ollama's /api/chat — supports tool calling and
multimodal (image) input. Used by the agent and the vision tool."""
import base64
import os

import requests

from config import (
    OLLAMA_CHAT_URL,
    OLLAMA_TIMEOUT,
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    VISION_MAX_IMAGE_BYTES,
)


def _encode_image(path):
    """Return base64-encoded contents of an image file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    size = os.path.getsize(path)
    if size > VISION_MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large ({size} B > {VISION_MAX_IMAGE_BYTES} B)"
        )
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def chat(messages, tools=None, model=None, images=None,
         temperature=None, timeout=None):
    """Call Ollama /api/chat. Returns the parsed JSON response.

    Args:
        messages: list of {role, content, [tool_call_id], [name]} dicts.
        tools: optional list of tool schemas (OpenAI-style).
        model: ollama model tag (defaults to AGENT_MODEL).
        images: optional list of file paths; if provided, attached to
            the *last* user message as base64-encoded `images`.
        temperature: sampling temperature (defaults to AGENT_TEMPERATURE).
        timeout: HTTP timeout seconds.
    """
    msgs = [dict(m) for m in messages]  # shallow copy

    if images:
        # Attach to the last user message
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["images"] = [_encode_image(p) for p in images]
                break

    payload = {
        "model": model or AGENT_MODEL,
        "messages": msgs,
        "stream": False,
        "options": {
            "temperature": (
                AGENT_TEMPERATURE if temperature is None else temperature
            ),
        },
    }
    if tools:
        payload["tools"] = tools

    r = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=timeout or OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def is_available():
    """Quick health-check on the local Ollama instance."""
    try:
        url = OLLAMA_CHAT_URL.replace("/api/chat", "/api/tags")
        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models():
    """Return the list of locally pulled Ollama models (tags)."""
    try:
        url = OLLAMA_CHAT_URL.replace("/api/chat", "/api/tags")
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
