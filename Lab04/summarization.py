"""Text summarization via Ollama (local LLM)."""
import os
import time

import requests

from config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    SUMMARY_LENGTH_TOKENS,
    SUMMARIES_DIR,
)


_PROMPTS = {
    "extractive": (
        "Summarize the following text by keeping its key information. "
        "Use only fragments present in the original text. "
        "Reply in the same language as the input.\n\n"
        "Text:\n{text}\n\nSummary:"
    ),
    "abstractive": (
        "Write a short, abstractive summary of the following text in your "
        "own words. Reply in the same language as the input.\n\n"
        "Text:\n{text}\n\nSummary:"
    ),
    "bullets": (
        "Summarize the following text as a bulleted list of 3-7 points. "
        "Start every bullet with '- '. Reply in the same language as the "
        "input.\n\nText:\n{text}\n\nBullet list:"
    ),
    "custom": "{custom_prompt}\n\nText:\n{text}\n\nResult:",
}


def is_ollama_available():
    try:
        r = requests.get(
            OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=3
        )
        return r.status_code == 200
    except Exception:
        return False


def generate(prompt, max_tokens=300, model=None):
    """Low-level Ollama call. Returns generated text or raises."""
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def summarize(text, kind="abstractive", length="medium", custom_prompt=None,
              model=None):
    if kind not in _PROMPTS:
        raise ValueError(f"Unknown summary type: {kind}")
    if kind == "custom":
        if not custom_prompt:
            raise ValueError("custom_prompt is required for kind='custom'")
        prompt = _PROMPTS["custom"].format(
            custom_prompt=custom_prompt, text=text
        )
    else:
        prompt = _PROMPTS[kind].format(text=text)
    max_tokens = SUMMARY_LENGTH_TOKENS.get(length, 200)
    return generate(prompt, max_tokens=max_tokens, model=model)


def save_summary(original, summary, kind, length, model_name):
    """Persist a generated summary as a UTF-8 text file. Returns the path."""
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{kind}_{length}_{timestamp}.txt"
    path = os.path.join(SUMMARIES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Type: {kind}\n")
        f.write(f"Length: {length}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n--- Original ---\n")
        f.write(original)
        f.write("\n\n--- Summary ---\n")
        f.write(summary)
    return path
