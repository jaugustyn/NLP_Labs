"""Local knowledge tool — searches data accumulated across labs:
- Lab 4 NEL cache (cache/nel_cache.json)
- Lab 4 saved summaries (lab4results/summaries/*.txt)
- Lab 1 sentences dataset (Lab01/sentences.json), if present
"""
import glob
import json
import os
import re

from config import (
    BASE_DIR,
    NEL_CACHE_FILE,
    SUMMARIES_DIR,
    LOCAL_KB_MAX_HITS,
)


_LAB1_SENTENCES = os.path.normpath(
    os.path.join(BASE_DIR, "..", "Lab01", "sentences.json")
)


def _tokenize(s):
    return [t.lower() for t in re.findall(r"\w+", s or "")]


def _score(text, query_tokens):
    if not query_tokens:
        return 0
    text_tokens = set(_tokenize(text))
    return sum(1 for t in query_tokens if t in text_tokens)


def _search_nel_cache(query_tokens):
    if not os.path.exists(NEL_CACHE_FILE):
        return []
    try:
        with open(NEL_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f) or {}
    except Exception:
        return []
    hits = []
    for key, entries in cache.items():
        if not isinstance(entries, list):
            continue
        for ent in entries:
            blob = " ".join([
                str(ent.get("label", "")),
                str(ent.get("description", "")),
                str(ent.get("qid", "")),
                key,
            ])
            s = _score(blob, query_tokens)
            if s > 0:
                hits.append({
                    "source": "nel_cache",
                    "score": s,
                    "key": key,
                    "qid": ent.get("qid"),
                    "label": ent.get("label"),
                    "description": ent.get("description"),
                })
    return hits


def _search_summaries(query_tokens):
    if not os.path.isdir(SUMMARIES_DIR):
        return []
    hits = []
    for path in glob.glob(os.path.join(SUMMARIES_DIR, "*.txt")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        s = _score(content, query_tokens)
        if s > 0:
            snippet = content[:400].replace("\n", " ")
            hits.append({
                "source": "summary",
                "score": s,
                "path": os.path.relpath(path, BASE_DIR),
                "snippet": snippet,
            })
    return hits


def _search_lab1_sentences(query_tokens):
    if not os.path.exists(_LAB1_SENTENCES):
        return []
    try:
        with open(_LAB1_SENTENCES, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    hits = []
    sentences = data if isinstance(data, list) else data.get("sentences", [])
    for item in sentences:
        text = item.get("text") if isinstance(item, dict) else str(item)
        if not text:
            continue
        s = _score(text, query_tokens)
        if s > 0:
            hits.append({
                "source": "lab1_sentences",
                "score": s,
                "text": text,
                "label": item.get("label") if isinstance(item, dict) else None,
            })
    return hits


def local_knowledge(query, max_hits=LOCAL_KB_MAX_HITS):
    """Search across local knowledge sources (NEL cache, summaries,
    Lab 1 sentences)."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "Empty query."}
    qt = _tokenize(query)
    if not qt:
        return {"error": "No usable tokens in query."}
    hits = (
        _search_nel_cache(qt)
        + _search_summaries(qt)
        + _search_lab1_sentences(qt)
    )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return {
        "query": query,
        "total_hits": len(hits),
        "hits": hits[:max_hits],
    }


SCHEMA = {
    "type": "function",
    "function": {
        "name": "local_knowledge",
        "description": (
            "Search local knowledge accumulated by the bot across "
            "previous labs: cached Wikidata entity links (Lab 4 NEL "
            "cache), saved summaries, and the Lab 1 sentences dataset. "
            "Use BEFORE calling web_search when the user might be "
            "asking about something already seen by the bot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, e.g. 'Steve Jobs'.",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}
