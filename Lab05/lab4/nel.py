"""Named Entity Linking against Wikidata + local cache."""
import json
import os
import requests

from config import (
    NEL_CACHE_FILE,
    CACHE_DIR,
    WIKIDATA_API,
    NEL_TOP_K,
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
)

_cache = None
_HEADERS = {"User-Agent": HTTP_USER_AGENT}


def _load_cache():
    global _cache
    if _cache is not None:
        return _cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(NEL_CACHE_FILE):
        try:
            with open(NEL_CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache():
    cache = _load_cache()
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(NEL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _cache_key(name, lang):
    return f"{lang}::{name.lower()}"


def search_wikidata(name, lang="en", limit=NEL_TOP_K):
    """Search Wikidata, return list of candidate dicts."""
    cache = _load_cache()
    key = _cache_key(name, lang)
    if key in cache:
        return cache[key]

    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": lang,
        "uselang": lang,
        "format": "json",
        "limit": limit,
        "type": "item",
    }
    try:
        r = requests.get(
            WIKIDATA_API, params=params, headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # Do not cache transient errors so the next call can retry.
        return [{"error": str(e)}]

    candidates = []
    for hit in data.get("search", []):
        candidates.append(
            {
                "qid": hit.get("id"),
                "label": hit.get("label", ""),
                "description": hit.get("description", ""),
                "url": hit.get("concepturi") or hit.get("url", ""),
            }
        )
    # Only cache successful, non-empty results to avoid poisoning the
    # cache with empty hits caused by network blips or rate limits.
    if candidates:
        cache[key] = candidates
        _save_cache()
    return candidates


def link_entity(name, lang="en"):
    """Return top candidate or None."""
    candidates = search_wikidata(name, lang=lang, limit=1)
    if not candidates or "error" in candidates[0]:
        return None
    return candidates[0]


def link_entities(entities, lang="en"):
    """Link a list of entities (each dict with 'text'). Adds 'link' key."""
    out = []
    for e in entities:
        link = link_entity(e["text"], lang=lang)
        e2 = dict(e)
        e2["link"] = link
        out.append(e2)
    return out
