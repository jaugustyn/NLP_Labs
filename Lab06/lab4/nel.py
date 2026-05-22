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

# Small offline demo fallback used when Wikidata is unavailable.
_DEMO_LOCAL_KB = {
    "en": {
        "steve jobs": {
            "qid": "Q19837",
            "label": "Steve Jobs",
            "description": "American entrepreneur and co-founder of Apple",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Steve_Jobs",
        },
        "elon musk": {
            "qid": "Q317521",
            "label": "Elon Musk",
            "description": "Businessperson associated with Tesla, SpaceX and xAI",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Elon_Musk",
        },
        "tesla": {
            "qid": "Q478214",
            "label": "Tesla, Inc.",
            "description": "American electric vehicle and clean energy company",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Tesla,_Inc.",
        },
        "spacex": {
            "qid": "Q193701",
            "label": "SpaceX",
            "description": "American space technology company",
            "wikipedia_url": "https://en.wikipedia.org/wiki/SpaceX",
        },
        "xai": {
            "qid": "Q119973459",
            "label": "xAI",
            "description": "Artificial intelligence company founded by Elon Musk",
            "wikipedia_url": "https://en.wikipedia.org/wiki/XAI_(company)",
        },
        "austin": {
            "qid": "Q16559",
            "label": "Austin",
            "description": "Capital city of Texas, United States",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Austin,_Texas",
        },
    },
    "pl": {
        "warszawa": {
            "qid": "Q270",
            "label": "Warszawa",
            "description": "Stolica Polski",
            "wikipedia_url": "https://pl.wikipedia.org/wiki/Warszawa",
        },
        "polska": {
            "qid": "Q36",
            "label": "Polska",
            "description": "Państwo w Europie Środkowej",
            "wikipedia_url": "https://pl.wikipedia.org/wiki/Polska",
        },
    },
}


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


def search_local_kb(name, lang="en"):
    records = _DEMO_LOCAL_KB.get(lang, {})
    record = records.get((name or "").strip().lower())
    if not record and lang != "en":
        record = _DEMO_LOCAL_KB.get("en", {}).get((name or "").strip().lower())
    if not record:
        return []

    candidate = dict(record)
    candidate["source"] = "local"
    return [candidate]


def local_wikipedia_url(qid):
    for records in _DEMO_LOCAL_KB.values():
        for record in records.values():
            if record.get("qid") == qid:
                return record.get("wikipedia_url", "")
    return ""


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


def search_candidates(name, lang="en", limit=NEL_TOP_K):
    merged = []
    seen = set()
    for candidate in search_local_kb(name, lang=lang):
        qid = candidate.get("qid")
        if qid and qid not in seen:
            merged.append(candidate)
            seen.add(qid)

    wikidata_candidates = search_wikidata(name, lang=lang, limit=limit)
    if wikidata_candidates and "error" in wikidata_candidates[0]:
        return merged or wikidata_candidates

    for candidate in wikidata_candidates:
        qid = candidate.get("qid")
        if qid and qid not in seen:
            candidate = dict(candidate)
            candidate.setdefault("source", "wikidata")
            merged.append(candidate)
            seen.add(qid)
    return merged[:limit]


def link_entity(name, lang="en"):
    """Return top candidate or None."""
    candidates = search_candidates(name, lang=lang, limit=1)
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
