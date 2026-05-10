"""Web search tool: Wikipedia REST summary + DuckDuckGo Instant Answer."""
import urllib.parse

import requests

from config import (
    WIKIPEDIA_REST_TEMPLATE,
    DUCKDUCKGO_API_URL,
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
    WEB_SEARCH_MAX_CHARS,
)


_HEADERS = {"User-Agent": HTTP_USER_AGENT}


def _wikipedia_summary(query, lang="en"):
    title = urllib.parse.quote(query.replace(" ", "_"))
    url = WIKIPEDIA_REST_TEMPLATE.format(lang=lang, title=title)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    return {
        "source": "wikipedia",
        "lang": lang,
        "title": data.get("title", query),
        "summary": extract[:WEB_SEARCH_MAX_CHARS],
        "url": (data.get("content_urls", {}).get("desktop") or {}).get("page", ""),
    }


def _duckduckgo(query):
    try:
        r = requests.get(
            DUCKDUCKGO_API_URL,
            params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    abstract = (data.get("AbstractText") or "").strip()
    if not abstract:
        # Try the first related-topic text
        for rt in (data.get("RelatedTopics") or [])[:1]:
            if isinstance(rt, dict) and rt.get("Text"):
                abstract = rt["Text"]
                break
    if not abstract:
        return None
    return {
        "source": "duckduckgo",
        "title": data.get("Heading") or query,
        "summary": abstract[:WEB_SEARCH_MAX_CHARS],
        "url": data.get("AbstractURL") or "",
    }


def web_search(query, language="en"):
    """Search the web (Wikipedia → DuckDuckGo fallback) and return a
    short summary of the top result."""
    if not isinstance(query, str) or not query.strip():
        return {"error": "Empty query."}
    q = query.strip()
    lang = (language or "en").split("-")[0].lower() or "en"

    # 1) Wikipedia in requested language
    res = _wikipedia_summary(q, lang=lang)
    # 2) Wikipedia in English (fallback)
    if not res and lang != "en":
        res = _wikipedia_summary(q, lang="en")
    # 3) DuckDuckGo instant answer
    if not res:
        res = _duckduckgo(q)
    if not res:
        return {"query": q, "error": "No results."}
    res["query"] = q
    return res


SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for factual information (Wikipedia + "
            "DuckDuckGo). Use whenever the user asks about a person, "
            "place, organisation, event, definition, or any fact you "
            "are not sure about. Returns a short summary and a URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up, e.g. 'CEO of Tesla'.",
                },
                "language": {
                    "type": "string",
                    "description": "ISO 639-1 code for Wikipedia language (en, pl, ...). Defaults to 'en'.",
                },
            },
            "required": ["query"],
        },
    },
}
