"""Web search tool: Wikipedia REST summary + DuckDuckGo Instant Answer."""
import re
import urllib.parse

import requests

from config import (
    WIKIDATA_API,
    WIKIPEDIA_API_TEMPLATE,
    WIKIPEDIA_REST_TEMPLATE,
    DUCKDUCKGO_API_URL,
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
    WEB_SEARCH_MAX_CHARS,
)


_HEADERS = {"User-Agent": HTTP_USER_AGENT}

def _fold(text):
    return (
        (text or "").lower()
        .replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ó", "o")
        .replace("ś", "s")
        .replace("ź", "z")
        .replace("ż", "z")
    )


def _normalise_query(query):
    cleaned = (query or "").strip(" \"'")
    folded = _fold(cleaned)
    if any(k in folded for k in ("ceo", "prezes")):
        match = re.search(
            r"(?:ceo|prezes(?:em)?(?:\s+firmy)?)\s+(?:firmy\s+)?(.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"(?:kto\s+jest|who\s+is).*?(?:ceo|prezes).*?(?:firmy\s+)?(.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
        entity = match.group(1) if match else cleaned
        entity = re.sub(r"[?.!,;:].*$", "", entity).strip(" \"'")
        entity = _normalise_entity_phrase(entity)
        if entity:
            return f"CEO {entity}"
    return cleaned


def _normalise_entity_phrase(text):
    words = []
    for token in re.split(r"\s+", text or ""):
        token = token.strip(" \"'.,!?;:()[]{}")
        if not token:
            continue
        lower = token.lower()
        if lower.endswith("'a"):
            token = token[:-2]
        elif lower.endswith("u") and len(token) > 4:
            token = token[:-1]
        elif lower.endswith("i") and len(token) > 4:
            token = token[:-1] + "a"
        words.append(token)
    return " ".join(words)


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


def _wikipedia_search(query, lang="en"):
    url = WIKIPEDIA_API_TEMPLATE.format(lang=lang)
    try:
        r = requests.get(
            url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        hits = ((r.json() or {}).get("query") or {}).get("search") or []
    except Exception:
        return None
    if not hits:
        return None
    title = hits[0].get("title")
    if not title:
        return None
    return _wikipedia_summary(title, lang=lang)


def _wikidata_qid_for_title(title, lang="en"):
    url = WIKIPEDIA_API_TEMPLATE.format(lang=lang)
    try:
        r = requests.get(
            url,
            params={
                "action": "query",
                "prop": "pageprops",
                "titles": title,
                "format": "json",
            },
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        pages = ((r.json() or {}).get("query") or {}).get("pages") or {}
    except Exception:
        return None
    for page in pages.values():
        qid = (page.get("pageprops") or {}).get("wikibase_item")
        if qid:
            return qid
    return None


def _wikidata_claim_labels(qid, prop, lang="en"):
    try:
        r = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": qid,
                "props": "claims",
                "format": "json",
            },
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        entity = ((r.json() or {}).get("entities") or {}).get(qid) or {}
    except Exception:
        return []

    claims = ((entity.get("claims") or {}).get(prop) or [])
    preferred = [c for c in claims if c.get("rank") == "preferred"]
    active = [
        c for c in claims
        if "P582" not in (c.get("qualifiers") or {})  # end time
    ]
    selected_claims = preferred or active or claims

    ids = []
    for claim in selected_claims:
        value = (((claim.get("mainsnak") or {}).get("datavalue") or {})
                 .get("value") or {})
        target_id = value.get("id") if isinstance(value, dict) else None
        if target_id and target_id not in ids:
            ids.append(target_id)
    if not ids:
        return []

    try:
        r = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(ids),
                "props": "labels",
                "languages": f"{lang}|en",
                "format": "json",
            },
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        entities = (r.json() or {}).get("entities") or {}
    except Exception:
        return []

    labels = []
    for item_id in ids:
        labels_map = (entities.get(item_id) or {}).get("labels") or {}
        label = (labels_map.get(lang) or labels_map.get("en") or {}).get("value")
        if label:
            labels.append(label)
    return labels


def _enrich_with_structured_facts(query, result, lang):
    if not result or result.get("source") != "wikipedia":
        return result
    lowered = query.lower()
    facts = {}
    if any(k in lowered for k in ("ceo", "chief executive", "prezes")):
        qid = _wikidata_qid_for_title(result.get("title"), result.get("lang") or lang)
        if qid:
            ceos = _wikidata_claim_labels(qid, "P169", lang=lang)
            if ceos:
                facts["chief_executive_officer"] = ceos
        if "chief_executive_officer" not in facts:
            summary = (result.get("summary") or "").lower()
            if any(phrase in summary for phrase in (
                "ceo of",
                "chief executive officer of",
                "dyrektorem generalnym firmy",
                "dyrektor generalny firmy",
                "prezesem zarządu",
                "prezesem zarzadu",
            )):
                title = result.get("title")
                if title:
                    facts["chief_executive_officer"] = [title]
    if facts:
        result["facts"] = facts
    return result


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
    q = _normalise_query(query.strip())
    lang = (language or "en").split("-")[0].lower() or "en"

    # 1) Wikipedia in requested language
    res = _wikipedia_summary(q, lang=lang)
    # 1b) Wikipedia search API (for natural queries like "CEO Tesla")
    if not res:
        res = _wikipedia_search(q, lang=lang)
    # 2) Wikipedia in English (fallback)
    if not res and lang != "en":
        res = _wikipedia_summary(q, lang="en")
    if not res and lang != "en":
        res = _wikipedia_search(q, lang="en")
    # 3) DuckDuckGo instant answer
    if not res:
        res = _duckduckgo(q)
    if not res:
        return {"query": q, "error": "No results."}
    res["query"] = q
    return _enrich_with_structured_facts(q, res, lang)


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
                    "description": (
                        "ISO 639-1 code for Wikipedia language "
                        "(en, pl, ...). Defaults to 'en'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}
