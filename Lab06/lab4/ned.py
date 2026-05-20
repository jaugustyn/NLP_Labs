"""Named Entity Disambiguation via Wikidata candidate ranking."""
import re
import requests

from config import (
    WIKIDATA_API,
    NEL_TOP_K,
    NEL_MIN_CONFIDENCE,
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
)
from lab4.nel import local_wikipedia_url, search_candidates

_HEADERS = {"User-Agent": HTTP_USER_AGENT}


_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Minimal multilingual stopword set used to suppress noise in token
# overlap scoring (very short / function words match almost everything).
_STOPWORDS = {
    # English
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "from",
    "with", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "as", "it", "its", "this", "that", "these", "those", "he",
    "she", "they", "we", "i", "you", "his", "her", "their", "our", "my",
    "your", "has", "have", "had", "do", "does", "did", "not", "no",
    # Polish
    "i", "oraz", "lub", "albo", "ale", "to", "jest", "byl", "była",
    "były", "się", "w", "we", "z", "ze", "na", "do", "od", "po", "za",
    "u", "o", "przy", "przez",
}


def _tokens(text):
    return {
        t.lower() for t in _WORD_RE.findall(text or "")
        if len(t) > 1 and t.lower() not in _STOPWORDS
    }


def _phrases_around(name, context, max_len=4):
    """Return phrases (2..max_len words) from context that contain `name`.

    Used to capture multi-word entity expressions like 'Paris Hilton'
    when disambiguating just 'Paris'.
    """
    if not name or not context:
        return []
    words = _WORD_RE.findall(context)
    name_words = _WORD_RE.findall(name)
    if not name_words:
        return []
    n = len(name_words)
    name_lower = [w.lower() for w in name_words]
    phrases = set()
    for i in range(len(words) - n + 1):
        if [w.lower() for w in words[i:i + n]] != name_lower:
            continue
        # Expand window left and right by 1..(max_len - n) words
        for left in range(0, max_len - n + 1):
            for right in range(0, max_len - n - left + 1):
                if left == 0 and right == 0:
                    continue
                start = max(0, i - left)
                end = min(len(words), i + n + right)
                phrase = " ".join(words[start:end])
                if len(phrase.split()) > n:
                    phrases.add(phrase)
    return list(phrases)


def _score(candidate, context_tokens, context_phrases_lower):
    """Score candidate by:
    1. Token overlap between candidate description/label and context.
    2. Bonus if candidate's label matches a multi-word phrase appearing
       in context (e.g. 'Paris Hilton' for the candidate Q47899).
    """
    desc = candidate.get("description", "") or ""
    label = candidate.get("label", "") or ""
    desc_tokens = _tokens(desc) | _tokens(label)
    base = 0.0
    if desc_tokens and context_tokens:
        overlap = desc_tokens & context_tokens
        base = len(overlap) / max(1, len(desc_tokens))

    # Phrase-level boost: if the candidate label is exactly a phrase in
    # context (case-insensitive), it almost certainly is the referent.
    label_lower = label.lower()
    if label_lower and label_lower in context_phrases_lower:
        return min(1.0, base + 0.6)
    return base


def disambiguate(name, context, lang="en", top_k=NEL_TOP_K):
    """Return ranked list of candidates with 'score' field.

    Candidate pool is built from:
      - direct Wikidata search for `name`
      - Wikidata search for multi-word phrases from context that
        contain `name` (e.g. 'Paris Hilton' when name='Paris').
    Phrase-level matches are heavily boosted in scoring.
    """
    candidates = list(search_candidates(name, lang=lang, limit=top_k) or [])
    if candidates and "error" in candidates[0]:
        return candidates

    # Track original Wikidata search rank as tie-breaker (lower = more
    # relevant per Wikidata's own scoring).
    rank_by_qid = {
        c.get("qid"): i for i, c in enumerate(candidates) if c.get("qid")
    }

    phrases = _phrases_around(name, context, max_len=4)
    phrase_set_lower = {p.lower() for p in phrases}
    seen_qids = {c.get("qid") for c in candidates if c.get("qid")}
    for phrase in phrases:
        extra = search_candidates(phrase, lang=lang, limit=3) or []
        if extra and "error" in extra[0]:
            continue
        for c in extra:
            qid = c.get("qid")
            if qid and qid not in seen_qids:
                candidates.append(c)
                seen_qids.add(qid)
                rank_by_qid[qid] = len(rank_by_qid) + 100  # after primary

    ctx_tokens = _tokens(context) - _tokens(name)
    scored = []
    for c in candidates:
        c2 = dict(c)
        base = _score(c, ctx_tokens, phrase_set_lower)
        # Wikidata-rank baseline confidence so the most popular candidate
        # never gets a flat 0.0 when the context is generic / short.
        rank = rank_by_qid.get(c.get("qid"), 999)
        if rank < 100:  # primary search hits
            prior = max(0.0, 0.20 - 0.03 * rank)  # 0.20, 0.17, 0.14, ...
        else:           # phrase-expansion hits
            prior = 0.05
        c2["score"] = round(min(1.0, base + prior), 3)
        scored.append(c2)
    # Sort by score desc, then by Wikidata's relevance rank asc.
    scored.sort(
        key=lambda x: (-x["score"], rank_by_qid.get(x.get("qid"), 999))
    )
    return scored[:top_k]


def best_match(name, context, lang="en"):
    """Return best candidate or None if confidence too low."""
    ranked = disambiguate(name, context, lang=lang)
    if not ranked or "error" in ranked[0]:
        return None
    top = ranked[0]
    if top.get("score", 0.0) < NEL_MIN_CONFIDENCE:
        # Still return top, but mark as low confidence
        top = dict(top)
        top["low_confidence"] = True
    return top


def get_entity_details(qid, lang="en"):
    """Fetch label/description/aliases/claims for a Wikidata QID."""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "languages": lang,
        "format": "json",
        "props": "labels|descriptions|aliases|claims|sitelinks/urls",
        "sitefilter": f"{lang}wiki|enwiki",
    }
    try:
        r = requests.get(
            WIKIDATA_API, params=params, headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": str(e)}

    ent = data.get("entities", {}).get(qid)
    if not ent:
        return {"error": "not found"}

    labels = ent.get("labels", {})
    descs = ent.get("descriptions", {})
    aliases = ent.get("aliases", {}).get(lang, [])
    sitelinks = ent.get("sitelinks", {}) or {}

    wiki_key = f"{lang}wiki" if f"{lang}wiki" in sitelinks else "enwiki"
    wikipedia_url = sitelinks.get(wiki_key, {}).get("url", "")

    return {
        "qid": qid,
        "label": labels.get(lang, {}).get("value")
        or labels.get("en", {}).get("value", ""),
        "description": descs.get(lang, {}).get("value")
        or descs.get("en", {}).get("value", ""),
        "aliases": [a.get("value") for a in aliases],
        "claim_count": len(ent.get("claims", {})),
        "wikipedia_url": wikipedia_url,
    }


_wiki_cache = {}


def get_wikipedia_url(qid, lang="en"):
    """Lightweight cached lookup of the Wikipedia URL for a QID."""
    if not qid:
        return ""
    key = (qid, lang)
    if key in _wiki_cache:
        return _wiki_cache[key]
    details = get_entity_details(qid, lang=lang)
    url = details.get("wikipedia_url", "") if "error" not in details else ""
    if not url:
        url = local_wikipedia_url(qid)
    _wiki_cache[key] = url
    return url
