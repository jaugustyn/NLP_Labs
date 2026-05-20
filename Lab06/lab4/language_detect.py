"""Language detection helpers for Lab 4."""

import re


try:
    from langdetect import DetectorFactory, LangDetectException, detect, detect_langs

    DetectorFactory.seed = 42
except Exception:
    DetectorFactory = None
    LangDetectException = Exception
    detect = None
    detect_langs = None


_LANGUAGE_KEYWORDS = {
    "pl": {
        "ale",
        "bardzo",
        "jest",
        "oraz",
        "polska",
        "polski",
        "się",
        "stolica",
        "to",
        "warszawa",
    },
    "en": {
        "and",
        "city",
        "company",
        "founded",
        "hello",
        "is",
        "of",
        "the",
        "this",
        "world",
    },
    "de": {"berlin", "das", "der", "deutschland", "die", "ein", "ist", "und"},
    "fr": {"bonjour", "de", "est", "et", "france", "la", "le", "les", "paris"},
    "es": {"de", "el", "en", "es", "espana", "hola", "la", "madrid", "y"},
}

_DIACRITIC_HINTS = {
    "pl": "ąćęłńóśźż",
    "de": "äöüß",
    "fr": "àâçéèêëîïôùûüÿœæ",
    "es": "áéíñóúü¿¡",
}

_WORD_RE = re.compile(r"(?u)\b\w+\b")


def detect_language(text):
    lang, _ = detect_language_with_confidence(text)
    return lang


def detect_language_with_confidence(text):
    if not text or not text.strip():
        return "unknown", 0.0

    heuristic_lang, heuristic_confidence = _detect_by_heuristics(text)
    if heuristic_confidence >= 0.85:
        return heuristic_lang, heuristic_confidence
    token_count = len(_WORD_RE.findall(text))
    if token_count <= 4 and heuristic_confidence >= 0.75:
        return heuristic_lang, heuristic_confidence

    if detect_langs is not None:
        try:
            results = detect_langs(text)
            if results:
                top = results[0]
                if top.prob >= heuristic_confidence:
                    return top.lang, float(top.prob)
        except LangDetectException:
            pass

    if heuristic_lang != "unknown":
        return heuristic_lang, heuristic_confidence
    return "unknown", 0.0


def _detect_by_heuristics(text):
    lowered = text.lower()
    for lang, chars in _DIACRITIC_HINTS.items():
        if any(char in lowered for char in chars):
            return lang, 0.95

    tokens = {token.lower() for token in _WORD_RE.findall(lowered)}
    if not tokens:
        return "unknown", 0.0

    scores = {
        lang: len(tokens & keywords)
        for lang, keywords in _LANGUAGE_KEYWORDS.items()
    }
    best_lang, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "unknown", 0.0

    confidence = min(0.8, 0.35 + best_score / max(4, len(tokens)))
    return best_lang, confidence
