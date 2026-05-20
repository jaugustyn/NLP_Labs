"""Expose Lab 1-4 NLP capabilities as a Lab 5 agent tool."""

import re
import unicodedata

from lab4 import language_detect, ner, summarization, translation


_OPS = ("translate", "summarize", "extract_entities", "classify_sentiment")

_NEGATORS = {"nie", "not", "no", "never", "nigdy"}
_INTENSIFIERS = {
    "bardzo", "mega", "super", "strasznie", "okropnie", "wyjatkowo",
    "naprawde", "really", "very", "extremely", "absolutely",
    "niesamowicie",
}
_POSITIVE_TERMS = {
    "polecam", "polecic", "dobry", "dobra", "dobre", "swietny",
    "swietna", "swietne", "super", "doskonaly", "doskonala",
    "wspanialy", "wspaniala", "fantastyczny", "fantastyczna",
    "genialny", "genialna", "rewelacyjny", "rewelacyjna", "idealny",
    "idealna", "znakomity", "znakomita", "kocham", "uwielbiam",
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "best", "awesome", "perfect", "recommend", "brilliant",
}
_NEGATIVE_TERMS = {
    "niepolecam", "odradzam", "zly", "zla", "zle", "slaby", "slaba",
    "slabe", "fatalny", "fatalna", "fatalne", "okropny", "okropna",
    "okropne", "straszny", "straszna", "straszne", "najgorszy",
    "najgorsza", "beznadziejny", "beznadziejna", "beznadziejne",
    "tragiczny", "tragiczna", "tragiczne", "kiepski", "kiepska",
    "kiepskie", "marny", "marna", "marne", "zalosny", "zalosna",
    "glupi", "glupia", "glupie", "nudny", "nudna", "nudne",
    "rozczarowujacy", "rozczarowujaca", "rozczarowanie",
    "bad", "terrible", "awful", "horrible", "worst", "hate", "poor",
    "disappointing", "useless", "waste", "broken", "boring",
}
_STRONG_NEGATIVE_PHRASES = (
    r"\bnie\s+(?:bardzo\s+)?polecam\b",
    r"\bnie\s+warto\b",
    r"\bnie\s+podoba(?:l|la|lo)?\b",
    r"\bstrata\s+czasu\b",
    r"\bodradzam\b",
)
_STRONG_POSITIVE_PHRASES = (
    r"\bgoraco\s+polecam\b",
    r"\bbardzo\s+polecam\b",
    r"\bwarto\s+obejrzec\b",
)


def _resolve_lang(text, lang):
    if lang and lang != "auto":
        return lang
    try:
        d = language_detect.detect_language(text)
        return d if d and d != "unknown" else "en"
    except Exception:
        return "en"


def _fold(text):
    text = (text or "").translate(
        str.maketrans({
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
            "Ą": "A",
            "Ć": "C",
            "Ę": "E",
            "Ł": "L",
            "Ń": "N",
            "Ó": "O",
            "Ś": "S",
            "Ź": "Z",
            "Ż": "Z",
        })
    )
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _sentiment_tokens(text):
    return re.findall(r"[a-z0-9']+", _fold(text))


def _term_score(token):
    if token in _POSITIVE_TERMS:
        return 1.0
    if token in _NEGATIVE_TERMS:
        return -1.0
    # Polish inflection fallback: keep this stem-based and generic.
    if any(
        token.startswith(stem)
        for stem in (
            "swietn",
            "doskonal",
            "wspanial",
            "fantastycz",
            "genialn",
            "rewelacyj",
            "znakomit",
            "idealn",
        )
    ):
        return 1.0
    if any(
        token.startswith(stem)
        for stem in (
            "fataln",
            "okropn",
            "straszn",
            "beznadziej",
            "tragiczn",
            "kiepsk",
            "glup",
            "nudn",
            "rozczarow",
        )
    ):
        return -1.0
    return 0.0


def _classify_sentiment_rule(text):
    """Small deterministic PL/EN rule set with negation handling.

    Lab03's original rule list is intentionally simple and, in this
    project copy, contains mojibake in Polish words. The Lab05 tool needs
    a more robust direct-test path, especially for phrases like
    "nie polecam".
    """
    folded = _fold(text)
    score = 0.0
    for pattern in _STRONG_NEGATIVE_PHRASES:
        if re.search(pattern, folded):
            score -= 2.0
    for pattern in _STRONG_POSITIVE_PHRASES:
        if re.search(pattern, folded):
            score += 1.5

    tokens = _sentiment_tokens(text)
    for i, token in enumerate(tokens):
        base = _term_score(token)
        if not base:
            continue
        window = tokens[max(0, i - 3):i]
        if any(t in _NEGATORS for t in window):
            base *= -1.2
        if window and window[-1] in _INTENSIFIERS:
            base *= 1.4
        score += base

    if abs(score) < 0.4:
        return "neutralny", 0.5
    confidence = round(min(0.95, 0.35 + abs(score) / 4.0), 4)
    return ("pozytywny" if score > 0 else "negatywny"), confidence


def nlp_tools(
    operation,
    text,
    language="auto",
    target_language=None,
    summary_type="abstractive",
    length="short",
):
    """Run one Lab 1-4 NLP operation."""
    if operation not in _OPS:
        return {
            "error": (
                f"Unknown operation: {operation}. "
                f"Allowed: {list(_OPS)}"
            )
        }
    if not isinstance(text, str) or not text.strip():
        return {"error": "text is required."}

    if operation == "translate":
        src = _resolve_lang(text, language)
        tgt = (target_language or "en").lower()
        if src == tgt:
            return {
                "operation": "translate",
                "src": src,
                "tgt": tgt,
                "translation": text,
                "note": "src == tgt, no-op",
            }
        if (
            hasattr(translation, "validate_pair")
            and not translation.validate_pair(src, tgt)
        ):
            return {
                "operation": "translate",
                "src": src,
                "tgt": tgt,
                "error": f"Unsupported translation pair: {src}->{tgt}",
            }
        try:
            out = translation.translate(text, src, tgt)
            return {
                "operation": "translate",
                "src": src,
                "tgt": tgt,
                "translation": out,
            }
        except Exception as e:
            return {"operation": "translate", "error": str(e)}

    if operation == "summarize":
        try:
            out = summarization.summarize(
                text,
                kind=summary_type,
                length=length,
            )
            return {
                "operation": "summarize",
                "type": summary_type,
                "length": length,
                "summary": out,
            }
        except Exception as e:
            return {"operation": "summarize", "error": str(e)}

    if operation == "extract_entities":
        lang = _resolve_lang(text, language)
        try:
            ents = ner.extract_entities(text, method="spacy", lang=lang)
            return {
                "operation": "extract_entities",
                "language": lang,
                "count": len(ents),
                "entities": [
                    {"text": e.get("text"), "label": e.get("label")}
                    for e in ents
                ],
            }
        except Exception as e:
            return {"operation": "extract_entities", "error": str(e)}

    if operation == "classify_sentiment":
        try:
            label, conf = _classify_sentiment_rule(text)
            return {
                "operation": "classify_sentiment",
                "method": "rule_v2",
                "label": label,
                "confidence": float(conf),
            }
        except Exception as e:
            return {"operation": "classify_sentiment", "error": str(e)}

    return {"error": "unreachable"}


SCHEMA = {
    "type": "function",
    "function": {
        "name": "nlp_tools",
        "description": (
            "Run a built-in NLP pipeline from Lab 1-4 on a piece of text. "
            "Operations: 'translate' (machine translation), 'summarize' "
            "(short summary via local LLM), 'extract_entities' (NER), "
            "'classify_sentiment' (rule-based sentiment). Use this to "
            "chain steps, e.g. translate the output of vision/web_search "
            "or summarise long search results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(_OPS),
                    "description": "Which NLP operation to run.",
                },
                "text": {
                    "type": "string",
                    "description": "Input text.",
                },
                "language": {
                    "type": "string",
                    "description": (
                        "ISO 639-1 source language or 'auto'. Defaults "
                        "to 'auto'."
                    ),
                },
                "target_language": {
                    "type": "string",
                    "description": (
                        "Target ISO 639-1 code for 'translate'. "
                        "E.g. 'en', 'pl', 'de', 'fr', 'es'."
                    ),
                },
                "summary_type": {
                    "type": "string",
                    "enum": ["abstractive", "extractive", "bullets"],
                    "description": "Used by 'summarize'. Defaults to 'abstractive'.",
                },
                "length": {
                    "type": "string",
                    "enum": ["short", "medium", "long"],
                    "description": "Used by 'summarize'. Defaults to 'short'.",
                },
            },
            "required": ["operation", "text"],
        },
    },
}
