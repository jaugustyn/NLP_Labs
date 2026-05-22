"""Expose Lab 1-4 NLP capabilities as a Lab 5 agent tool."""

import re

from lab3.sentiment_methods import predict_rule
from lab4 import language_detect, ner, summarization, translation


_OPS = ("translate", "summarize", "extract_entities", "classify_sentiment")


def _resolve_lang(text, lang):
    if lang and lang != "auto":
        return lang
    try:
        d = language_detect.detect_language(text)
        return d if d and d != "unknown" else "en"
    except Exception:
        return "en"


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
        if not target_language:
            return {
                "operation": "translate",
                "src": src,
                "error": "target_language is required for translation.",
            }
        tgt = target_language.lower()
        if not re.fullmatch(r"[a-z]{2}", tgt):
            return {
                "operation": "translate",
                "src": src,
                "tgt": tgt,
                "error": "target_language must be a two-letter ISO 639-1 code.",
            }
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
            label, conf = predict_rule(text)
            return {
                "operation": "classify_sentiment",
                "method": "lab3_rule",
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
                        "Required target ISO 639-1 code for 'translate'. "
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
