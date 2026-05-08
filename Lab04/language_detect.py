"""Language detection via langdetect."""
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException

# Make detection deterministic
DetectorFactory.seed = 42


def detect_language(text):
    """Return ISO 639-1 code (e.g. 'en', 'pl') or 'unknown'."""
    if not text or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_language_with_confidence(text):
    """Return (lang, confidence) or ('unknown', 0.0)."""
    if not text or not text.strip():
        return "unknown", 0.0
    try:
        results = detect_langs(text)
        if not results:
            return "unknown", 0.0
        top = results[0]
        return top.lang, float(top.prob)
    except LangDetectException:
        return "unknown", 0.0
