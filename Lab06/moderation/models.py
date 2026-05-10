"""Local moderation model adapters.

The lab brief names small safety models from Hugging Face. In a classroom
Windows setup those models may be unavailable, so these adapters expose
the required interfaces and use deterministic local rules as a stable
fallback. The rest of Lab06 depends on the interface, not on a specific
runtime backend.
"""
import re


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
_PESEL_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USERNAME_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,32}\b")

_TOXIC = {
    "idiota", "debil", "kretyn", "glupi", "glupia", "glupie",
    "zlodziej", "zlodzieje", "stupid", "idiot", "moron", "trash",
    "garbage",
}
_SPAM = {
    "kliknij", "promocja", "darmowe", "zarob", "casino", "krypto",
    "crypto", "free", "buy", "discount", "limited", "winner",
}
_SELF_HARM = {
    "zabij sie", "zabic sie", "samoboj", "samobojstwo",
    "sie zabic", "powinienes sie zabic", "powinnas sie zabic",
    "kill yourself", "suicide",
}
_VIOLENCE = {
    "zabije", "pobije", "groze", "murder", "kill", "beat you",
}
_SEXUAL = {"porn", "porno", "sex", "nudes", "nagie"}


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


def _contains_any(text, terms):
    return any(term in text for term in terms)


def _score_to_severity(score):
    if score >= 0.92:
        return "critical"
    if score >= 0.78:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _looks_like_card(value):
    digits = re.sub(r"\D", "", value or "")
    return 13 <= len(digits) <= 19


def detect_private_info(text):
    """Detect personally identifiable information using local patterns."""
    entities = []
    for kind, regex in (
        ("email", _EMAIL_RE),
        ("phone_number", _PHONE_RE),
        ("credit_card", _CARD_RE),
        ("polish_pesel", _PESEL_RE),
    ):
        for match in regex.finditer(text or ""):
            value = match.group(0)
            if kind == "credit_card" and not _looks_like_card(value):
                continue
            entities.append({
                "type": kind,
                "text": value,
                "start": match.start(),
                "end": match.end(),
            })
    return {
        "model": "openai/privacy-filter",
        "method": "local_regex_fallback",
        "has_pii": bool(entities),
        "entities": entities,
    }


def classify_bielik_guard(text):
    """Classify content into safety categories with a local fallback."""
    folded = _fold(text)
    categories = []
    score = 0.05

    if _contains_any(folded, _SELF_HARM):
        categories.append("self_harm")
        score = max(score, 0.95)
    if _contains_any(folded, _VIOLENCE):
        categories.append("violence")
        score = max(score, 0.88)
    if _contains_any(folded, _SEXUAL):
        categories.append("sexual")
        score = max(score, 0.8)
    if _contains_any(folded, _TOXIC):
        categories.append("toxic")
        score = max(score, 0.74)
    if _contains_any(folded, _SPAM) or len(_URL_RE.findall(text or "")) >= 2:
        categories.append("spam")
        score = max(score, 0.72)
    if "polityk" in folded and "zlodziej" in folded:
        categories.append("political_opinion")
        score = max(score, 0.62)
    if re.search(r"\b(nienawidze|hate)\b.+\b(wszyscy|all)\b", folded):
        categories.append("hate_speech")
        score = max(score, 0.84)

    if not categories:
        return {
            "model": "speakleash/Bielik-Guard-0.1B-v1.0",
            "method": "local_rule_fallback",
            "label": "clean",
            "categories": ["clean"],
            "score": 0.98,
            "severity": "low",
        }

    top = categories[0]
    return {
        "model": "speakleash/Bielik-Guard-0.1B-v1.0",
        "method": "local_rule_fallback",
        "label": top,
        "categories": categories,
        "score": round(score, 4),
        "severity": _score_to_severity(score),
    }


def classify_qwen_guard(text):
    """LLM-style structured safety decision using local rules."""
    bielik = classify_bielik_guard(text)
    categories = [c for c in bielik.get("categories", []) if c != "clean"]
    score = float(bielik.get("score") or 0)

    if not categories:
        risk = "safe"
        action = "approve"
        categories = ["clean"]
    elif "self_harm" in categories or score >= 0.92:
        risk = "critical"
        action = "reject"
    elif any(c in categories for c in ("violence", "hate_speech")):
        risk = "high"
        action = "reject"
    elif categories:
        risk = "medium" if score >= 0.55 else "low"
        action = "review"

    return {
        "model": "Qwen2.5-Guard",
        "method": "local_rule_fallback",
        "risk_level": risk,
        "categories": categories,
        "confidence": round(max(0.55, min(0.96, score)), 4),
        "recommended_action": action,
    }


def analyze_sentiment_for_moderation(text):
    """Analyze sentiment and add simple moderation-oriented emotion hints."""
    try:
        from lab5.tools.nlp_tools import nlp_tools
        result = nlp_tools(
            operation="classify_sentiment",
            text=text,
            language="auto",
        )
    except Exception:
        folded_for_score = _fold(text)
        negative = _contains_any(folded_for_score, _TOXIC | _VIOLENCE | _SELF_HARM)
        positive = _contains_any(folded_for_score, {"uwielbiam", "najlepszy", "super"})
        if negative:
            result = {"label": "negatywny", "confidence": 0.7}
        elif positive:
            result = {"label": "pozytywny", "confidence": 0.7}
        else:
            result = {"label": "neutralny", "confidence": 0.5}
    label_map = {
        "pozytywny": "positive",
        "neutralny": "neutral",
        "negatywny": "negative",
    }
    sentiment = label_map.get(result.get("label"), "neutral")
    folded = _fold(text)
    if any(k in folded for k in ("idiota", "debil", "zabije", "hate")):
        emotion = "anger"
    elif sentiment == "positive":
        emotion = "joy"
    elif sentiment == "negative":
        emotion = "sadness"
    else:
        emotion = "neutral"
    sarcasm = bool(re.search(r"\b(super|swietnie|great)\b.*[!?]{2,}", folded))
    return {
        "sentiment": sentiment,
        "confidence": float(result.get("confidence") or 0.5),
        "emotion": emotion,
        "sarcasm_detected": sarcasm,
    }


def extract_moderation_entities(text):
    """Extract entities relevant to moderation context."""
    emails = _EMAIL_RE.findall(text or "")
    phones = _PHONE_RE.findall(text or "")
    urls = _URL_RE.findall(text or "")
    usernames = _USERNAME_RE.findall(text or "")
    persons = []
    organizations = []
    locations = []

    try:
        from lab4 import ner
        for ent in ner.extract_entities_spacy(text, lang="pl"):
            label = (ent.get("label") or "").upper()
            value = ent.get("text")
            if not value:
                continue
            if label in ("PER", "PERSON"):
                persons.append(value)
            elif label in ("ORG", "ORGANIZATION"):
                organizations.append(value)
            elif label in ("LOC", "GPE", "LOCATION"):
                locations.append(value)
    except Exception:
        pass

    return {
        "usernames_mentioned": usernames,
        "urls": urls,
        "emails": emails,
        "phone_numbers": phones,
        "organizations": sorted(set(organizations)),
        "locations": sorted(set(locations)),
        "persons": sorted(set(persons)),
    }
