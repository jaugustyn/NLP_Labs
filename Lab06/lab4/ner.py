"""Named Entity Recognition via spaCy, Stanza or a small regex fallback."""

import re

from config import SPACY_MODELS


_spacy_models = {}
_stanza_pipelines = {}

_CAPITALIZED_SEQUENCE_RE = re.compile(
    r"\b(?:[A-ZŁŚŻŹĆŃÓĘ][\wŁŚŻŹĆŃÓĘłśżźćńóę-]+"
    r"(?:\s+[A-ZŁŚŻŹĆŃÓĘ][\wŁŚŻŹĆŃÓĘłśżźćńóę-]+){0,3})\b"
)
_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}\b")
_MONEY_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:USD|EUR|PLN|zł|euro|dollars?)\b", re.I)
_ORG_SUFFIXES = {
    "AG",
    "AI",
    "Corp",
    "Corporation",
    "Foundation",
    "GmbH",
    "Group",
    "Inc",
    "Institute",
    "LLC",
    "Ltd",
    "S.A.",
    "University",
}
_LOCATION_HINTS = {
    "Austin",
    "Berlin",
    "France",
    "Germany",
    "London",
    "Paris",
    "Poland",
    "San Francisco",
    "United States",
    "Warsaw",
    "Warszawa",
}


def _get_spacy(lang):
    if lang in _spacy_models:
        return _spacy_models[lang]

    import spacy

    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        raise ValueError(f"spaCy model not configured for language '{lang}'")

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc

    _spacy_models[lang] = nlp
    return nlp


def _get_stanza(lang):
    if lang in _stanza_pipelines:
        return _stanza_pipelines[lang]

    import stanza

    try:
        nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,ner",
            verbose=False,
            download_method=None,
        )
    except Exception:
        stanza.download(lang, verbose=False)
        nlp = stanza.Pipeline(lang=lang, processors="tokenize,ner", verbose=False)

    _stanza_pipelines[lang] = nlp
    return nlp


def extract_entities_spacy(text, lang="en"):
    nlp = _get_spacy(lang)
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "method": "spacy",
        }
        for ent in doc.ents
    ]


def extract_entities_stanza(text, lang="en"):
    nlp = _get_stanza(lang)
    doc = nlp(text)
    entities = []
    for sentence in doc.sentences:
        for ent in sentence.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "method": "stanza",
                }
            )
    return entities


def extract_entities_regex(text):
    entities = []
    for match in _DATE_RE.finditer(text):
        entities.append(_entity(match.group(0), "DATE", match.start(), match.end()))
    for match in _MONEY_RE.finditer(text):
        entities.append(_entity(match.group(0), "MONEY", match.start(), match.end()))
    for match in _CAPITALIZED_SEQUENCE_RE.finditer(text):
        value = match.group(0).strip()
        if len(value) < 2 or value.lower() in {"i", "the", "a"}:
            continue
        entities.append(_entity(value, _guess_label(value), match.start(), match.end()))
    return merge_entities(entities)


def _entity(text, label, start, end):
    return {
        "text": text,
        "label": label,
        "start": start,
        "end": end,
        "method": "regex",
    }


def _guess_label(value):
    words = value.split()
    if value in _LOCATION_HINTS:
        return "GPE"
    if any(word.strip(",.") in _ORG_SUFFIXES for word in words):
        return "ORG"
    if len(words) >= 2:
        return "PERSON"
    if value.lower().endswith(("ai", "corp", "inc")):
        return "ORG"
    return "MISC"


def extract_entities(text, method="spacy", lang="en", allow_fallback=True):
    method = method.lower()
    try:
        if method == "spacy":
            return extract_entities_spacy(text, lang)
        if method == "stanza":
            return extract_entities_stanza(text, lang)
    except Exception:
        if not allow_fallback:
            raise
        return extract_entities_regex(text)

    if method == "regex":
        return extract_entities_regex(text)
    raise ValueError(f"Unknown NER method: {method}")


def merge_entities(*entity_lists):
    seen = {}
    for entities in entity_lists:
        for entity in entities:
            key = (
                entity["text"].strip().lower(),
                entity.get("label", "MISC"),
                entity.get("start"),
                entity.get("end"),
            )
            if key not in seen:
                seen[key] = entity
    return sorted(seen.values(), key=lambda item: (item["start"], item["end"]))


def group_by_label(entities):
    grouped = {}
    for entity in entities:
        grouped.setdefault(entity["label"], []).append(entity["text"])
    return grouped
