"""Named Entity Recognition via spaCy or Stanza."""
from config import SPACY_MODELS

# Lazy caches
_spacy_models = {}
_stanza_pipelines = {}


def _get_spacy(lang):
    if lang in _spacy_models:
        return _spacy_models[lang]
    import spacy
    model_name = SPACY_MODELS.get(lang)
    if not model_name:
        raise ValueError(f"spaCy model not configured for language '{lang}'")
    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from e
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
        # Try downloading
        stanza.download(lang, verbose=False)
        nlp = stanza.Pipeline(
            lang=lang, processors="tokenize,ner", verbose=False
        )
    _stanza_pipelines[lang] = nlp
    return nlp


def extract_entities_spacy(text, lang="en"):
    """Return list of dicts {text, label, start, end}."""
    nlp = _get_spacy(lang)
    doc = nlp(text)
    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]


def extract_entities_stanza(text, lang="en"):
    nlp = _get_stanza(lang)
    doc = nlp(text)
    out = []
    for sent in doc.sentences:
        for ent in sent.ents:
            out.append(
                {
                    "text": ent.text,
                    "label": ent.type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
    return out


def extract_entities(text, method="spacy", lang="en"):
    method = method.lower()
    if method == "spacy":
        return extract_entities_spacy(text, lang)
    if method == "stanza":
        return extract_entities_stanza(text, lang)
    raise ValueError(f"Unknown NER method: {method}")


def merge_entities(*entity_lists):
    """Merge entity lists from multiple sources, deduplicating by (text, label)."""
    seen = {}
    for ents in entity_lists:
        for e in ents:
            key = (e["text"].lower(), e["label"])
            if key not in seen:
                seen[key] = e
    return list(seen.values())


def group_by_label(entities):
    grouped = {}
    for e in entities:
        grouped.setdefault(e["label"], []).append(e["text"])
    return grouped
