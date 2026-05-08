"""Machine translation via Helsinki-NLP/Opus-MT (MarianMT)."""
from config import (
    TRANSLATION_MODEL_TEMPLATE,
    TRANSLATION_MODEL_OVERRIDES,
    SUPPORTED_TRANSLATION_PAIRS,
)

# key: (src, tgt) -> (tokenizer, model)
_pipelines = {}


def _get_pipeline(src, tgt):
    key = (src, tgt)
    if key in _pipelines:
        return _pipelines[key]
    if (src, tgt) not in SUPPORTED_TRANSLATION_PAIRS:
        raise ValueError(
            f"Unsupported translation pair: {src}->{tgt}. "
            f"Supported: {SUPPORTED_TRANSLATION_PAIRS}"
        )
    # MarianMT works directly without relying on the high-level
    # `pipeline('translation', ...)` task, which is no longer registered
    # in some transformers versions.
    from transformers import MarianMTModel, MarianTokenizer

    override = TRANSLATION_MODEL_OVERRIDES.get((src, tgt))
    if override:
        model_name, target_prefix = override
    else:
        model_name = TRANSLATION_MODEL_TEMPLATE.format(src=src, tgt=tgt)
        target_prefix = ""
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load translation model '{model_name}': {e}\n"
            f"Make sure you have internet access on first use, or run "
            f"`pip install -U transformers sentencepiece sacremoses`."
        ) from e
    _pipelines[key] = (tokenizer, model, target_prefix)
    return _pipelines[key]


def translate(text, src, tgt):
    if src == tgt:
        return text
    tokenizer, model, target_prefix = _get_pipeline(src, tgt)
    prepared = f"{target_prefix} {text}".strip() if target_prefix else text
    inputs = tokenizer(
        [prepared], return_tensors="pt", truncation=True, max_length=512
    )
    generated = model.generate(**inputs, max_length=512)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def supported_pairs():
    return list(SUPPORTED_TRANSLATION_PAIRS)
