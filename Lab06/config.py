"""Shared configuration used by Lab06 and inherited lab packages."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PLOTS_DIR = os.path.join(BASE_DIR, "lab4plots")
RESULTS_DIR = os.path.join(BASE_DIR, "lab4results")
SUMMARIES_DIR = os.path.join(RESULTS_DIR, "summaries")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODERATION_DATA_DIR = os.path.join(BASE_DIR, "moderation_data")

NEL_CACHE_FILE = os.path.join(CACHE_DIR, "nel_cache.json")

SUPPORTED_LANGUAGES = ["en", "pl", "de", "fr", "es"]

NER_METHODS = ["spacy", "stanza"]
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "pl": "pl_core_news_sm",
}

TRANSLATION_MODEL_TEMPLATE = "Helsinki-NLP/opus-mt-{src}-{tgt}"
# Some pairs published by Helsinki-NLP became gated on Hugging Face.
# These overrides point to publicly available equivalents. Each value
# is a tuple (model_name, target_token_prefix). The prefix is required
# for multi-target Opus models like opus-mt-en-sla, where the decoder
# selects the output language from a token like '>>pol<<'.
TRANSLATION_MODEL_OVERRIDES = {
    ("en", "pl"): ("Helsinki-NLP/opus-mt-en-sla", ">>pol<<"),
}
SUPPORTED_TRANSLATION_PAIRS = [
    ("en", "pl"),
    ("pl", "en"),
    ("en", "de"),
    ("de", "en"),
    ("en", "fr"),
    ("fr", "en"),
    ("en", "es"),
    ("es", "en"),
    ("pl", "de"),
    ("de", "pl"),
    ("pl", "fr"),
    ("fr", "pl"),
]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_TIMEOUT = 120  # seconds
SUMMARY_TYPES = ["extractive", "abstractive", "bullets", "custom"]
SUMMARY_LENGTHS = ["short", "medium", "long"]
SUMMARY_LENGTH_TOKENS = {
    "short": 80,
    "medium": 200,
    "long": 400,
}

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"
NEL_TOP_K = 5
NEL_MIN_CONFIDENCE = 0.05
HTTP_TIMEOUT = 10  # seconds
# Wikimedia API requires an identifying User-Agent (otherwise 403).
# https://meta.wikimedia.org/wiki/User-Agent_policy
HTTP_USER_AGENT = (
    "NLP-Lab4-Bot/1.0 "
    "(https://github.com/jaugustyn/NLP_Labs; educational use)"
)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

AGENT_MODEL = "qwen2.5:1.5b"
VISION_MODEL = "qwen2.5vl:3b"

MAX_AGENT_ITERATIONS = 5
AGENT_TEMPERATURE = 0.2

SESSIONS_DIR = os.path.join(CACHE_DIR, "sessions")
AGENT_HISTORY_TURNS = 10  # how many recent turns to keep in chat history

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WIKIPEDIA_REST_TEMPLATE = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"

WEB_SEARCH_MAX_CHARS = 800
LOCAL_KB_MAX_HITS = 5
VISION_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB cap on inline base64 images
