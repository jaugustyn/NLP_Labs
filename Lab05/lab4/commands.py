"""Lab 4 command handlers — NER, NEL, NED, translation, summarization,
language detection, knowledge graph. Lab 1 + Lab 2 + Lab 3 commands are
delegated to the lab3 subpackage."""
import os
import re
import time

from config import (
    SUPPORTED_LANGUAGES,
    NER_METHODS,
    SUPPORTED_TRANSLATION_PAIRS,
    SUMMARY_TYPES,
    SUMMARY_LENGTHS,
    OLLAMA_MODEL,
    NEL_TOP_K,
)
from utils import parse_params, log_error, format_duration, truncate

from lab4 import language_detect
from lab4 import ner as ner_mod
from lab4 import nel as nel_mod
from lab4 import ned as ned_mod
from lab4 import translation as translation_mod
from lab4 import summarization as summarization_mod
from lab4 import knowledge_graph as kg_mod

from lab3.commands import register_handlers as register_lab123_handlers


# =====================================================================
#  HELP
# =====================================================================

HELP_TEXT = (
    "NLP Bot — Lab 1 + Lab 2 + Lab 3 + Lab 4\n\n"
    "--- Lab 1 ---\n"
    "/task <name> \"text\" \"class\"\n"
    "/full_pipeline \"text\" \"class\"\n"
    "/classifier \"text\"\n"
    "/stats\n\n"
    "Tasks: tokenize, remove_stopwords, lemmatize, stemming,\n"
    "stats, n-grams, plot_histogram, plot_wordcloud, plot_barchart\n\n"
    "--- Lab 2 ---\n"
    "/classify dataset=<name> method=<model> gridsearch=<true/false> run=<n>\n"
    "  Datasets: 20news_group, imdb, amazon, ag_news\n"
    "  Methods: nb, rf, mlp, logreg, all\n\n"
    "--- Lab 3 ---\n"
    "/sentiment method=<m> text=\"...\" [dataset=<d>]\n"
    "  Methods: rule, nb, rf, transformer, textblob, stanza,\n"
    "  simplernn, lstm, gru\n\n"
    "/train model=<simplernn|lstm|gru> dataset=<amazon|imdb|custom>\n"
    "/compare dataset=<amazon|imdb|custom> methods=<m1,m2,...>\n"
    "/add_sentiment \"text\" \"label\"\n"
    "/models\n\n"
    "--- Lab 4 ---\n"
    "/language_detect text=\"...\"\n"
    "/ner method=<spacy|stanza> text=\"...\" [language=auto|en|pl]\n"
    "/nel text=\"...\" [language=auto|en|pl]\n"
    "/ned entity=\"...\" context=\"...\" [language=auto|en|pl]\n"
    "/analyze_entities text=\"...\" [link=true|false] [language=auto|en|pl]\n"
    "/translate text=\"...\" target_lang=<en|pl|de|fr|es> [source_lang=auto]\n"
    "/summarize text=\"...\" [summary_type=abstractive|extractive|bullets|custom]\n"
    "  [length=short|medium|long] [prompt=\"...\"]\n"
    "/knowledge_graph text=\"...\" [language=auto|en|pl]\n\n"
    "Examples:\n"
    "/language_detect text=\"Warszawa to stolica Polski\"\n"
    "/ner method=spacy text=\"Apple was founded by Steve Jobs\"\n"
    "/nel text=\"Steve Jobs\" language=en\n"
    "/ned entity=\"Paris\" context=\"Paris is the capital of France\"\n"
    "/translate text=\"Hello world\" target_lang=pl\n"
    "/summarize text=\"...long text...\" summary_type=bullets length=short\n"
    "/analyze_entities text=\"Elon Musk founded SpaceX\" link=true\n"
    "/knowledge_graph text=\"Elon Musk founded SpaceX in 2002\"\n"
)


# =====================================================================
#  Helpers
# =====================================================================

def _resolve_lang(text, requested):
    if requested and requested != "auto":
        return requested
    return language_detect.detect_language(text)


def _get_lang_param(params):
    """Accept both `language=` (spec) and `lang=` (alias)."""
    return params.get("language") or params.get("lang")


def _format_entities_with_offsets(entities, max_items=50):
    if not entities:
        return "(no entities)"
    lines = []
    for e in entities[:max_items]:
        lines.append(
            f"- {e['text']} ({e['label']}) [{e['start']}:{e['end']}]"
        )
    return "\n".join(lines)


# =====================================================================
#  /language_detect
# =====================================================================

def _handle_language_detect(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            bot.reply_to(
                message,
                "Usage: /language_detect text=\"your text\"\n\n"
                "Example:\n"
                "/language_detect text=\"Warszawa to stolica Polski\""
            )
            return
        lang, confidence = language_detect.detect_language_with_confidence(text)
        bot.reply_to(
            message,
            f"Language: {lang}\n"
            f"Confidence: {confidence:.2%}"
        )
    except Exception as e:
        log_error("language_detect", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /ner
# =====================================================================

def _handle_ner(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            bot.reply_to(
                message,
                "Usage: /ner method=<spacy|stanza> text=\"your text\" "
                "[language=auto|en|pl]\n\n"
                f"Available methods: {', '.join(NER_METHODS)}\n"
                "Example:\n"
                "/ner method=spacy text=\"Apple was founded by Steve Jobs\""
            )
            return

        method = params.get("method", "spacy").lower()
        if method not in NER_METHODS:
            bot.reply_to(
                message,
                f"Unknown method: '{method}'\n"
                f"Available: {', '.join(NER_METHODS)}"
            )
            return

        lang = _resolve_lang(text, _get_lang_param(params))
        if lang not in SUPPORTED_LANGUAGES:
            bot.reply_to(
                message,
                f"Language '{lang}' not supported. Falling back to 'en'."
            )
            lang = "en"

        bot.reply_to(message, f"Running NER ({method}, language={lang})...")
        t0 = time.time()
        entities = ner_mod.extract_entities(text, method=method, lang=lang)
        elapsed = format_duration(time.time() - t0)

        bot.send_message(
            message.chat.id,
            f"Method: {method}\n"
            f"Language: {lang}\n"
            f"TEXT: {truncate(text, 200)}\n\n"
            f"ENTITIES ({len(entities)}, {elapsed}):\n"
            + _format_entities_with_offsets(entities)
        )
    except Exception as e:
        log_error("ner", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /nel
# =====================================================================

def _handle_nel(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            bot.reply_to(
                message,
                "Usage: /nel text=\"your text\" [language=auto|en|pl]\n\n"
                "Example:\n"
                "/nel text=\"Steve Jobs\" language=en"
            )
            return

        lang = _resolve_lang(text, _get_lang_param(params))
        bot.reply_to(
            message,
            f"Running NER + Wikidata candidate search (language={lang})..."
        )

        entities = ner_mod.extract_entities(text, method="spacy", lang=lang)
        if not entities:
            # Treat the whole text as a single entity name (spec example)
            entities = [{"text": text, "label": "MISC",
                         "start": 0, "end": len(text)}]

        sections = []
        for ent in entities[:10]:
            ranked = ned_mod.disambiguate(ent["text"], text, lang=lang)
            section = [f"Entity: {ent['text']} ({ent['label']})"]
            if not ranked or "error" in ranked[0]:
                section.append("  No candidates found.")
                sections.append("\n".join(section))
                continue
            section.append("Candidates:")
            for i, c in enumerate(ranked[:NEL_TOP_K], 1):
                qid = c.get("qid", "?")
                wiki_url = ned_mod.get_wikipedia_url(qid, lang=lang)
                desc = truncate(c.get("description", ""), 80)
                section.append(
                    f"{i}. {c.get('label', '?')} ({qid}) - {desc}"
                )
                if wiki_url:
                    section.append(f"   - Wikipedia: {wiki_url}")
                section.append(f"   - Confidence: {c.get('score', 0):.2f}")
            sections.append("\n".join(section))

        bot.send_message(message.chat.id, "\n\n".join(sections))
    except Exception as e:
        log_error("nel", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /ned
# =====================================================================

def _handle_ned(bot, message):
    try:
        params = parse_params(message.text)
        entity = params.get("entity")
        context = params.get("context")
        if not entity or not context:
            bot.reply_to(
                message,
                "Usage: /ned entity=\"name\" context=\"context text\" "
                "[language=auto|en|pl]\n\n"
                "Example:\n"
                "/ned entity=\"Paris\" "
                "context=\"Paris is the capital of France\""
            )
            return

        lang = _resolve_lang(context, _get_lang_param(params))
        bot.reply_to(message, f"Disambiguating '{entity}' (language={lang})...")

        ranked = ned_mod.disambiguate(entity, context, lang=lang)
        if not ranked or "error" in ranked[0]:
            err = ranked[0].get("error") if ranked else "no candidates"
            bot.send_message(message.chat.id, f"No candidates ({err}).")
            return

        lines = [f"Top {min(NEL_TOP_K, len(ranked))} candidates for '{entity}':\n"]
        for i, c in enumerate(ranked[:NEL_TOP_K], 1):
            lines.append(
                f"{i}. {c.get('qid', '?')} - {c.get('label', '?')} "
                f"(confidence={c.get('score', 0):.2f})\n"
                f"   {truncate(c.get('description', ''), 100)}"
            )
        bot.send_message(message.chat.id, "\n".join(lines))
    except Exception as e:
        log_error("ned", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /analyze_entities
# =====================================================================

def _handle_analyze_entities(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            bot.reply_to(
                message,
                "Usage: /analyze_entities text=\"your text\" "
                "[link=true|false] [language=auto|en|pl]\n\n"
                "Runs NER (spaCy + Stanza) and merges results. With "
                "link=true (default) each entity is linked to Wikidata "
                "and a Wikipedia URL is resolved.\n\n"
                "Example:\n"
                "/analyze_entities text=\"Elon Musk founded SpaceX\" link=true"
            )
            return

        link_flag = params.get("link", "true").lower() != "false"
        lang = _resolve_lang(text, _get_lang_param(params))
        bot.reply_to(
            message,
            f"Full entity analysis (language={lang}, link={link_flag})..."
        )
        t0 = time.time()

        spacy_ents = ner_mod.extract_entities(text, method="spacy", lang=lang)
        try:
            stanza_ents = ner_mod.extract_entities(
                text, method="stanza", lang=lang
            )
        except Exception:
            stanza_ents = []
        merged = ner_mod.merge_entities(spacy_ents, stanza_ents)

        results = []
        if link_flag:
            for e in merged:
                best = ned_mod.best_match(e["text"], text, lang=lang)
                results.append({**e, "link": best})
        else:
            results = [{**e, "link": None} for e in merged]

        elapsed = format_duration(time.time() - t0)

        lines = [
            f"Language: {lang}",
            f"spaCy: {len(spacy_ents)} | Stanza: {len(stanza_ents)} | "
            f"merged: {len(merged)} ({elapsed})",
            "",
            "ENTITIES FOUND:",
        ]
        for e in results[:25]:
            lines.append(
                f"- {e['text']} ({e['label']}) [{e['start']}:{e['end']}]"
            )
            link = e.get("link")
            if link and "qid" in link:
                qid = link["qid"]
                lines.append(f"  Wikidata: {qid}")
                wiki_url = ned_mod.get_wikipedia_url(qid, lang=lang)
                if wiki_url:
                    lines.append(f"  Wikipedia: {wiki_url}")
            elif link_flag:
                lines.append("  Wikidata: Not found")
        bot.send_message(message.chat.id, "\n".join(lines))
    except Exception as e:
        log_error("analyze_entities", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /translate
# =====================================================================

def _handle_translate(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            pairs = ", ".join(
                f"{a}->{b}" for a, b in SUPPORTED_TRANSLATION_PAIRS
            )
            bot.reply_to(
                message,
                "Usage: /translate text=\"your text\" "
                "target_lang=<en|pl|de|fr|es> [source_lang=auto]\n\n"
                f"Supported pairs: {pairs}\n\n"
                "Example:\n"
                "/translate text=\"Hello world\" target_lang=pl"
            )
            return

        src = (params.get("source_lang") or params.get("src")
               or "auto").lower()
        tgt = (params.get("target_lang") or params.get("tgt")
               or "en").lower()
        auto_src = src == "auto"
        if auto_src:
            src = language_detect.detect_language(text)
            # langdetect can mislabel very short texts (e.g. "Hello world"
            # -> "nl"). If detected source has no model for the target,
            # fall back to a supported source — preferring 'en'.
            if (src, tgt) not in SUPPORTED_TRANSLATION_PAIRS:
                fallbacks = [s for (s, t) in SUPPORTED_TRANSLATION_PAIRS
                             if t == tgt]
                if "en" in fallbacks:
                    src = "en"
                elif fallbacks:
                    src = fallbacks[0]
        if src == tgt:
            bot.send_message(
                message.chat.id,
                f"Source: {src}\nTarget: {tgt}\nTranslation: {text}\n"
                "(source language equals target)"
            )
            return

        bot.reply_to(
            message,
            f"Translating {src} -> {tgt} "
            "(first use downloads the model, ~300 MB)..."
        )
        t0 = time.time()
        translated = translation_mod.translate(text, src, tgt)
        elapsed = format_duration(time.time() - t0)

        bot.send_message(
            message.chat.id,
            f"Source: {src}\n"
            f"Target: {tgt}\n"
            f"Translation: {translated}\n\n"
            f"Generation time: {elapsed}"
        )
    except Exception as e:
        log_error("translate", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /summarize
# =====================================================================

def _handle_summarize(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            bot.reply_to(
                message,
                "Usage: /summarize text=\"your text\" "
                "[summary_type=abstractive|extractive|bullets|custom] "
                "[length=short|medium|long] [prompt=\"...\"]\n\n"
                f"Types: {', '.join(SUMMARY_TYPES)}\n"
                f"Lengths: {', '.join(SUMMARY_LENGTHS)}\n\n"
                "Example:\n"
                "/summarize text=\"...long text...\" "
                "summary_type=bullets length=short"
            )
            return

        kind = (params.get("summary_type") or params.get("type")
                or "abstractive").lower()
        length = params.get("length", "medium").lower()
        custom_prompt = params.get("prompt")

        if kind not in SUMMARY_TYPES:
            bot.reply_to(
                message,
                f"Unknown summary_type: '{kind}'. "
                f"Available: {', '.join(SUMMARY_TYPES)}"
            )
            return
        if length not in SUMMARY_LENGTHS:
            bot.reply_to(
                message,
                f"Unknown length: '{length}'. "
                f"Available: {', '.join(SUMMARY_LENGTHS)}"
            )
            return
        if kind == "custom" and not custom_prompt:
            bot.reply_to(
                message,
                "summary_type=custom requires prompt=\"your instructions\""
            )
            return

        if not summarization_mod.is_ollama_available():
            bot.reply_to(
                message,
                "Ollama is not reachable on localhost:11434.\n"
                f"Run 'ollama serve' and 'ollama pull {OLLAMA_MODEL}'."
            )
            return

        token_count = len(text.split())
        bot.reply_to(
            message,
            f"Summarizing (summary_type={kind}, length={length}) "
            f"via {OLLAMA_MODEL}..."
        )
        t0 = time.time()
        summary = summarization_mod.summarize(
            text, kind=kind, length=length, custom_prompt=custom_prompt
        )
        elapsed = format_duration(time.time() - t0)

        try:
            saved_path = summarization_mod.save_summary(
                text, summary, kind=kind, length=length,
                model_name=OLLAMA_MODEL,
            )
            saved_rel = os.path.relpath(saved_path)
        except Exception as save_err:
            log_error("summarize/save", save_err)
            saved_rel = "(not saved)"

        bot.send_message(
            message.chat.id,
            f"Model: {OLLAMA_MODEL}\n"
            f"Text length: {token_count} tokens\n"
            f"Summary type: {kind.capitalize()}\n"
            f"Summary length: {length.capitalize()}\n\n"
            f"SUMMARY:\n{summary}\n\n"
            f"Generation time: {elapsed}\n"
            f"Saved to: {saved_rel}"
        )
    except Exception as e:
        log_error("summarize", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  /knowledge_graph
# =====================================================================

def _handle_knowledge_graph(bot, message):
    try:
        params = parse_params(message.text)
        text = params.get("text")
        if not text:
            bot.reply_to(
                message,
                "Usage: /knowledge_graph text=\"your text\" "
                "[language=auto|en|pl]\n\n"
                "Builds an entity co-occurrence graph (per sentence) with "
                "Wikidata links and saves a PNG visualization.\n\n"
                "Example:\n"
                "/knowledge_graph text=\"Elon Musk founded SpaceX in 2002\""
            )
            return

        lang = _resolve_lang(text, _get_lang_param(params))
        bot.reply_to(message, f"Building knowledge graph (language={lang})...")

        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        per_sentence = []
        for sentence in sentences:
            ents = ner_mod.extract_entities(sentence, method="spacy", lang=lang)
            linked = nel_mod.link_entities(ents, lang=lang)
            per_sentence.append(linked)

        graph = kg_mod.build_graph(per_sentence)
        stats = kg_mod.graph_stats(graph)
        if graph.number_of_nodes() == 0:
            bot.send_message(message.chat.id, "No entities to build a graph.")
            return

        path = kg_mod.plot_graph(graph, title=f"Knowledge Graph ({lang})")

        caption = (
            f"Nodes: {stats['nodes']}\n"
            f"Edges: {stats['edges']}\n"
            f"Components: {stats['components']}\n"
            f"Density: {stats['density']:.3f}"
        )
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                bot.send_photo(message.chat.id, f, caption=caption)
        else:
            bot.send_message(message.chat.id, caption)
    except Exception as e:
        log_error("knowledge_graph", e)
        bot.reply_to(message, f"Error: {type(e).__name__}: {e}")


# =====================================================================
#  Registration
# =====================================================================

def register_handlers(bot):
    @bot.message_handler(commands=["start", "help"])
    def cmd_help(message):
        bot.reply_to(message, HELP_TEXT)

    @bot.message_handler(commands=["language_detect"])
    def cmd_language_detect(message):
        _handle_language_detect(bot, message)

    @bot.message_handler(commands=["ner"])
    def cmd_ner(message):
        _handle_ner(bot, message)

    @bot.message_handler(commands=["nel"])
    def cmd_nel(message):
        _handle_nel(bot, message)

    @bot.message_handler(commands=["ned"])
    def cmd_ned(message):
        _handle_ned(bot, message)

    @bot.message_handler(commands=["analyze_entities"])
    def cmd_analyze_entities(message):
        _handle_analyze_entities(bot, message)

    @bot.message_handler(commands=["translate"])
    def cmd_translate(message):
        _handle_translate(bot, message)

    @bot.message_handler(commands=["summarize"])
    def cmd_summarize(message):
        _handle_summarize(bot, message)

    @bot.message_handler(commands=["knowledge_graph"])
    def cmd_knowledge_graph(message):
        _handle_knowledge_graph(bot, message)

    register_lab123_handlers(bot)
