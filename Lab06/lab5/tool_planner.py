"""Deterministic planning helpers for Lab 5 tool calling."""

import json
import re
from dataclasses import dataclass

from lab4 import language_detect

from .location_resolver import city_lookup_keys, fold_text


DEFAULT_TIMEZONE = "Europe/Warsaw"


@dataclass(frozen=True)
class PlannedToolCall:
    tool: str
    arguments: dict


def _unique_tool_key(name, arguments):
    return name, json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)


def already_called(tool_trace, name, arguments=None):
    if arguments is None:
        return any(step.get("tool") == name for step in tool_trace)
    key = _unique_tool_key(name, arguments)
    return any(
        _unique_tool_key(step.get("tool"), step.get("arguments")) == key
        for step in tool_trace
    )


def _weather_already_called(tool_trace, arguments):
    city_keys = city_lookup_keys((arguments or {}).get("city"))
    if not city_keys:
        return already_called(tool_trace, "get_weather", arguments)

    for step in tool_trace:
        if step.get("tool") != "get_weather":
            continue
        previous_keys = city_lookup_keys(
            (step.get("arguments") or {}).get("city")
        )
        result = step.get("result")
        if isinstance(result, dict):
            previous_keys.update(city_lookup_keys(result.get("city")))
        if city_keys & previous_keys:
            return True
    return False


def already_called_equivalent(tool_trace, name, arguments=None):
    if name == "get_weather" and arguments:
        return _weather_already_called(tool_trace, arguments)
    return already_called(tool_trace, name, arguments)


def is_image_request(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "opisz", "co widac", "co widać", "zdjec", "zdjęc",
        "obraz", "fot", "image", "photo", "picture", "read text",
    ))


def language_is_polish(text):
    try:
        detected = language_detect.detect_language(text)
    except Exception:
        detected = "unknown"
    if detected != "unknown":
        return detected == "pl"

    folded = fold_text(text)
    return bool(re.search(r"[ąćęłńóśźż]", (text or "").lower())) or any(
        keyword in folded for keyword in (
            "kto", "jest", "pogoda", "godzina", "opisz", "porownaj",
            "porównaj", "czy",
        )
    )


def strip_text_param_noise(text):
    cleaned = (text or "").strip()
    if cleaned.lower().startswith("text="):
        cleaned = cleaned[5:].strip()
    return cleaned.strip(" \"'")


def clean_location(text):
    value = strip_text_param_noise(text)
    value = re.sub(
        r"\b(?:jest|są|sa|is|are|będzie|bedzie|teraz|dzisiaj|dzis|"
        r"dziś|today|now)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", value.strip(" .,!?:;\"'()[]{}"))


def _split_locations(phrase):
    out = []
    for part in re.split(
        r"\s+(?:i|oraz|and|vs|versus)\s+|,",
        phrase or "",
        flags=re.IGNORECASE,
    ):
        cleaned = clean_location(part)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def extract_locations_after_preposition(text):
    cleaned_text = strip_text_param_noise(text)
    matches = re.findall(
        r"(?:\bw\b|\bwe\b|\bin\b|\bfor\b|\bdla\b)\s+([^?.,!]+)",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    locations = []
    for match in matches:
        for location in _split_locations(match):
            if location and location not in locations:
                locations.append(location)
    return locations


def extract_weather_cities(text):
    return extract_locations_after_preposition(text)


def looks_like_weather(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "pogod", "temperatur", "wiatr", "deszcz", "snieg",
        "spacer", "weather", "temperature", "wind", "rain", "snow",
    ))


def looks_like_calculation(text):
    folded = fold_text(text)
    if any(keyword in folded for keyword in (
        "ile to", "oblicz", "policz", "calculate", "sqrt", "sin",
        "cos", "tan", "log",
    )):
        return True
    return bool(re.search(r"\d\s*(?:\+|-|\*|/|//|%|\*\*)\s*\d", text or ""))


def looks_like_sentiment(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "sentyment", "wydzwiek", "wydźwięk", "sentiment",
        "pozytywn", "negatywn", "neutraln",
    ))


def looks_like_local_kb(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "lokaln", "local kb", "local knowledge", "baza wiedzy",
        "zapisane", "saved summary", "co wiemy lokalnie", "w lokalnej bazie",
    ))


def extract_local_kb_query(text):
    cleaned = re.sub(
        r"(?i)\b(co\s+wiemy\s+lokalnie\s+o|w\s+lokalnej\s+bazie|"
        r"lokaln(?:a|ej|e)?\s+baz(?:a|ie|y)\s+wiedzy|local\s+kb|"
        r"local\s+knowledge|zapisane\s+dane\s+o)\b",
        " ",
        text or "",
    )
    return cleaned.strip(" .,!?:;\"'()[]{}") or text


def extract_quoted_text(text):
    match = re.search(
        r'"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'',
        text or "",
    )
    if match:
        return _unescape_quoted_text(match.group(1) or match.group(2) or "").strip()
    return ""


def _unescape_quoted_text(text):
    return (
        (text or "")
        .replace(r"\"", '"')
        .replace(r"\'", "'")
        .replace(r"\\", "\\")
    )


def extract_sentiment_text(text):
    quoted = extract_quoted_text(text)
    if quoted:
        return quoted
    cleaned = re.sub(
        r"(?i)\b(jaki\s+jest\s+)?(?:sentyment|wyd[zź]wi[eę]k|sentiment)"
        r"(?:\s+tekstu)?\b[:\s-]*",
        "",
        text or "",
    )
    return cleaned.strip(" .,!?:;\"'()[]{}")


def extract_expression(text):
    expr = text or ""
    expr = re.sub(
        r"(?i)\b(ile to|oblicz|policz|calculate|what is|wynosi)\b",
        " ",
        expr,
    )
    return expr.strip().strip(" ?.")


def looks_like_translation(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "przetlumacz", "przetłumacz", "translate",
    ))


def extract_translation_target(text):
    match = re.search(
        r"\b(?:na|to|target(?:_language|_lang)?)\s*[=:]?\s*([a-z]{2})\b",
        fold_text(text),
    )
    if match:
        return match.group(1)
    return None


def extract_translation_text(text):
    quoted = extract_quoted_text(text)
    if quoted:
        return quoted
    cleaned = re.sub(
        r"(?i)\b(przet[lł]umacz|translate)\b",
        "",
        text or "",
    )
    cleaned = re.sub(r"(?i)\b(?:na|to)\s+[a-z]{2}\b", "", cleaned)
    return cleaned.strip(" .,!?:;\"'()[]{}")


def looks_like_summary(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "podsumuj", "stresc", "streść", "summarize", "summary",
    ))


def extract_summary_length(text):
    folded = fold_text(text)
    if any(keyword in folded for keyword in ("krotk", "short")):
        return "short"
    if any(keyword in folded for keyword in ("dlug", "dług", "long")):
        return "long"
    return "medium"


def looks_like_entity_extraction(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "wyciagnij encje", "wyciągnij encje", "znajdz encje",
        "znajdź encje", "extract entities", "ner",
    ))


def looks_like_datetime(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "dzis", "teraz", "aktualna data", "jaka godzina", "jaki dzien",
        "today", "now", "current date", "current time",
    ))


def extract_datetime_city(text):
    locations = extract_locations_after_preposition(text)
    return locations[-1] if locations else None


def looks_like_web_fact(text):
    folded = fold_text(text)
    return any(keyword in folded for keyword in (
        "kto jest", "kim jest", "czym jest", "co to jest", "who is",
        "what is", "ceo", "prezes", "aktualny", "obecny", "definition",
        "definicja",
    ))


def clean_entity_phrase(text):
    text = re.sub(r"[?.!,;:].*$", "", text or "")
    return re.sub(r"\s+", " ", text.strip(" \"'.,!?;:()[]{}"))


def extract_web_query(text):
    raw = strip_text_param_noise(text)
    folded = fold_text(raw)
    if any(keyword in folded for keyword in ("ceo", "prezes")):
        match = re.search(
            r"(?:ceo|prezes(?:em)?(?:\s+firmy)?)\s+(?:firmy\s+)?(.+)$",
            raw,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"(?:kto\s+jest|who\s+is).*?(?:ceo|prezes).*?(?:firmy\s+)?(.+)$",
                raw,
                flags=re.IGNORECASE,
            )
        entity = match.group(1) if match else raw
        entity = clean_entity_phrase(entity)
        return f"CEO {entity}" if entity else raw
    return raw


def needs_average_weather_search(text):
    folded = fold_text(text)
    return looks_like_weather(text) and any(keyword in folded for keyword in (
        "typow", "sredni", "średni", "average", "typical", "norma",
    ))


def build_average_weather_query(city, user_text):
    context = strip_text_param_noise(user_text)
    context = re.sub(r"\s+", " ", context).strip(" .,!?:;\"'()[]{}")
    if context:
        return f"{city} average weather climate {context[:140]}"
    return f"{city} average weather climate"


class ToolPlanner:
    """Deterministic safety net around the local controller model."""

    def repair_tool_args(self, name, arguments, user_text):
        arguments = dict(arguments or {})
        if name == "datetime_now":
            city = extract_datetime_city(user_text)
            if city and not arguments.get("city"):
                arguments.pop("tz", None)
                arguments["city"] = city
            elif not arguments.get("tz") and not arguments.get("city"):
                arguments["tz"] = DEFAULT_TIMEZONE
        elif name == "web_search" and looks_like_web_fact(user_text):
            arguments["query"] = extract_web_query(
                arguments.get("query") or user_text
            )
            if not arguments.get("language"):
                arguments["language"] = (
                    "pl" if language_is_polish(user_text) else "en"
                )
        elif name == "get_weather":
            if arguments.get("city"):
                arguments["city"] = clean_location(arguments["city"])
            else:
                cities = extract_weather_cities(user_text)
                if cities:
                    arguments["city"] = cities[0]
        elif name == "calculator" and not arguments.get("expression"):
            arguments["expression"] = extract_expression(user_text)
        elif name == "nlp_tools":
            self._repair_nlp_args(arguments, user_text)
        elif name == "local_knowledge":
            arguments["query"] = (
                arguments.get("query") or extract_local_kb_query(user_text)
            )
        return arguments

    def _repair_nlp_args(self, arguments, user_text):
        operation = (arguments.get("operation") or "").lower()
        if operation:
            arguments["operation"] = operation
        if not arguments.get("text") or arguments.get("text") == user_text:
            arguments["text"] = extract_quoted_text(user_text) or user_text
        if operation == "classify_sentiment":
            arguments["text"] = extract_sentiment_text(user_text) or arguments["text"]
            arguments.setdefault("language", "auto")
        elif operation == "translate":
            target = extract_translation_target(user_text)
            if target and not arguments.get("target_language"):
                arguments["target_language"] = target
            arguments.setdefault("language", "auto")
            extracted = extract_translation_text(user_text)
            if extracted:
                arguments["text"] = extracted
        elif operation == "summarize":
            summary_type = (
                "bullets" if "bullet" in fold_text(user_text)
                else "abstractive"
            )
            arguments.setdefault("summary_type", summary_type)
            arguments.setdefault("length", extract_summary_length(user_text))
        elif operation == "extract_entities":
            arguments.setdefault("language", "auto")

    def missing_required_info_answer(self, user_text):
        if looks_like_weather(user_text) and not extract_weather_cities(user_text):
            return (
                "Podaj miasto, dla którego mam sprawdzić pogodę. "
                "Nie zgaduję lokalizacji automatycznie."
            )
        if (
            looks_like_translation(user_text)
            and not extract_translation_target(user_text)
        ):
            return (
                "Podaj docelowy język jako dwuliterowy kod ISO, np. `en`, "
                "`pl`, `de`, `fr` albo `es`. Nie zgaduję nazw języków "
                "automatycznie."
            )
        return None

    def plan_fallback_calls(self, user_text, tool_trace):
        calls = []
        self._plan_weather(user_text, tool_trace, calls)
        self._plan_calculator(user_text, tool_trace, calls)
        self._plan_nlp(user_text, tool_trace, calls)
        self._plan_local_kb(user_text, tool_trace, calls)
        self._plan_datetime(user_text, tool_trace, calls)
        self._plan_web_fact(user_text, tool_trace, calls)
        return calls

    def _append_once(self, calls, tool, arguments, tool_trace):
        if not already_called_equivalent(tool_trace, tool, arguments):
            calls.append(PlannedToolCall(tool, arguments))

    def _plan_weather(self, user_text, tool_trace, calls):
        if not looks_like_weather(user_text):
            return
        cities = extract_weather_cities(user_text)
        for city in cities[:3]:
            self._append_once(
                calls,
                "get_weather",
                {"city": city},
                tool_trace,
            )

        if cities and needs_average_weather_search(user_text):
            query = build_average_weather_query(cities[0], user_text)
            language = "pl" if language_is_polish(user_text) else "en"
            self._append_once(
                calls,
                "web_search",
                {"query": query, "language": language},
                tool_trace,
            )

    def _plan_calculator(self, user_text, tool_trace, calls):
        if looks_like_calculation(user_text):
            self._append_once(
                calls,
                "calculator",
                {"expression": extract_expression(user_text)},
                tool_trace,
            )

    def _plan_nlp(self, user_text, tool_trace, calls):
        if already_called_equivalent(tool_trace, "nlp_tools"):
            return
        if looks_like_sentiment(user_text):
            text = extract_sentiment_text(user_text)
            if text:
                calls.append(PlannedToolCall("nlp_tools", {
                    "operation": "classify_sentiment",
                    "text": text,
                    "language": "auto",
                }))
        elif looks_like_translation(user_text):
            text = extract_translation_text(user_text)
            target = extract_translation_target(user_text)
            if text and target:
                calls.append(PlannedToolCall("nlp_tools", {
                    "operation": "translate",
                    "text": text,
                    "target_language": target,
                    "language": "auto",
                }))
        elif looks_like_summary(user_text):
            text = extract_quoted_text(user_text) or user_text
            calls.append(PlannedToolCall("nlp_tools", {
                "operation": "summarize",
                "text": text,
                "summary_type": (
                    "bullets" if "bullet" in fold_text(user_text)
                    else "abstractive"
                ),
                "length": extract_summary_length(user_text),
            }))
        elif looks_like_entity_extraction(user_text):
            text = extract_quoted_text(user_text) or user_text
            calls.append(PlannedToolCall("nlp_tools", {
                "operation": "extract_entities",
                "text": text,
                "language": "auto",
            }))

    def _plan_local_kb(self, user_text, tool_trace, calls):
        if looks_like_local_kb(user_text):
            self._append_once(
                calls,
                "local_knowledge",
                {"query": extract_local_kb_query(user_text)},
                tool_trace,
            )

    def _plan_datetime(self, user_text, tool_trace, calls):
        if not looks_like_datetime(user_text):
            return
        city = extract_datetime_city(user_text)
        arguments = {"city": city} if city else {"tz": DEFAULT_TIMEZONE}
        self._append_once(
            calls,
            "datetime_now",
            arguments,
            tool_trace,
        )

    def _plan_web_fact(self, user_text, tool_trace, calls):
        if not looks_like_web_fact(user_text):
            return
        arguments = {
            "query": extract_web_query(user_text),
            "language": "pl" if language_is_polish(user_text) else "en",
        }
        self._append_once(
            calls,
            "web_search",
            arguments,
            tool_trace,
        )
