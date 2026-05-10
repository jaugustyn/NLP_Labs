"""Multi-step tool-calling agent built on top of Ollama /api/chat."""
import json
import re

from . import ollama_client
from . import tools as tools_mod
from config import AGENT_MODEL, MAX_AGENT_ITERATIONS


SYSTEM_PROMPT = (
    "You are an NLP assistant for a Telegram bot. You MUST use the "
    "available tools instead of guessing whenever the user's request "
    "matches one of these categories:\n"
    "- weather / temperature / wind / 'czy dobra pogoda' → get_weather\n"
    "- arithmetic, sqrt, sin/cos/log, percentages, anything numeric → calculator\n"
    "- 'kto jest', 'who is', 'czym jest', 'what is', current facts, CEOs, "
    "definitions → web_search (try local_knowledge first only if the "
    "user explicitly asks 'co wiemy lokalnie' / 'in the local KB')\n"
    "- image / photo / 'co widac', 'opisz zdjecie' → analyze_image\n"
    "- 'today', 'now', 'jaka godzina', 'jaki dzień' → datetime_now\n"
    "- translate / summarize / extract entities / sentiment of a given "
    "text → nlp_tools\n\n"
    "Rules:\n"
    "1. NEVER fabricate weather numbers, math results, dates, or facts "
    "that a tool can provide. If unsure whether to call a tool — call it.\n"
    "2. You may call several tools in one turn (e.g. get_weather('Warsaw') "
    "AND get_weather('Paris') in parallel for a comparison) and chain "
    "tools across iterations (e.g. get_weather then web_search).\n"
    "3. After tool results arrive, base your final answer ONLY on the "
    "actual numbers/text from those results — do not invent values.\n"
    "4. Reply in the same language the user used (Polish or English). "
    "Keep answers short and useful. Do not dump raw JSON.\n\n"
    "Examples of REQUIRED tool use:\n"
    "- 'Kto jest CEO Tesli?' → web_search(query='CEO Tesla')\n"
    "- 'Jaka jest pogoda w Warszawie?' → get_weather(city='Warsaw')\n"
    "- 'Ile to (12+7)*sqrt(81)?' → calculator(expression='(12+7)*sqrt(81)')\n"
)


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


def _unique_tool_key(name, args):
    return name, json.dumps(args or {}, sort_keys=True, ensure_ascii=False)


def _already_called(tool_trace, name, args=None):
    if args is None:
        return any(t.get("tool") == name for t in tool_trace)
    key = _unique_tool_key(name, args)
    return any(
        _unique_tool_key(t.get("tool"), t.get("arguments")) == key
        for t in tool_trace
    )


def _append_tool_result(messages, name, args, result):
    """Append a synthetic assistant tool call and the tool result.

    Ollama's current API examples use `tool_name` on role=tool messages.
    Keep `name` too because older OpenAI-style integrations expect it.
    """
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_name": name,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    })


def _execute_tool(messages, tool_trace, iteration, name, args):
    result = tools_mod.call_tool(name, args)
    tool_trace.append({
        "iteration": iteration,
        "tool": name,
        "arguments": args,
        "result": result,
    })
    _append_tool_result(messages, name, args, result)
    return result


def _repair_tool_args(name, args, user_text):
    args = dict(args or {})
    if name == "datetime_now":
        city = _extract_datetime_city(user_text)
        if city and not args.get("city"):
            args.pop("tz", None)
            args["city"] = city
        elif not args.get("tz") and not args.get("city"):
            args["tz"] = "Europe/Warsaw"
    elif name == "web_search" and _looks_like_web_fact(user_text):
        args["query"] = _extract_web_query(args.get("query") or user_text)
        if not args.get("language"):
            args["language"] = "pl" if _language_is_polish(user_text) else "en"
    elif name == "get_weather" and args.get("city"):
        args["city"] = _clean_location(args["city"])
    elif name == "nlp_tools":
        op = (args.get("operation") or "").lower()
        if op == "classify_sentiment":
            args["operation"] = "classify_sentiment"
            if not args.get("text") or args.get("text") == user_text:
                args["text"] = _extract_sentiment_text(user_text) or user_text
            if not args.get("language"):
                args["language"] = "auto"
    return args


def _tool_results(tool_trace, name):
    return [
        t.get("result") or {}
        for t in tool_trace
        if t.get("tool") == name and isinstance(t.get("result"), dict)
    ]


def _is_image_request(text):
    folded = _fold(text)
    return any(k in folded for k in (
        "opisz", "co widac", "co widać", "zdjec", "zdjęc",
        "obraz", "fot", "image", "photo", "picture", "read text",
    ))


def _language_is_polish(text):
    folded = _fold(text)
    return bool(re.search(r"[ąćęłńóśźż]", (text or "").lower())) or any(
        k in folded for k in (
            "kto", "jest", "pogoda", "godzina", "opisz", "porownaj",
            "porównaj", "czy", "w warsz", "w paryz", "w pary",
        )
    )


def _format_number(value):
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _format_weather_answer(user_text, tool_trace):
    weather = [r for r in _tool_results(tool_trace, "get_weather") if not r.get("error")]
    if not weather:
        return None
    parts = []
    for item in weather:
        city = item.get("city") or "miasto"
        country = item.get("country")
        place = f"{city}, {country}" if country else city
        temp = item.get("temperature_c")
        wind = item.get("wind_kmh")
        desc = item.get("description") or "brak opisu"
        observed = item.get("observed_at")
        sentence = (
            f"{place}: {_format_number(temp)}°C, {desc}, "
            f"wiatr {_format_number(wind)} km/h"
        )
        if observed:
            sentence += f" (pomiar: {observed})"
        parts.append(sentence + ".")

    if _needs_average_weather_search(user_text):
        searches = [r for r in _tool_results(tool_trace, "web_search") if not r.get("error")]
        if searches:
            src = searches[-1]
            summary = src.get("summary") or ""
            folded_summary = _fold(summary)
            if any(k in folded_summary for k in (
                "climate", "temperature", "weather", "average",
                "temperatur", "pogod", "klimat", "średni", "sredni",
            )):
                parts.append(
                    "Do oceny typowości użyłem też wyniku wyszukiwania "
                    f"`{src.get('query')}`. {summary[:350]}"
                )
            else:
                parts.append(
                    "Wynik wyszukiwania nie zawierał konkretnych średnich "
                    "majowych, więc nie porównuję tego na siłę."
                )
        else:
            parts.append(
                "Nie udało się pobrać wiarygodnego wyniku o średniej majowej, "
                "więc mogę podać tylko aktualne warunki."
            )
    return "\n".join(parts)


def _format_datetime_answer(user_text, tool_trace):
    all_results = _tool_results(tool_trace, "datetime_now")
    results = [r for r in all_results if not r.get("error")]
    if not results:
        errors = [r for r in all_results if r.get("error")]
        if errors:
            return errors[-1].get("error")
        return None
    result = results[-1]
    tz = result.get("timezone")
    city = result.get("city")
    country = result.get("country")
    place = f"{city}, {country}" if city and country else (city or f"strefie {tz}")
    return (
        f"Dla lokalizacji {place} ({tz}) aktualny czas to "
        f"{result.get('time')} {result.get('date')} ({result.get('weekday')})."
    )


def _format_web_fact_answer(user_text, tool_trace):
    searches = [r for r in _tool_results(tool_trace, "web_search") if not r.get("error")]
    if not searches:
        return None
    result = searches[-1]
    facts = result.get("facts") or {}
    ceos = facts.get("chief_executive_officer") or []
    if ceos:
        query = result.get("query") or ""
        subject = query[4:].strip() if query.lower().startswith("ceo ") else (
            result.get("title") or query
        )
        return f"CEO {subject} to {', '.join(ceos)}."
    return None


def _format_image_answer(user_text, tool_trace):
    if not _is_image_request(user_text):
        return None
    results = [
        r for r in _tool_results(tool_trace, "analyze_image")
        if not r.get("error")
    ]
    if not results:
        errors = [
            r.get("error") for r in _tool_results(tool_trace, "analyze_image")
            if r.get("error")
        ]
        return errors[-1] if errors else None
    description = (results[-1].get("description") or "").strip()
    return description if description else None


def _answer_from_tools(user_text, tool_trace):
    for formatter in (
        _format_image_answer,
        _format_nlp_answer,
        _format_web_fact_answer,
        _format_datetime_answer,
        _format_weather_answer,
    ):
        answer = formatter(user_text, tool_trace)
        if answer:
            return answer
    return None


def _looks_like_weather(text):
    folded = _fold(text)
    return any(k in folded for k in (
        "pogod", "temperatur", "wiatr", "deszcz", "snieg",
        "spacer", "weather", "temperature", "wind", "rain", "snow",
    ))


def _normalise_city(city):
    return _clean_location(city)


def _strip_text_param_noise(text):
    cleaned = (text or "").strip()
    if cleaned.lower().startswith("text="):
        cleaned = cleaned[5:].strip()
    return cleaned.strip(" \"'")


def _clean_location(text):
    value = _strip_text_param_noise(text)
    value = re.sub(
        r"\b(?:jest|są|sa|is|are|będzie|bedzie|teraz|dzisiaj|dzis|dziś|today|now)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", value.strip(" .,!?:;\"'()[]{}"))


def _split_locations(phrase):
    out = []
    for part in re.split(r"\s+(?:i|oraz|and|vs|versus)\s+|,", phrase or "", flags=re.IGNORECASE):
        cleaned = _clean_location(part)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _extract_locations_after_preposition(text):
    cleaned_text = _strip_text_param_noise(text)
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


def _extract_weather_cities(text):
    return _extract_locations_after_preposition(text)


def _looks_like_calculation(text):
    folded = _fold(text)
    if any(k in folded for k in (
        "ile to", "oblicz", "policz", "calculate", "sqrt", "sin",
        "cos", "tan", "log",
    )):
        return True
    return bool(re.search(r"\d\s*(?:\+|-|\*|/|//|%|\*\*)\s*\d", text or ""))


def _looks_like_sentiment(text):
    folded = _fold(text)
    return any(k in folded for k in (
        "sentyment", "wydzwiek", "wydźwięk", "sentiment",
        "pozytywn", "negatywn", "neutraln",
    ))


def _extract_quoted_text(text):
    match = re.search(r'"([^"]+)"|\'([^\']+)\'', text or "")
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return ""


def _extract_sentiment_text(text):
    quoted = _extract_quoted_text(text)
    if quoted:
        return quoted
    cleaned = re.sub(
        r"(?i)\b(jaki\s+jest\s+)?(?:sentyment|wyd[zź]wi[eę]k|sentiment)"
        r"(?:\s+tekstu)?\b[:\s-]*",
        "",
        text or "",
    )
    return cleaned.strip(" .,!?:;\"'()[]{}")


def _format_nlp_answer(user_text, tool_trace):
    results = [
        r for r in _tool_results(tool_trace, "nlp_tools")
        if r.get("operation") == "classify_sentiment" and not r.get("error")
    ]
    if not results:
        return None
    result = results[-1]
    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)):
        conf_text = f"{confidence:.2f}"
    else:
        conf_text = str(confidence)
    return f"Sentyment: {result.get('label')} (pewność {conf_text})."


def _extract_expression(text):
    expr = text or ""
    expr = re.sub(
        r"(?i)\b(ile to|oblicz|policz|calculate|what is|wynosi)\b",
        " ",
        expr,
    )
    expr = expr.strip().strip(" ?.")
    return expr


def _looks_like_datetime(text):
    folded = _fold(text)
    return any(k in folded for k in (
        "dzis", "teraz", "aktualna data", "jaka godzina", "jaki dzien",
        "today", "now", "current date", "current time",
    ))


def _extract_datetime_city(text):
    locations = _extract_locations_after_preposition(text)
    return locations[-1] if locations else None


def _looks_like_web_fact(text):
    folded = _fold(text)
    return any(k in folded for k in (
        "kto jest", "kim jest", "czym jest", "co to jest", "who is",
        "what is", "ceo", "prezes", "aktualny", "obecny", "definition",
        "definicja",
    ))


def _extract_web_query(text):
    raw = _strip_text_param_noise(text)
    folded = _fold(raw)
    if any(k in folded for k in ("ceo", "prezes")):
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
        entity = re.sub(r"[?.!,;:].*$", "", entity).strip(" \"'")
        entity = _normalise_entity_phrase(entity)
        return f"CEO {entity}" if entity else raw
    return raw


def _normalise_entity_phrase(text):
    words = []
    for token in re.split(r"\s+", text or ""):
        token = token.strip(" \"'.,!?;:()[]{}")
        if not token:
            continue
        lower = token.lower()
        if lower.endswith("'a"):
            token = token[:-2]
        elif lower.endswith("u") and len(token) > 4:
            token = token[:-1]
        elif lower.endswith("i") and len(token) > 4:
            token = token[:-1] + "a"
        words.append(token)
    return " ".join(words)


def _needs_average_weather_search(text):
    folded = _fold(text)
    return _looks_like_weather(text) and any(k in folded for k in (
        "typow", "sredni", "średni", "average", "typical", "norma",
        "maja", "may",
    ))


def _fallback_tool_calls(user_text, tool_trace):
    """Deterministic safety net when a small local model skips a tool.

    The model still gets the first chance. This only fires for obvious
    cases where answering from memory would be worse than using the
    required Lab05 tools.
    """
    calls = []

    if _looks_like_weather(user_text):
        cities = _extract_weather_cities(user_text) or ["Warsaw"]
        for city in cities[:3]:
            args = {"city": city}
            if not _already_called(tool_trace, "get_weather", args):
                calls.append(("get_weather", args))

        if _needs_average_weather_search(user_text):
            city = cities[0] if cities else "Warsaw"
            query = f"average weather {city} May"
            args = {"query": query, "language": "en"}
            if not _already_called(tool_trace, "web_search", args):
                calls.append(("web_search", args))

    if _looks_like_calculation(user_text):
        args = {"expression": _extract_expression(user_text)}
        if not _already_called(tool_trace, "calculator", args):
            calls.append(("calculator", args))

    if _looks_like_sentiment(user_text) and not _already_called(tool_trace, "nlp_tools"):
        text = _extract_sentiment_text(user_text)
        if text:
            calls.append(("nlp_tools", {
                "operation": "classify_sentiment",
                "text": text,
                "language": "auto",
            }))

    if _looks_like_datetime(user_text) and not _already_called(tool_trace, "datetime_now"):
        city = _extract_datetime_city(user_text)
        args = {"city": city} if city else {"tz": "Europe/Warsaw"}
        calls.append(("datetime_now", args))

    if _looks_like_web_fact(user_text) and not _already_called(tool_trace, "web_search"):
        args = {
            "query": _extract_web_query(user_text),
            "language": "pl" if _language_is_polish(user_text) else "en",
        }
        calls.append(("web_search", args))

    return calls


def _normalise_tool_call(tc):
    """Ollama returns tool_calls in OpenAI-ish format; flatten to
    (name, arguments_dict, id)."""
    fn = tc.get("function") or {}
    name = fn.get("name") or tc.get("name") or ""
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {}
    return name, args, tc.get("id", "")


class Agent:
    def __init__(self, model=None, max_iters=MAX_AGENT_ITERATIONS,
                 enabled_tools=None, system_prompt=None):
        self.model = model or AGENT_MODEL
        self.max_iters = max_iters
        self.enabled_tools = enabled_tools  # None = all
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def run(self, user_text, images=None, history=None,
            extra_context=None):
        """Run one agent turn.

        Args:
            user_text: the user's message.
            images: optional list of local image paths attached.
            history: previous messages (list of {role, content}).
            extra_context: optional system-level note appended to the
                system prompt (e.g. "An image is attached at <path>").

        Returns dict with: answer, tool_trace, iterations, messages.
        """
        sys_content = self.system_prompt
        if extra_context:
            sys_content = sys_content + "\n\n" + extra_context

        messages = [{"role": "system", "content": sys_content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        tool_payload = tools_mod.get_tools_payload(self.enabled_tools)
        tool_trace = []
        iterations = 0

        # The controller model is text-only. Analyse attached images
        # immediately and feed the textual result back into the agent
        # loop so further tools can still be chained by the model.
        if images:
            for image_path in images:
                args = {
                    "image_path": image_path,
                    "prompt": user_text or "Describe this image.",
                }
                _execute_tool(
                    messages, tool_trace, 0, "analyze_image", args,
                )
            image_answer = _answer_from_tools(user_text, tool_trace)
            if image_answer and not _fallback_tool_calls(user_text, tool_trace):
                return {
                    "answer": image_answer,
                    "tool_trace": tool_trace,
                    "iterations": 0,
                    "messages": messages,
                }

        for i in range(self.max_iters):
            iterations = i + 1
            try:
                resp = ollama_client.chat(
                    messages=messages,
                    tools=tool_payload,
                    model=self.model,
                )
            except Exception as e:
                return {
                    "answer": f"Agent error: {e}",
                    "tool_trace": tool_trace,
                    "iterations": iterations,
                    "messages": messages,
                    "error": str(e),
                }

            msg = (resp or {}).get("message") or {}
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []

            # Append assistant message (preserve tool_calls for protocol)
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                fallback_calls = _fallback_tool_calls(user_text, tool_trace)
                if fallback_calls:
                    for name, args in fallback_calls:
                        _execute_tool(
                            messages, tool_trace, iterations, name, args,
                        )
                    direct_answer = _answer_from_tools(user_text, tool_trace)
                    if direct_answer:
                        return {
                            "answer": direct_answer,
                            "tool_trace": tool_trace,
                            "iterations": iterations,
                            "messages": messages,
                        }
                    continue
                direct_answer = _answer_from_tools(user_text, tool_trace)
                if direct_answer:
                    content = direct_answer
                return {
                    "answer": content or "(empty answer)",
                    "tool_trace": tool_trace,
                    "iterations": iterations,
                    "messages": messages,
                }

            # Execute tools and append their results
            for tc in tool_calls:
                name, args, tc_id = _normalise_tool_call(tc)
                args = _repair_tool_args(name, args, user_text)
                result = tools_mod.call_tool(name, args)
                tool_trace.append({
                    "iteration": iterations,
                    "tool": name,
                    "arguments": args,
                    "result": result,
                })
                tool_msg = {
                    "role": "tool",
                    "tool_name": name,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                if tc_id:
                    tool_msg["tool_call_id"] = tc_id
                messages.append(tool_msg)

        # Loop budget exhausted
        return {
            "answer": (
                "(stopped after max iterations) "
                + ((messages[-1].get("content") or "")
                   if messages and messages[-1].get("role") == "assistant"
                   else "")
            ).strip(),
            "tool_trace": tool_trace,
            "iterations": iterations,
            "messages": messages,
            "error": "max_iterations_reached",
        }
