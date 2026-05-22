"""Multi-step tool-calling agent built on top of Ollama /api/chat."""
import json

from . import ollama_client
from . import tools as tools_mod
from .location_resolver import city_lookup_keys, fold_text as _fold
from .tool_planner import (
    ToolPlanner,
    already_called_equivalent,
    is_image_request,
    needs_average_weather_search,
)
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


def _tool_results(tool_trace, name):
    return [
        t.get("result") or {}
        for t in tool_trace
        if t.get("tool") == name and isinstance(t.get("result"), dict)
    ]


def _format_number(value):
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _format_weather_answer(user_text, tool_trace):
    weather = [
        r for r in _tool_results(tool_trace, "get_weather")
        if not r.get("error")
    ]
    if not weather:
        return None
    parts = []
    seen = set()
    for item in weather:
        result_key = _weather_result_key(item)
        if result_key in seen:
            continue
        seen.add(result_key)
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

    if needs_average_weather_search(user_text):
        searches = [
            r for r in _tool_results(tool_trace, "web_search")
            if not r.get("error")
        ]
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
                    "dla podanego kontekstu, więc nie porównuję tego na siłę."
                )
        else:
            parts.append(
                "Nie udało się pobrać wiarygodnego wyniku o typowej pogodzie, "
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
    searches = [
        r for r in _tool_results(tool_trace, "web_search")
        if not r.get("error")
    ]
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
    if not is_image_request(user_text):
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


def _format_calculator_answer(user_text, tool_trace):
    results = _tool_results(tool_trace, "calculator")
    if not results:
        return None
    result = results[-1]
    if result.get("error"):
        return f"Nie udało się obliczyć: {result.get('error')}"
    return f"Wynik: {_format_number(result.get('result'))}."


def _format_local_kb_answer(user_text, tool_trace):
    results = [
        r for r in _tool_results(tool_trace, "local_knowledge")
        if not r.get("error")
    ]
    if not results:
        return None
    result = results[-1]
    hits = result.get("hits") or []
    if not hits:
        return (
            "Nie znalazłem pasujących wpisów w lokalnej bazie dla: "
            f"{result.get('query')}."
        )

    lines = [
        f"Lokalna baza wiedzy: {len(hits)} wynik(ów) "
        f"dla `{result.get('query')}`:"
    ]
    for hit in hits[:5]:
        label = hit.get("label") or hit.get("source")
        details = (
            hit.get("description")
            or hit.get("snippet")
            or hit.get("text")
            or ""
        )
        lines.append(f"- {label}: {details[:220]}")
    return "\n".join(lines)


def _answer_from_tools(user_text, tool_trace):
    for formatter in (
        _format_image_answer,
        _format_calculator_answer,
        _format_local_kb_answer,
        _format_nlp_answer,
        _format_web_fact_answer,
        _format_datetime_answer,
        _format_weather_answer,
    ):
        answer = formatter(user_text, tool_trace)
        if answer:
            return answer
    return None


def _weather_result_key(result):
    lat = result.get("latitude")
    lon = result.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        return "geo", round(lat, 4), round(lon, 4)
    city_keys = sorted(city_lookup_keys(result.get("city") or ""))
    return "city", result.get("country") or "", city_keys[0] if city_keys else ""


def _format_nlp_answer(user_text, tool_trace):
    results = [
        r for r in _tool_results(tool_trace, "nlp_tools")
        if not r.get("error")
    ]
    if not results:
        return None
    result = results[-1]
    operation = result.get("operation")
    if operation == "translate":
        return (
            f"Tłumaczenie ({result.get('src')} -> {result.get('tgt')}): "
            f"{result.get('translation')}"
        )
    if operation == "summarize":
        return f"Podsumowanie:\n{result.get('summary')}"
    if operation == "extract_entities":
        entities = result.get("entities") or []
        if not entities:
            return "Nie znaleziono encji."
        lines = ["Encje:"]
        for entity in entities[:20]:
            lines.append(f"- {entity.get('text')} ({entity.get('label')})")
        return "\n".join(lines)
    if operation != "classify_sentiment":
        return None
    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)):
        conf_text = f"{confidence:.2f}"
    else:
        conf_text = str(confidence)
    return f"Sentyment: {result.get('label')} (pewność {conf_text})."


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
                 enabled_tools=None, system_prompt=None, planner=None):
        self.model = model or AGENT_MODEL
        self.max_iters = max_iters
        self.enabled_tools = enabled_tools  # None = all
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.planner = planner or ToolPlanner()

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
            if image_answer and not self.planner.plan_fallback_calls(
                user_text,
                tool_trace,
            ):
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
                missing_answer = self.planner.missing_required_info_answer(user_text)
                if missing_answer:
                    return {
                        "answer": missing_answer,
                        "tool_trace": tool_trace,
                        "iterations": iterations,
                        "messages": messages,
                    }

                fallback_calls = self.planner.plan_fallback_calls(
                    user_text,
                    tool_trace,
                )
                if fallback_calls:
                    for planned in fallback_calls:
                        _execute_tool(
                            messages,
                            tool_trace,
                            iterations,
                            planned.tool,
                            planned.arguments,
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
            executed_any = False
            for tc in tool_calls:
                name, args, tc_id = _normalise_tool_call(tc)
                args = self.planner.repair_tool_args(name, args, user_text)
                if already_called_equivalent(tool_trace, name, args):
                    continue
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
                executed_any = True

            if not executed_any:
                direct_answer = _answer_from_tools(user_text, tool_trace)
                if direct_answer:
                    return {
                        "answer": direct_answer,
                        "tool_trace": tool_trace,
                        "iterations": iterations,
                        "messages": messages,
                    }

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
