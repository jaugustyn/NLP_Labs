"""Lab 6 command handlers.

Includes Lab 5 agent/tool calling plus the Lab 6 content moderation
pipeline and moderator commands.
"""
import json
import os
import re
import tempfile

from utils import parse_params, log_error, truncate

from lab5 import ollama_client
from lab5 import session_store
from lab5 import tools as tools_mod
from lab5.agent import Agent
from moderation import actions as moderation_actions
from moderation import pipeline as moderation_pipeline
from moderation import storage as moderation_storage

from lab4.commands import register_handlers as register_lab1234


HELP_TEXT = (
    "NLP Bot — Lab 1 + Lab 2 + Lab 3 + Lab 4 + Lab 5 + Lab 6\n\n"
    "--- Lab 6 (Content Moderation) ---\n"
    "/moderate \"tekst\" [user_id=...] [username=...]\n"
    "/mod_policy_check \"tekst\" — analiza bez zapisu akcji\n"
    "/mod_status <content_id>\n"
    "/mod_history <user_id>\n"
    "/mod_analytics\n"
    "/mod_add_feedback <content_id> \"komentarz\" \"poprawna_decyzja\"\n"
    "/mod_watchlist\n"
    "/mod_train_on_feedback\n"
    "/mod_help\n\n"
    "--- Lab 5 (Tool Calling Agent) ---\n"
    "/agent text=\"...\"   (or send a photo + caption)\n"
    "  Single-shot agent with tool calling.\n"
    "/chat                — toggle conversational mode (history kept)\n"
    "/chat_reset          — clear chat history for this user\n"
    "/agent_history [n=5] — last agent runs with tool trace\n"
    "/tools               — list registered tools\n"
    "Direct tool invocations (for testing):\n"
    "  /tool_calc expression=\"2+2*sqrt(16)\"\n"
    "  /tool_weather city=\"Warsaw\"\n"
    "  /tool_search query=\"OpenAI\" [language=en|pl]\n"
    "  /tool_local_kb query=\"Paris\"\n"
    "  /tool_datetime [city=Tokyo|tz=Europe/Warsaw]\n"
    "  /tool_nlp operation=<translate|summarize|extract_entities|"
    "classify_sentiment> text=\"...\" [target_language=...] [language=...]\n"
    "  /tool_vision  — send a photo with this caption\n\n"
    "--- Lab 1 ---\n"
    "/task <name> \"text\" \"class\"\n"
    "/full_pipeline \"text\" \"class\"\n"
    "/classifier \"text\"\n"
    "/stats\n\n"
    "--- Lab 2 ---\n"
    "/classify dataset=<name> method=<model> ...\n\n"
    "--- Lab 3 ---\n"
    "/sentiment method=<m> text=\"...\"\n"
    "/train model=<m> dataset=<d>\n"
    "/compare dataset=<d> methods=<m1,m2>\n"
    "/add_sentiment \"text\" \"label\"\n"
    "/models\n\n"
    "--- Lab 4 ---\n"
    "/language_detect /ner /nel /ned /analyze_entities\n"
    "/translate /summarize /knowledge_graph\n"
)


# In-memory toggle: chat_id -> bool (conversational mode on/off)
_chat_mode = {}


# =====================================================================
#  Helpers
# =====================================================================

def _format_trace(trace, max_arg_len=200):
    if not trace:
        return "(no tool calls)"
    out = []
    for step in trace:
        args_repr = json.dumps(step.get("arguments", {}), ensure_ascii=False)
        out.append(
            f"  [{step.get('iteration')}] "
            f"{step.get('tool')}({truncate(args_repr, max_arg_len)})"
        )
    return "\n".join(out)


def _send_long(bot, chat_id, text, chunk=3500):
    """Telegram has a ~4096 char limit per message."""
    if not text:
        bot.send_message(chat_id, "(empty)")
        return
    for i in range(0, len(text), chunk):
        bot.send_message(chat_id, text[i:i + chunk])


def _download_photo(bot, message):
    """Download the highest-res photo from a Telegram message; return
    local file path, or None if no photo."""
    if not message.photo:
        return None
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    data = bot.download_file(file_info.file_path)
    fd, path = tempfile.mkstemp(prefix="tg_", suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _message_text(message):
    return (message.text or message.caption or "").strip()


def _caption_command(message, command):
    """Return True for photo captions like '/agent ...'.

    pyTelegramBotAPI's built-in `commands=` filter only checks text
    messages, so photo captions need an explicit predicate.
    """
    raw = _message_text(message)
    if not raw.startswith("/"):
        return False
    token = raw.split(maxsplit=1)[0][1:]
    token = token.split("@", 1)[0]
    return token == command


def _is_command_message(message):
    return _message_text(message).startswith("/")


def _extract_text_param(rest):
    """Extract text=... robustly, including an unclosed quote."""
    match = re.search(r"(?<!\w)text\s*=", rest or "", flags=re.IGNORECASE)
    if not match:
        return None
    value = rest[match.end():].strip()
    if not value:
        return ""
    if value[0] in ("\"", "'"):
        quote = value[0]
        end = value.find(quote, 1)
        if end >= 0:
            return value[1:end].strip()
        return value[1:].strip()
    next_param = re.search(r"\s+\w+\s*=", value)
    if next_param:
        return value[:next_param.start()].strip()
    return value.strip()


def _extract_agent_text(rest):
    text = _extract_text_param(rest)
    if text is not None:
        return text
    params = parse_params(rest)
    return params.get("text") or rest.strip()


def _run_agent(bot, message, user_text, history=None, extra_context=None):
    image_path = _download_photo(bot, message)
    images = [image_path] if image_path else None
    if image_path and extra_context is None:
        extra_context = (
            f"The user attached an image. Its local path is: {image_path}. "
            f"To answer any question about that picture you MUST call "
            f"the analyze_image tool with image_path='{image_path}'. "
            f"Do not guess what the image contains."
        )
    try:
        agent = Agent()
        result = agent.run(
            user_text=user_text,
            images=images,
            history=history,
            extra_context=extra_context,
        )
    except Exception as e:
        log_error("agent.run", e)
        bot.reply_to(message, f"Agent error: {e}")
        return None
    finally:
        if image_path:
            try:
                os.unlink(image_path)
            except Exception:
                pass
    return result


def _format_agent_reply(result):
    answer = result.get("answer") or "(no answer)"
    iters = result.get("iterations", "?")
    trace = result.get("tool_trace") or []
    body = f"{answer}\n\n— iterations: {iters}, tool calls: {len(trace)}"
    if trace:
        body += "\n" + _format_trace(trace)
    return body


# =====================================================================
#  Lab 6 moderation formatting / parsing
# =====================================================================

def _command_rest(message):
    parts = (_message_text(message) or "").split(maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def _extract_command_text(rest):
    text = _extract_text_param(rest)
    if text is not None:
        return text
    quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', rest or "")
    if quoted:
        first = quoted[0]
        return (first[0] or first[1] or "").strip()
    params = parse_params(rest)
    return params.get("text") or rest.strip()


def _format_moderation_result(result, dry_run=False):
    if result.get("error"):
        return f"Moderation error: {result['error']}"
    prefix = "[POLICY CHECK]" if dry_run else "[MODERATION]"
    pii = result.get("pii", {})
    bielik = result.get("bielik", {})
    qwen = result.get("qwen", {})
    sentiment = result.get("sentiment", {})
    entities = result.get("entities", {})
    lines = [
        f"{prefix} {result.get('content_id')}",
        f"Action: {result.get('action')}",
        f"Reason: {result.get('reason')}",
        f"Risk: {result.get('risk_level')} | consensus: {result.get('consensus')}",
        (
            "Bielik Guard: "
            f"{bielik.get('label')} ({bielik.get('score')}, "
            f"{bielik.get('severity')})"
        ),
        (
            "Qwen Guard: "
            f"{qwen.get('risk_level')} ({qwen.get('confidence')}) -> "
            f"{qwen.get('recommended_action')}"
        ),
        (
            "PII: "
            f"{'yes' if pii.get('has_pii') else 'no'} "
            f"({len(pii.get('entities') or [])} entities)"
        ),
        (
            "Sentiment: "
            f"{sentiment.get('sentiment')} "
            f"({sentiment.get('confidence')}, {sentiment.get('emotion')})"
        ),
    ]
    red_flags = []
    if entities.get("emails"):
        red_flags.append(f"emails={len(entities['emails'])}")
    if entities.get("phone_numbers"):
        red_flags.append(f"phones={len(entities['phone_numbers'])}")
    if entities.get("urls"):
        red_flags.append(f"urls={len(entities['urls'])}")
    if red_flags:
        lines.append("Entities: " + ", ".join(red_flags))
    return "\n".join(lines)


def _format_analytics(data):
    total = data.get("total") or 0
    actions = data.get("actions") or {}
    cats = data.get("top_categories") or []
    consensus = data.get("consensus") or {}
    offenders = data.get("repeat_offenders") or []
    lines = [
        "MODERATION ANALYTICS",
        f"Total posts reviewed: {total}",
        f"Approved: {actions.get('APPROVE', 0)}",
        f"Rejected: {actions.get('REJECT', 0)}",
        f"Flagged for review: {actions.get('FLAG_FOR_REVIEW', 0)}",
        f"Human overrides: {data.get('human_overrides', 0)}",
        "",
        "Top violations:",
    ]
    if cats:
        lines.extend(f"- {name}: {count}" for name, count in cats)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Model consensus:")
    if consensus:
        for name, count in sorted(consensus.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Repeat offenders:")
    if offenders:
        for row in offenders:
            lines.append(
                f"- {row.get('user_id')}: "
                f"{row.get('total_violations')} violations, "
                f"risk={row.get('risk_score')}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _handle_mod_help(bot, message):
    bot.reply_to(message, HELP_TEXT)


def _handle_moderate(bot, message):
    rest = _command_rest(message)
    params = parse_params(rest)
    text = _extract_command_text(rest)
    if not text:
        bot.reply_to(message, 'Usage: /moderate "tekst" [user_id=...]')
        return
    result = moderation_pipeline.moderate_content(
        text=text,
        user_id=params.get("user_id") or str(message.from_user.id),
        username=params.get("username") or (message.from_user.username or ""),
        content_id=params.get("content_id"),
        moderator_id=str(message.from_user.id),
    )
    _send_long(message.bot if hasattr(message, "bot") else bot,
               message.chat.id, _format_moderation_result(result))


def _handle_mod_policy_check(bot, message):
    rest = _command_rest(message)
    text = _extract_command_text(rest)
    if not text:
        bot.reply_to(message, 'Usage: /mod_policy_check "tekst"')
        return
    result = moderation_pipeline.policy_check(
        text=text,
        user_id=str(message.from_user.id),
    )
    _send_long(bot, message.chat.id, _format_moderation_result(result, dry_run=True))


def _handle_mod_status(bot, message):
    rest = _command_rest(message).strip()
    content_id = rest.split()[0] if rest else ""
    if not content_id:
        bot.reply_to(message, "Usage: /mod_status <content_id>")
        return
    row = moderation_storage.get_moderation(content_id)
    if not row:
        bot.reply_to(message, f"No moderation record for {content_id}.")
        return
    lines = [
        f"Status for {content_id}",
        f"Action: {row.get('action')}",
        f"Reason: {row.get('reason')}",
        f"Risk: {row.get('risk_level')}",
        f"User: {row.get('user_id')}",
        f"Timestamp: {row.get('timestamp')}",
    ]
    bot.reply_to(message, "\n".join(lines))


def _handle_mod_history(bot, message):
    rest = _command_rest(message).strip()
    user_id = rest.split()[0] if rest else str(message.from_user.id)
    history = moderation_actions.get_user_moderation_history(user_id)
    lines = [
        f"Moderation history for {user_id}",
        f"Violations: {history['violations_count']}",
        f"Last violation: {history['last_violation'] or '-'}",
        f"Categories: {', '.join(history['categories']) or '-'}",
        f"Risk score: {history['risk_score']}",
        f"Repeat offender: {history['is_repeat_offender']}",
    ]
    bot.reply_to(message, "\n".join(lines))


def _handle_mod_analytics(bot, message):
    bot.reply_to(message, _format_analytics(moderation_storage.analytics()))


def _handle_mod_add_feedback(bot, message):
    rest = _command_rest(message)
    parts = rest.split(maxsplit=1)
    if not parts:
        bot.reply_to(
            message,
            'Usage: /mod_add_feedback <content_id> "comment" "APPROVE|REJECT|FLAG_FOR_REVIEW"',
        )
        return
    content_id = parts[0]
    quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', parts[1] if len(parts) > 1 else "")
    values = [(a or b or "").strip() for a, b in quoted]
    comment = values[0] if values else ""
    override = values[1] if len(values) > 1 else ""
    if not override:
        params = parse_params(rest)
        override = params.get("decision") or params.get("override") or ""
    if not override:
        bot.reply_to(message, "Missing moderator decision.")
        return
    result = moderation_actions.add_feedback(content_id, override, comment)
    bot.reply_to(
        message,
        f"Feedback recorded for {content_id}: {result['moderator_override']}",
    )


def _handle_mod_watchlist(bot, message):
    rows = moderation_storage.list_watchlist()
    if not rows:
        bot.reply_to(message, "Watchlist is empty.")
        return
    lines = ["Watchlist:"]
    for row in rows[:20]:
        lines.append(
            f"- {row.get('user_id')}: {truncate(row.get('reason'), 120)}"
        )
    bot.reply_to(message, "\n".join(lines))


def _handle_mod_train_on_feedback(bot, message):
    result = moderation_actions.train_on_feedback()
    bot.reply_to(
        message,
        (
            f"Feedback examples: {result['feedback_examples']}\n"
            f"Training examples: {result['training_examples']}\n"
            f"{result['note']}"
        ),
    )


# =====================================================================
#  /agent — single-shot
# =====================================================================

def _handle_agent(bot, message):
    raw = _message_text(message)
    # Strip the command itself
    parts = raw.split(maxsplit=1)
    rest = parts[1] if len(parts) > 1 else ""
    user_text = _extract_agent_text(rest)
    if not user_text and not message.photo:
        bot.reply_to(
            message,
            "Usage: /agent text=\"...\"  (you can also attach a photo)."
        )
        return

    if not ollama_client.is_available():
        bot.reply_to(message, "Ollama is not reachable on localhost:11434.")
        return

    bot.send_chat_action(message.chat.id, "typing")
    result = _run_agent(bot, message, user_text or "Describe the image.")
    if result is None:
        return
    session_store.append_run(message.chat.id, user_text, result)
    _send_long(bot, message.chat.id, _format_agent_reply(result))


# =====================================================================
#  /chat — conversational mode
# =====================================================================

def _handle_chat_toggle(bot, message):
    chat_id = message.chat.id
    new_state = not _chat_mode.get(chat_id, False)
    _chat_mode[chat_id] = new_state
    if new_state:
        bot.reply_to(
            message,
            "Chat mode ON. Every message you send will go through the "
            "agent with conversation history. Send /chat again to turn "
            "it off, or /chat_reset to clear history.",
        )
    else:
        bot.reply_to(message, "Chat mode OFF.")


def _handle_chat_reset(bot, message):
    ok = session_store.reset_session(message.chat.id)
    bot.reply_to(message, "History cleared." if ok else "Failed to reset.")


def _handle_chat_message(bot, message):
    """Triggered for any non-command text/photo while chat mode is ON."""
    chat_id = message.chat.id
    user_text = (message.text or message.caption or "").strip()
    if not user_text and not message.photo:
        return
    if not ollama_client.is_available():
        bot.reply_to(message, "Ollama is not reachable on localhost:11434.")
        return

    bot.send_chat_action(chat_id, "typing")
    history = session_store.get_history(chat_id)
    result = _run_agent(
        bot, message, user_text or "Describe the image.", history=history,
    )
    if result is None:
        return
    answer = result.get("answer") or "(no answer)"
    session_store.append_turn(chat_id, user_text, answer)
    session_store.append_run(chat_id, user_text, result)
    _send_long(bot, chat_id, _format_agent_reply(result))


# =====================================================================
#  /agent_history
# =====================================================================

def _handle_agent_history(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    try:
        n = int(params.get("n", "5"))
    except ValueError:
        n = 5
    runs = session_store.list_runs(message.chat.id, n=n)
    if not runs:
        bot.reply_to(message, "No agent runs recorded yet.")
        return
    lines = [f"Last {len(runs)} run(s):", ""]
    for r in runs:
        tools_used = ", ".join(t.get("tool", "?") for t in r.get("tool_trace", []))
        lines.append(f"[{r.get('timestamp')}]")
        lines.append(f"  Q: {truncate(r.get('user', ''), 120)}")
        lines.append(f"  A: {truncate(r.get('answer', ''), 200)}")
        lines.append(f"  tools: {tools_used or '(none)'}")
        lines.append("")
    _send_long(bot, message.chat.id, "\n".join(lines))


# =====================================================================
#  /tools — list registered tools
# =====================================================================

def _handle_tools_list(bot, message):
    lines = ["Registered tools:", ""]
    for name, (_, schema) in sorted(tools_mod.TOOL_REGISTRY.items()):
        desc = schema.get("function", {}).get("description", "")
        lines.append(f"- {name} — {truncate(desc, 140)}")
    bot.reply_to(message, "\n".join(lines))


# =====================================================================
#  /tool_<name> direct invocations
# =====================================================================

def _direct_call(bot, message, tool_name, args):
    bot.send_chat_action(message.chat.id, "typing")
    result = tools_mod.call_tool(tool_name, args)
    pretty = json.dumps(result, ensure_ascii=False, indent=2)
    _send_long(bot, message.chat.id, f"{tool_name}:\n{pretty}")


def _handle_tool_calc(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    expr = params.get("expression") or params.get("expr")
    if not expr:
        bot.reply_to(message, 'Usage: /tool_calc expression="2+2*sqrt(16)"')
        return
    _direct_call(bot, message, "calculator", {"expression": expr})


def _handle_tool_weather(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    city = params.get("city")
    if not city:
        bot.reply_to(message, 'Usage: /tool_weather city="Warsaw"')
        return
    _direct_call(bot, message, "get_weather", {"city": city})


def _handle_tool_search(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    q = params.get("query") or params.get("q")
    if not q:
        bot.reply_to(message, 'Usage: /tool_search query="..." [language=en|pl]')
        return
    args = {"query": q}
    if params.get("language"):
        args["language"] = params["language"]
    _direct_call(bot, message, "web_search", args)


def _handle_tool_local_kb(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    q = params.get("query") or params.get("q")
    if not q:
        bot.reply_to(message, 'Usage: /tool_local_kb query="..."')
        return
    _direct_call(bot, message, "local_knowledge", {"query": q})


def _handle_tool_datetime(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    args = {}
    if params.get("city"):
        args["city"] = params["city"]
    if params.get("tz"):
        args["tz"] = params["tz"]
    _direct_call(bot, message, "datetime_now", args)


def _handle_tool_nlp(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    op = params.get("operation") or params.get("op")
    text = params.get("text")
    if not op or not text:
        bot.reply_to(
            message,
            'Usage: /tool_nlp operation=<translate|summarize|'
            'extract_entities|classify_sentiment> text="..." '
            '[target_language=pl] [language=auto]',
        )
        return
    args = {"operation": op, "text": text}
    for k in ("language", "target_language", "summary_type", "length"):
        if params.get(k):
            args[k] = params[k]
    _direct_call(bot, message, "nlp_tools", args)


def _handle_tool_vision(bot, message):
    image_path = _download_photo(bot, message)
    if not image_path:
        bot.reply_to(message, "Send a photo with caption /tool_vision to use this tool.")
        return
    parts = (message.caption or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    prompt = params.get("prompt")
    args = {"image_path": image_path}
    if prompt:
        args["prompt"] = prompt
    try:
        _direct_call(bot, message, "analyze_image", args)
    finally:
        try:
            os.unlink(image_path)
        except Exception:
            pass


# =====================================================================
#  Registration
# =====================================================================

def register_handlers(bot):
    @bot.message_handler(commands=["start", "help"])
    def cmd_help(message):
        bot.reply_to(message, HELP_TEXT)

    @bot.message_handler(commands=["agent"])
    def cmd_agent(message):
        _handle_agent(bot, message)

    @bot.message_handler(
        func=lambda m: _caption_command(m, "agent"),
        content_types=["photo"],
    )
    def cmd_agent_photo(message):
        _handle_agent(bot, message)

    @bot.message_handler(commands=["chat"])
    def cmd_chat(message):
        _handle_chat_toggle(bot, message)

    @bot.message_handler(commands=["chat_reset"])
    def cmd_chat_reset(message):
        _handle_chat_reset(bot, message)

    @bot.message_handler(commands=["agent_history"])
    def cmd_agent_history(message):
        _handle_agent_history(bot, message)

    @bot.message_handler(commands=["tools"])
    def cmd_tools(message):
        _handle_tools_list(bot, message)

    @bot.message_handler(commands=["moderate"])
    def cmd_moderate(message):
        _handle_moderate(bot, message)

    @bot.message_handler(commands=["mod_policy_check"])
    def cmd_mod_policy_check(message):
        _handle_mod_policy_check(bot, message)

    @bot.message_handler(commands=["mod_status"])
    def cmd_mod_status(message):
        _handle_mod_status(bot, message)

    @bot.message_handler(commands=["mod_history"])
    def cmd_mod_history(message):
        _handle_mod_history(bot, message)

    @bot.message_handler(commands=["mod_analytics"])
    def cmd_mod_analytics(message):
        _handle_mod_analytics(bot, message)

    @bot.message_handler(commands=["mod_add_feedback"])
    def cmd_mod_add_feedback(message):
        _handle_mod_add_feedback(bot, message)

    @bot.message_handler(commands=["mod_watchlist"])
    def cmd_mod_watchlist(message):
        _handle_mod_watchlist(bot, message)

    @bot.message_handler(commands=["mod_train_on_feedback"])
    def cmd_mod_train_on_feedback(message):
        _handle_mod_train_on_feedback(bot, message)

    @bot.message_handler(commands=["mod_help"])
    def cmd_mod_help(message):
        _handle_mod_help(bot, message)

    @bot.message_handler(commands=["tool_calc"])
    def cmd_tool_calc(message):
        _handle_tool_calc(bot, message)

    @bot.message_handler(commands=["tool_weather"])
    def cmd_tool_weather(message):
        _handle_tool_weather(bot, message)

    @bot.message_handler(commands=["tool_search"])
    def cmd_tool_search(message):
        _handle_tool_search(bot, message)

    @bot.message_handler(commands=["tool_local_kb"])
    def cmd_tool_local_kb(message):
        _handle_tool_local_kb(bot, message)

    @bot.message_handler(commands=["tool_datetime"])
    def cmd_tool_datetime(message):
        _handle_tool_datetime(bot, message)

    @bot.message_handler(commands=["tool_nlp"])
    def cmd_tool_nlp(message):
        _handle_tool_nlp(bot, message)

    @bot.message_handler(commands=["tool_vision"])
    def cmd_tool_vision_usage(message):
        _handle_tool_vision(bot, message)

    @bot.message_handler(
        func=lambda m: _caption_command(m, "tool_vision"),
        content_types=["photo"],
    )
    def cmd_tool_vision(message):
        _handle_tool_vision(bot, message)

    # Catch-all for chat mode (only fires when chat mode is ON for that
    # chat_id, and the message is NOT a command).
    @bot.message_handler(
        func=lambda m: (
            _chat_mode.get(m.chat.id, False)
            and not _is_command_message(m)
        ),
        content_types=["text", "photo"],
    )
    def cmd_chat_message(message):
        _handle_chat_message(bot, message)

    # Lab 1-4 handlers (registered last; their /help/start is shadowed
    # by ours above, registered first).
    register_lab1234(bot)
