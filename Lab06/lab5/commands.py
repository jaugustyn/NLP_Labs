"""Lab 5 command handlers."""

import json
import os
import re
import tempfile

from lab5 import ollama_client
from lab5 import session_store
from lab5 import tools as tools_mod
from lab5.agent import Agent
from utils import log_error, parse_params, truncate


HELP_SECTION = (
    "--- Lab 5 (Tool Calling Agent) ---\n"
    "/agent text=\"...\"   (or send a photo + caption)\n"
    "  Single-shot agent with tool calling.\n"
    "/chat                - toggle conversational mode (history kept)\n"
    "/chat_reset          - clear chat history for this user\n"
    "/agent_history [n=5] - last agent runs with tool trace\n"
    "/tools               - list registered tools\n"
    "Direct tool invocations (for testing):\n"
    "  /tool_calc expression=\"2+2*sqrt(16)\"\n"
    "  /tool_weather city=\"Warsaw\"\n"
    "  /tool_search query=\"OpenAI\" [language=en|pl]\n"
    "  /tool_local_kb query=\"Paris\"\n"
    "  /tool_datetime [city=Tokyo|tz=Europe/Warsaw]\n"
    "  /tool_nlp operation=<translate|summarize|extract_entities|"
    "classify_sentiment> text=\"...\" [target_language=...] [language=...]\n"
    "  /tool_vision  - send a photo with this caption\n"
)

_chat_mode = {}


def _format_trace(trace, max_arg_len=200):
    """Format agent tool trace for Telegram output."""
    if not trace:
        return "(no tool calls)"

    out = []
    for step in trace:
        args_repr = json.dumps(step.get("arguments", {}), ensure_ascii=False)
        line = (
            f"  [{step.get('iteration')}] "
            f"{step.get('tool')}({truncate(args_repr, max_arg_len)})"
        )
        result = step.get("result")
        if result is not None:
            result_repr = json.dumps(result, ensure_ascii=False)
            line += f" -> {truncate(result_repr, 220)}"
        out.append(line)
    return "\n".join(out)


def _send_long(bot, chat_id, text, chunk=3500):
    """Telegram has a ~4096 char limit per message."""
    if not text:
        bot.send_message(chat_id, "(empty)")
        return

    for i in range(0, len(text), chunk):
        bot.send_message(chat_id, text[i:i + chunk])


def _download_photo(bot, message):
    """Download the highest-resolution Telegram photo to a temp file."""
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
    """Return True for photo captions like '/agent ...'."""
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
            "To answer any question about that picture you MUST call "
            f"the analyze_image tool with image_path='{image_path}'. "
            "Do not guess what the image contains."
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

    body = f"{answer}\n\n- iterations: {iters}, tool calls: {len(trace)}"
    if trace:
        body += "\n" + _format_trace(trace)
    return body


def _handle_agent(bot, message):
    raw = _message_text(message)
    parts = raw.split(maxsplit=1)
    rest = parts[1] if len(parts) > 1 else ""
    user_text = _extract_agent_text(rest)
    if not user_text and not message.photo:
        bot.reply_to(
            message,
            "Usage: /agent text=\"...\"  (you can also attach a photo).",
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
        bot,
        message,
        user_text or "Describe the image.",
        history=history,
    )
    if result is None:
        return

    answer = result.get("answer") or "(no answer)"
    session_store.append_turn(chat_id, user_text, answer)
    session_store.append_run(chat_id, user_text, result)
    _send_long(bot, chat_id, _format_agent_reply(result))


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
    for run in runs:
        tools_used = ", ".join(t.get("tool", "?") for t in run.get("tool_trace", []))
        lines.append(f"[{run.get('timestamp')}]")
        lines.append(f"  Q: {truncate(run.get('user', ''), 120)}")
        lines.append(f"  A: {truncate(run.get('answer', ''), 200)}")
        lines.append(f"  tools: {tools_used or '(none)'}")
        for tool_call in (run.get("tool_trace") or [])[:3]:
            result = tool_call.get("result")
            if result is not None:
                result_repr = json.dumps(result, ensure_ascii=False)
                lines.append(
                    f"    {tool_call.get('tool')}: "
                    f"{truncate(result_repr, 160)}"
                )
        lines.append("")

    _send_long(bot, message.chat.id, "\n".join(lines))


def _handle_tools_list(bot, message):
    lines = ["Registered tools:", ""]
    for name, (_, schema) in sorted(tools_mod.TOOL_REGISTRY.items()):
        desc = schema.get("function", {}).get("description", "")
        lines.append(f"- {name} - {truncate(desc, 140)}")
    bot.reply_to(message, "\n".join(lines))


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
    query = params.get("query") or params.get("q")
    if not query:
        bot.reply_to(message, 'Usage: /tool_search query="..." [language=en|pl]')
        return

    args = {"query": query}
    if params.get("language"):
        args["language"] = params["language"]
    _direct_call(bot, message, "web_search", args)


def _handle_tool_local_kb(bot, message):
    parts = (message.text or "").split(maxsplit=1)
    params = parse_params(parts[1] if len(parts) > 1 else "")
    query = params.get("query") or params.get("q")
    if not query:
        bot.reply_to(message, 'Usage: /tool_local_kb query="..."')
        return
    _direct_call(bot, message, "local_knowledge", {"query": query})


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
    operation = params.get("operation") or params.get("op")
    text = params.get("text")
    if not operation or not text:
        bot.reply_to(
            message,
            'Usage: /tool_nlp operation=<translate|summarize|extract_entities|'
            'classify_sentiment> text="..." [target_language=pl] [language=auto]',
        )
        return

    args = {"operation": operation, "text": text}
    for key in ("language", "target_language", "summary_type", "length"):
        if params.get(key):
            args[key] = params[key]
    _direct_call(bot, message, "nlp_tools", args)


def _handle_tool_vision(bot, message):
    image_path = _download_photo(bot, message)
    if not image_path:
        bot.reply_to(
            message,
            "Send a photo with caption /tool_vision to use this tool.",
        )
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


def register_handlers(bot):
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

    @bot.message_handler(
        func=lambda m: (_chat_mode.get(m.chat.id, False) and not _is_command_message(m)),
        content_types=["text", "photo"],
    )
    def cmd_chat_message(message):
        _handle_chat_message(bot, message)


__all__ = ["HELP_SECTION", "register_handlers"]
