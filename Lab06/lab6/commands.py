"""Lab 6 moderation command handlers."""

import re

from lab6.moderation import actions as moderation_actions
from lab6.moderation import pipeline as moderation_pipeline
from lab6.moderation import storage as moderation_storage
from utils import parse_params, truncate


HELP_SECTION = (
    "--- Lab 6 (Content Moderation) ---\n"
    "/moderate \"text\" [user_id=...] [username=...]\n"
    "/mod_policy_check \"text\" - dry-run policy check\n"
    "/mod_status <content_id>\n"
    "/mod_history <user_id>\n"
    "/mod_analytics\n"
    "/mod_add_feedback <content_id> \"comment\" \"APPROVE|REJECT|FLAG_FOR_REVIEW\"\n"
    "/mod_watchlist\n"
    "/mod_train_on_feedback\n"
    "/mod_help\n"
)

_VALID_OVERRIDES = {"APPROVE", "REJECT", "FLAG_FOR_REVIEW"}


def _send_long(bot, chat_id, text, chunk=3500):
    """Telegram has a ~4096 char limit per message."""
    if not text:
        bot.send_message(chat_id, "(empty)")
        return

    for i in range(0, len(text), chunk):
        bot.send_message(chat_id, text[i:i + chunk])


def _command_rest(message):
    parts = (message.text or "").split(maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def _extract_text_param(rest):
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
    similar_cases = result.get("similar_cases") or []
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

    if similar_cases:
        case = similar_cases[0]
        lines.append(
            "Similar case: "
            f"{case.get('content_id')} "
            f"({case.get('action')}, score={case.get('score')})"
        )

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
        for name, count in sorted(
            consensus.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
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
    bot.reply_to(message, HELP_SECTION + "\nUse /help for full command list.")


def _handle_moderate(bot, message):
    rest = _command_rest(message)
    params = parse_params(rest)
    text = _extract_command_text(rest)
    if not text:
        bot.reply_to(message, 'Usage: /moderate "text" [user_id=...]')
        return

    result = moderation_pipeline.moderate_content(
        text=text,
        user_id=params.get("user_id") or str(message.from_user.id),
        username=params.get("username") or (message.from_user.username or ""),
        content_id=params.get("content_id"),
        moderator_id=str(message.from_user.id),
    )
    _send_long(bot, message.chat.id, _format_moderation_result(result))


def _handle_mod_policy_check(bot, message):
    rest = _command_rest(message)
    text = _extract_command_text(rest)
    if not text:
        bot.reply_to(message, 'Usage: /mod_policy_check "text"')
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
        override = (
            params.get("decision")
            or params.get("override")
            or params.get("correct_decision")
            or params.get("correct")
            or params.get("poprawna_decyzja")
            or ""
        )
    if not override:
        bot.reply_to(message, "Missing moderator decision.")
        return

    override_norm = override.strip().upper()
    if override_norm not in _VALID_OVERRIDES:
        bot.reply_to(
            message,
            "Invalid decision. Use APPROVE, REJECT or FLAG_FOR_REVIEW.",
        )
        return

    result = moderation_actions.add_feedback(content_id, override_norm, comment)
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
        lines.append(f"- {row.get('user_id')}: {truncate(row.get('reason'), 120)}")
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


def register_handlers(bot):
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


__all__ = ["HELP_SECTION", "register_handlers"]
