"""Function-calling wrappers for Lab06 moderation actions."""
from lab6.moderation import actions


def _schema(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


APPROVE_SCHEMA = _schema(
    "approve_content",
    "Approve and publish flagged content.",
    {
        "content_id": {"type": "string"},
        "moderator_id": {"type": "string"},
    },
    ["content_id"],
)
REJECT_SCHEMA = _schema(
    "reject_content",
    "Reject and remove content.",
    {
        "content_id": {"type": "string"},
        "reason": {"type": "string"},
        "moderator_id": {"type": "string"},
    },
    ["content_id", "reason"],
)
FLAG_SCHEMA = _schema(
    "flag_for_human_review",
    "Flag content for manual review by a human moderator.",
    {
        "content_id": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "reason": {"type": "string"},
    },
    ["content_id", "priority", "reason"],
)
SHADOW_BAN_SCHEMA = _schema(
    "shadow_ban_user",
    "Limit user visibility for a defined number of hours.",
    {
        "user_id": {"type": "string"},
        "duration_hours": {"type": "integer"},
        "reason": {"type": "string"},
    },
    ["user_id", "duration_hours", "reason"],
)
HISTORY_SCHEMA = _schema(
    "get_user_moderation_history",
    "Return a user's moderation history and risk score.",
    {"user_id": {"type": "string"}},
    ["user_id"],
)
SIMILAR_SCHEMA = _schema(
    "find_similar_violations",
    "Find similar previously moderated cases.",
    {
        "text": {"type": "string"},
        "limit": {"type": "integer"},
    },
    ["text"],
)
WATCHLIST_SCHEMA = _schema(
    "add_to_watchlist",
    "Add a user to the moderation watchlist.",
    {
        "user_id": {"type": "string"},
        "reason": {"type": "string"},
    },
    ["user_id", "reason"],
)


TOOL_SPECS = {
    "approve_content": (actions.approve_content, APPROVE_SCHEMA),
    "reject_content": (actions.reject_content, REJECT_SCHEMA),
    "flag_for_human_review": (
        actions.flag_for_human_review,
        FLAG_SCHEMA,
    ),
    "shadow_ban_user": (actions.shadow_ban_user, SHADOW_BAN_SCHEMA),
    "get_user_moderation_history": (
        actions.get_user_moderation_history,
        HISTORY_SCHEMA,
    ),
    "find_similar_violations": (
        actions.find_similar_violations,
        SIMILAR_SCHEMA,
    ),
    "add_to_watchlist": (actions.add_to_watchlist, WATCHLIST_SCHEMA),
}
