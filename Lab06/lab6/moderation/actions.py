"""Moderation action tools used by Lab06 function calling."""
from . import storage


_VALID_FEEDBACK_DECISIONS = {"APPROVE", "REJECT", "FLAG_FOR_REVIEW"}


def approve_content(content_id, moderator_id="bot"):
    storage.append_action(
        "APPROVE",
        content_id=content_id,
        moderator_id=moderator_id,
    )
    return {
        "content_id": content_id,
        "action": "APPROVE",
        "status": "approved",
    }


def reject_content(content_id, reason, moderator_id="bot"):
    storage.append_action(
        "REJECT",
        content_id=content_id,
        reason=reason,
        moderator_id=moderator_id,
    )
    return {
        "content_id": content_id,
        "action": "REJECT",
        "status": "rejected",
        "reason": reason,
    }


def flag_for_human_review(content_id, priority, reason):
    storage.append_action(
        "FLAG_FOR_REVIEW",
        content_id=content_id,
        priority=priority,
        reason=reason,
    )
    return {
        "content_id": content_id,
        "action": "FLAG_FOR_REVIEW",
        "priority": priority,
        "reason": reason,
    }


def shadow_ban_user(user_id, duration_hours, reason):
    try:
        duration_hours = int(duration_hours)
    except (TypeError, ValueError):
        duration_hours = 24
    duration_hours = max(1, min(duration_hours, 24 * 30))
    storage.append_action(
        "SHADOW_BAN",
        user_id=user_id,
        duration_hours=duration_hours,
        reason=reason,
    )
    return {
        "user_id": user_id,
        "action": "SHADOW_BAN",
        "duration_hours": duration_hours,
        "reason": reason,
    }


def get_user_moderation_history(user_id):
    row = storage.user_history(user_id)
    return {
        "user_id": row.get("user_id"),
        "violations_count": int(row.get("total_violations") or 0),
        "last_violation": row.get("last_violation_date") or None,
        "categories": [
            c for c in (row.get("categories") or "").split(";") if c
        ],
        "risk_score": float(row.get("risk_score") or 0.0),
        "is_repeat_offender": row.get("is_repeat_offender") == "True",
    }


def find_similar_violations(text, limit=5):
    return storage.find_similar(text, limit=limit)


def add_to_watchlist(user_id, reason):
    storage.append_watchlist(user_id, reason)
    return {
        "user_id": user_id,
        "action": "ADD_TO_WATCHLIST",
        "reason": reason,
        "status": "active",
    }


def add_feedback(content_id, moderator_override, comment=""):
    decision = (moderator_override or "").upper()
    if decision not in _VALID_FEEDBACK_DECISIONS:
        return {
            "error": (
                "moderator_override must be APPROVE, REJECT or "
                "FLAG_FOR_REVIEW"
            )
        }
    row = storage.append_feedback(content_id, decision, comment)
    return {
        "content_id": content_id,
        "original_bot_decision": row.get("original_bot_decision"),
        "moderator_override": row.get("moderator_override"),
        "comment": comment,
        "status": "recorded",
    }


def train_on_feedback():
    # This lab runs locally without fine-tuning infrastructure. The
    # feedback file is still produced, so it can be used as training data.
    feedback_count = len(storage._read_rows(storage.FEEDBACK_LOG))
    train_count = len(storage._read_rows(storage.TRAIN_DATA))
    return {
        "status": "prepared",
        "feedback_examples": feedback_count,
        "training_examples": train_count,
        "train_data_path": storage.TRAIN_DATA,
        "note": (
            "Feedback saved to train_data.csv; local fine-tuning is "
            "intentionally not run automatically."
        ),
    }
