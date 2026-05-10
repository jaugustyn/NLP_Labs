"""Multi-stage moderation pipeline for Lab06."""
import uuid

from . import actions, models, storage


def _risk_priority(risk_level):
    return {
        "safe": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "critical": "critical",
    }.get(risk_level, "medium")


def _vote_from_bielik(result):
    label = result.get("label")
    severity = result.get("severity")
    score = float(result.get("score") or 0)
    if label == "clean":
        return "approve"
    if score >= 0.8 or severity in ("high", "critical"):
        return "reject"
    return "review"


def _decide(pii, bielik, qwen, history):
    categories = sorted(set(
        [c for c in bielik.get("categories", []) if c != "clean"]
        + [c for c in qwen.get("categories", []) if c != "clean"]
    ))
    if pii.get("has_pii"):
        return {
            "action": "REJECT",
            "reason": "personally_identifiable_information",
            "risk_level": "critical",
            "categories": ["pii"] + categories,
            "consensus": "mandatory_pii_reject",
            "shadow_ban": False,
            "watchlist": False,
        }

    if qwen.get("risk_level") == "critical":
        return {
            "action": "REJECT",
            "reason": "+".join(categories or ["critical_policy_violation"]),
            "risk_level": "critical",
            "categories": categories,
            "consensus": "qwen_critical",
            "shadow_ban": True,
            "watchlist": True,
        }

    votes = [
        _vote_from_bielik(bielik),
        qwen.get("recommended_action") or "review",
    ]
    if history.get("is_repeat_offender") and categories:
        votes.append("review")
    else:
        votes.append("approve" if not categories else "review")

    if votes.count("reject") >= 2:
        action = "REJECT"
    elif votes.count("approve") >= 2:
        action = "APPROVE"
    else:
        action = "FLAG_FOR_REVIEW"

    reason = "+".join(categories) if categories else "clean"
    return {
        "action": action,
        "reason": reason,
        "risk_level": qwen.get("risk_level") or "safe",
        "categories": categories or ["clean"],
        "consensus": "/".join(votes),
        "shadow_ban": action == "REJECT" and qwen.get("risk_level") == "high",
        "watchlist": history.get("is_repeat_offender") and action != "APPROVE",
    }


def _execute_action(result, moderator_id):
    from lab5 import tools as tools_mod

    action = result["action"]
    content_id = result["content_id"]
    reason = result["reason"]
    if action == "APPROVE":
        result["tool_result"] = tools_mod.call_tool("approve_content", {
            "content_id": content_id,
            "moderator_id": moderator_id,
        })
    elif action == "REJECT":
        result["tool_result"] = tools_mod.call_tool("reject_content", {
            "content_id": content_id,
            "reason": reason,
            "moderator_id": moderator_id,
        })
    else:
        result["tool_result"] = tools_mod.call_tool("flag_for_human_review", {
            "content_id": content_id,
            "priority": _risk_priority(result.get("risk_level")),
            "reason": reason,
        })

    if result.get("shadow_ban"):
        result["shadow_ban_result"] = tools_mod.call_tool("shadow_ban_user", {
            "user_id": result["user_id"],
            "duration_hours": 24,
            "reason": reason,
        })
    if result.get("watchlist"):
        result["watchlist_result"] = tools_mod.call_tool("add_to_watchlist", {
            "user_id": result["user_id"],
            "reason": reason,
        })


def moderate_content(text, user_id="anonymous", username="",
                     content_id=None, moderator_id="bot",
                     execute_action=True):
    """Run the full Lab06 moderation pipeline."""
    if not isinstance(text, str) or not text.strip():
        return {"error": "text is required"}
    storage.ensure_storage()
    content_id = content_id or f"mod_{uuid.uuid4().hex[:10]}"
    history = actions.get_user_moderation_history(user_id)

    pii = models.detect_private_info(text)
    bielik = models.classify_bielik_guard(text)
    qwen = models.classify_qwen_guard(text)
    sentiment = models.analyze_sentiment_for_moderation(text)
    entities = models.extract_moderation_entities(text)
    decision = _decide(pii, bielik, qwen, history)

    result = {
        "content_id": content_id,
        "user_id": user_id,
        "username": username,
        "text": text,
        "pii": pii,
        "bielik": bielik,
        "qwen": qwen,
        "sentiment": sentiment,
        "entities": entities,
        "user_history": history,
        **decision,
    }

    if execute_action:
        _execute_action(result, moderator_id)
        storage.save_moderation(result)
    return result


def policy_check(text, user_id="anonymous"):
    """Run moderation analysis without saving or executing actions."""
    return moderate_content(
        text=text,
        user_id=user_id,
        content_id=f"dry_{uuid.uuid4().hex[:10]}",
        execute_action=False,
    )
