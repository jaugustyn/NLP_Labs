"""CSV storage for Lab06 moderation decisions."""
import csv
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

from lab6.config import MODERATION_DATA_DIR
from utils import log_error


MODERATION_LOG = os.path.join(MODERATION_DATA_DIR, "moderation_log.csv")
USER_HISTORY = os.path.join(MODERATION_DATA_DIR, "user_moderation_history.csv")
FEEDBACK_LOG = os.path.join(MODERATION_DATA_DIR, "feedback_log.csv")
TRAIN_DATA = os.path.join(MODERATION_DATA_DIR, "train_data.csv")
ACTION_LOG = os.path.join(MODERATION_DATA_DIR, "moderation_actions.csv")
WATCHLIST = os.path.join(MODERATION_DATA_DIR, "watchlist.csv")

MODERATION_FIELDS = [
    "timestamp",
    "content_id",
    "user_id",
    "username",
    "text",
    "model_bielik_decision",
    "model_bielik_score",
    "model_qwen_decision",
    "model_qwen_score",
    "pii_detected",
    "sentiment",
    "action",
    "moderator_override",
    "reason",
    "appeal_filed",
    "categories",
    "risk_level",
    "consensus",
]
USER_FIELDS = [
    "user_id",
    "username",
    "total_violations",
    "last_violation_date",
    "categories",
    "risk_score",
    "is_repeat_offender",
    "shadow_bans",
    "appeals_filed",
]
FEEDBACK_FIELDS = [
    "content_id",
    "original_bot_decision",
    "moderator_override",
    "text_sample",
    "category",
    "confidence_before",
    "confidence_after",
    "comment",
    "timestamp",
]
TRAIN_FIELDS = ["content_id", "text", "label", "category", "comment", "timestamp"]
ACTION_FIELDS = [
    "timestamp",
    "content_id",
    "user_id",
    "action",
    "reason",
    "moderator_id",
    "priority",
    "duration_hours",
]
WATCHLIST_FIELDS = ["timestamp", "user_id", "reason", "active"]


def _storage_files():
    return {
        MODERATION_LOG: MODERATION_FIELDS,
        USER_HISTORY: USER_FIELDS,
        FEEDBACK_LOG: FEEDBACK_FIELDS,
        TRAIN_DATA: TRAIN_FIELDS,
        ACTION_LOG: ACTION_FIELDS,
        WATCHLIST: WATCHLIST_FIELDS,
    }


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def ensure_storage():
    os.makedirs(MODERATION_DATA_DIR, exist_ok=True)
    for path, fields in _storage_files().items():
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()


def _backup_file(path, reason):
    if not os.path.exists(path):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{reason}_{timestamp}.bak"
    try:
        os.replace(path, backup_path)
    except OSError as e:
        log_error(f"moderation.storage.backup:{os.path.basename(path)}", e)


def _read_rows(path):
    ensure_storage()
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected_fields = _storage_files().get(path)
            if expected_fields and reader.fieldnames != expected_fields:
                raise ValueError(
                    "Unexpected CSV header in "
                    f"{os.path.basename(path)}: {reader.fieldnames!r}"
                )
            return list(reader)
    except Exception as e:
        log_error(f"moderation.storage.read:{os.path.basename(path)}", e)
        _backup_file(path, "invalid")
        ensure_storage()
        return []


def _append_row(path, fields, row):
    ensure_storage()
    clean = {key: row.get(key, "") for key in fields}
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(clean)


def _rewrite_rows(path, fields, rows):
    ensure_storage()
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def get_moderation(content_id):
    for row in reversed(_read_rows(MODERATION_LOG)):
        if row.get("content_id") == content_id:
            return row
    return None


def append_action(
    action,
    content_id="",
    user_id="",
    reason="",
    moderator_id="bot",
    priority="",
    duration_hours="",
):
    _append_row(
        ACTION_LOG,
        ACTION_FIELDS,
        {
            "timestamp": now_iso(),
            "content_id": content_id,
            "user_id": user_id,
            "action": action.upper(),
            "reason": reason,
            "moderator_id": moderator_id,
            "priority": priority,
            "duration_hours": duration_hours,
        },
    )


def append_watchlist(user_id, reason):
    for row in list_watchlist():
        if row.get("user_id") == user_id:
            return
    _append_row(
        WATCHLIST,
        WATCHLIST_FIELDS,
        {
            "timestamp": now_iso(),
            "user_id": user_id,
            "reason": reason,
            "active": "True",
        },
    )


def list_watchlist():
    return [r for r in _read_rows(WATCHLIST) if r.get("active") == "True"]


def append_feedback(content_id, moderator_override, comment=""):
    original = get_moderation(content_id) or {}
    row = {
        "content_id": content_id,
        "original_bot_decision": original.get("action", ""),
        "moderator_override": moderator_override.upper(),
        "text_sample": (original.get("text") or "")[:240],
        "category": original.get("categories", ""),
        "confidence_before": original.get("model_qwen_score", ""),
        "confidence_after": "",
        "comment": comment,
        "timestamp": now_iso(),
    }
    _append_row(FEEDBACK_LOG, FEEDBACK_FIELDS, row)
    _append_row(
        TRAIN_DATA,
        TRAIN_FIELDS,
        {
            "content_id": content_id,
            "text": original.get("text", ""),
            "label": moderator_override.upper(),
            "category": original.get("categories", ""),
            "comment": comment,
            "timestamp": row["timestamp"],
        },
    )
    return row


def save_moderation(result):
    row = {
        "timestamp": now_iso(),
        "content_id": result.get("content_id"),
        "user_id": result.get("user_id"),
        "username": result.get("username"),
        "text": result.get("text"),
        "model_bielik_decision": result.get("bielik", {}).get("label"),
        "model_bielik_score": result.get("bielik", {}).get("score"),
        "model_qwen_decision": result.get("qwen", {}).get("risk_level"),
        "model_qwen_score": result.get("qwen", {}).get("confidence"),
        "pii_detected": result.get("pii", {}).get("has_pii"),
        "sentiment": result.get("sentiment", {}).get("sentiment"),
        "action": result.get("action"),
        "moderator_override": "False",
        "reason": result.get("reason"),
        "appeal_filed": "False",
        "categories": ";".join(result.get("categories") or []),
        "risk_level": result.get("risk_level"),
        "consensus": result.get("consensus"),
    }
    _append_row(MODERATION_LOG, MODERATION_FIELDS, row)
    rebuild_user_history()
    return row


def user_history(user_id):
    for row in _read_rows(USER_HISTORY):
        if row.get("user_id") == user_id:
            return row
    return {
        "user_id": user_id,
        "username": "",
        "total_violations": "0",
        "last_violation_date": "",
        "categories": "",
        "risk_score": "0.0",
        "is_repeat_offender": "False",
        "shadow_bans": "0",
        "appeals_filed": "0",
    }


def rebuild_user_history():
    rows = _read_rows(MODERATION_LOG)
    shadow_bans = Counter(
        r.get("user_id")
        for r in _read_rows(ACTION_LOG)
        if r.get("action") == "SHADOW_BAN"
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("user_id") or "anonymous"].append(row)

    out = []
    for user_id, items in grouped.items():
        violations = [
            r for r in items if r.get("action") in ("REJECT", "SHADOW_BAN")
        ]
        cats = Counter()
        for item in violations:
            for cat in (item.get("categories") or "").split(";"):
                if cat:
                    cats[cat] += 1
        total = len(violations)
        risk = min(1.0, total / 5.0 + shadow_bans[user_id] * 0.15)
        last = max((v.get("timestamp") for v in violations), default="")
        username = next(
            (r.get("username") for r in reversed(items) if r.get("username")),
            "",
        )
        out.append(
            {
                "user_id": user_id,
                "username": username,
                "total_violations": total,
                "last_violation_date": last,
                "categories": ";".join(sorted(cats)),
                "risk_score": round(risk, 4),
                "is_repeat_offender": total >= 3,
                "shadow_bans": shadow_bans[user_id],
                "appeals_filed": 0,
            }
        )
    _rewrite_rows(USER_HISTORY, USER_FIELDS, out)


def find_similar(text, limit=5):
    try:
        limit = int(limit or 5)
    except (TypeError, ValueError):
        limit = 5
    words = set(re.findall(r"\w+", (text or "").lower()))
    hits = []
    for row in _read_rows(MODERATION_LOG):
        other = set(re.findall(r"\w+", (row.get("text") or "").lower()))
        if not words or not other:
            continue
        score = len(words & other) / max(1, len(words | other))
        if score > 0:
            hits.append(
                {
                    "content_id": row.get("content_id"),
                    "action": row.get("action"),
                    "reason": row.get("reason"),
                    "score": round(score, 4),
                    "text": (row.get("text") or "")[:160],
                }
            )
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:max(1, min(limit, 20))]


def analytics():
    rows = _read_rows(MODERATION_LOG)
    actions = Counter(r.get("action") for r in rows)
    consensus = Counter(r.get("consensus") for r in rows if r.get("consensus"))
    categories = Counter()
    for row in rows:
        for cat in (row.get("categories") or "").split(";"):
            if cat:
                categories[cat] += 1
    feedback = _read_rows(FEEDBACK_LOG)
    users = sorted(
        _read_rows(USER_HISTORY),
        key=lambda r: float(r.get("risk_score") or 0),
        reverse=True,
    )
    return {
        "total": len(rows),
        "actions": dict(actions),
        "top_categories": categories.most_common(5),
        "consensus": dict(consensus),
        "repeat_offenders": [
            u for u in users if u.get("is_repeat_offender") == "True"
        ][:5],
        "human_overrides": len(feedback),
    }
