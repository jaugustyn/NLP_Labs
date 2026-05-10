"""Datetime tool — current local date/time, optionally for a timezone."""
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
    _ZONEINFO_OK = True
except Exception:
    _ZONEINFO_OK = False


def datetime_now(tz=None):
    """Return the current date/time. Optional IANA timezone like
    'Europe/Warsaw' or 'UTC'. Defaults to system local time."""
    try:
        if tz:
            if not _ZONEINFO_OK:
                return {"error": "zoneinfo unavailable on this system."}
            now = datetime.now(ZoneInfo(tz))
            tz_name = tz
        else:
            now = datetime.now().astimezone()
            tz_name = str(now.tzinfo) if now.tzinfo else "local"
    except Exception as e:
        return {"error": f"Bad timezone: {e}"}

    return {
        "timezone": tz_name,
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": now.year,
        "utc_iso": datetime.now(timezone.utc).isoformat(),
    }


SCHEMA = {
    "type": "function",
    "function": {
        "name": "datetime_now",
        "description": (
            "Return the current date and time. Use whenever the user's "
            "question depends on 'today', 'now', the current month, "
            "season, day of week, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tz": {
                    "type": "string",
                    "description": (
                        "Optional IANA timezone, e.g. 'Europe/Warsaw'. "
                        "Defaults to local system time."
                    ),
                },
            },
        },
    },
}
