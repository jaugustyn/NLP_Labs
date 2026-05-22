"""Datetime tool — current local date/time for a timezone or city."""
from datetime import datetime, timezone

from lab5.location_resolver import geocode_city

try:
    from zoneinfo import ZoneInfo
    _ZONEINFO_OK = True
except Exception:
    _ZONEINFO_OK = False


def datetime_now(tz=None, city=None):
    """Return the current date/time. Accepts either an IANA timezone
    like 'Europe/Warsaw' or a city resolved through Open-Meteo
    geocoding. Defaults to 'Europe/Warsaw'."""
    geo = None
    if city:
        try:
            geo = geocode_city(city, require_timezone=True)
        except Exception as e:
            return {"error": f"City geocoding failed: {e}", "city_query": city}
        if not geo:
            return {"error": f"City not found: {city}", "city_query": city}
        tz = geo["timezone"]
    if not tz:
        tz = "Europe/Warsaw"
    try:
        if not _ZONEINFO_OK:
            return {"error": "zoneinfo unavailable on this system."}
        now = datetime.now(ZoneInfo(tz))
    except Exception as e:
        return {"error": f"Bad timezone: {e}"}

    result = {
        "timezone": tz,
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": now.year,
        "utc_iso": datetime.now(timezone.utc).isoformat(),
    }
    if geo:
        result.update({
            "city": geo.get("name"),
            "country": geo.get("country"),
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
            "timezone": geo.get("timezone"),
        })
        result["city_query"] = city
    return result


SCHEMA = {
    "type": "function",
    "function": {
        "name": "datetime_now",
        "description": (
            "Return the current date and time. Use whenever the user's "
            "question depends on 'today', 'now', the current month, "
            "season, day of week, or time in a city."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "Optional city name to resolve to a timezone, "
                        "e.g. 'Tokyo', 'Barcelona', 'Waszyngton'."
                    ),
                },
                "tz": {
                    "type": "string",
                    "description": (
                        "Optional IANA timezone, e.g. 'Europe/Warsaw' or "
                        "'Asia/Tokyo'. Used when city is not provided. "
                        "Defaults to 'Europe/Warsaw'."
                    ),
                },
            },
        },
    },
}
