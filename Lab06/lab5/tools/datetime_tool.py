"""Datetime tool — current local date/time for a timezone or city."""
from datetime import datetime, timezone
import re

import requests

from config import (
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
    OPEN_METEO_GEOCODING_URL,
)

try:
    from zoneinfo import ZoneInfo
    _ZONEINFO_OK = True
except Exception:
    _ZONEINFO_OK = False


_HEADERS = {"User-Agent": HTTP_USER_AGENT}


def _fold(text):
    return (
        (text or "").lower()
        .replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ó", "o")
        .replace("ś", "s")
        .replace("ź", "z")
        .replace("ż", "z")
    )


def _city_candidates(city):
    raw = re.sub(r"\s+", " ", (city or "").strip(" .,!?:;\"'()[]{}"))
    if not raw:
        return []

    candidates = []

    def add(value):
        value = re.sub(r"\s+", " ", (value or "").strip())
        if value and value not in candidates:
            candidates.append(value)

    add(raw)
    folded = _fold(raw)
    if folded != raw.lower():
        add(folded)

    # Generic Polish locative/genitive approximations. These are not
    # city-specific aliases; they create search candidates for geocoding.
    lower = raw.lower()
    transforms = []
    if lower.endswith("awie"):
        transforms.append(raw[:-4] + "awa")
    if lower.endswith("onie"):
        transforms.extend([raw[:-2], raw[:-3] + "na"])
    if lower.endswith("nie"):
        transforms.append(raw[:-2])
    if lower.endswith("ie"):
        transforms.append(raw[:-2])
    if lower.endswith("żu") or lower.endswith("zu"):
        transforms.append(raw[:-1])
    if lower.endswith("u") and len(raw) > 4:
        transforms.append(raw[:-1])

    for value in transforms:
        add(value)
        folded_value = _fold(value)
        if folded_value != value.lower():
            add(folded_value)

    return candidates


def _geocode_city(city):
    for candidate in _city_candidates(city):
        for language in ("pl", "en"):
            r = requests.get(
                OPEN_METEO_GEOCODING_URL,
                params={
                    "name": candidate,
                    "count": 5,
                    "language": language,
                    "format": "json",
                },
                headers=_HEADERS,
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            results = (r.json() or {}).get("results") or []
            if not results:
                continue
            item = max(results, key=lambda x: x.get("population") or 0)
            if item.get("timezone"):
                return {
                    "city": item.get("name"),
                    "country": item.get("country"),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "timezone": item.get("timezone"),
                }
    return None


def datetime_now(tz=None, city=None):
    """Return the current date/time. Accepts either an IANA timezone
    like 'Europe/Warsaw' or a city resolved through Open-Meteo
    geocoding. Defaults to 'Europe/Warsaw'."""
    geo = None
    if city:
        try:
            geo = _geocode_city(city)
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
        result.update(geo)
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
