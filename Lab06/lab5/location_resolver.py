"""Shared location normalization and Open-Meteo geocoding helpers."""

import re

import requests

from config import (
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
    OPEN_METEO_GEOCODING_URL,
)


_HEADERS = {"User-Agent": HTTP_USER_AGENT}


def fold_text(text):
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


def normalize_location_text(text):
    value = re.sub(r"\s+", " ", (text or "").strip(" .,!?:;\"'()[]{}"))
    return value


def location_candidates(location):
    raw = normalize_location_text(location)
    if not raw:
        return []

    candidates = []

    def add(value):
        value = normalize_location_text(value)
        if value and value not in candidates:
            candidates.append(value)

    add(raw)
    folded = fold_text(raw)
    if folded != raw.lower():
        add(folded)

    lower = raw.lower()
    transforms = []
    if lower.endswith("awie"):
        transforms.append(raw[:-4] + "awa")
    if lower.endswith("owie"):
        transforms.append(raw[:-2])
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
        folded_value = fold_text(value)
        if folded_value != value.lower():
            add(folded_value)

    return candidates


def city_lookup_keys(city):
    """Return normalized search keys used to deduplicate city tool calls."""
    return set(location_candidates(city))


def geocode_city(city, require_timezone=False):
    """Resolve a city through Open-Meteo geocoding.

    Returns the most populous matching result for the first candidate
    that Open-Meteo can resolve.
    """
    for candidate in location_candidates(city):
        for language in ("pl", "en"):
            response = requests.get(
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
            response.raise_for_status()
            results = (response.json() or {}).get("results") or []
            if not results:
                continue
            item = max(results, key=lambda result: result.get("population") or 0)
            if require_timezone and not item.get("timezone"):
                continue
            return {
                "name": item.get("name"),
                "country": item.get("country"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "timezone": item.get("timezone"),
            }
    return None
