"""Weather tool using Open-Meteo (no API key required)."""

import requests

from config import (
    OPEN_METEO_FORECAST_URL,
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
)
from lab5.location_resolver import geocode_city


_HEADERS = {"User-Agent": HTTP_USER_AGENT}

# Subset of Open-Meteo WMO weather codes
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(city):
    """Return current weather for a city."""
    if not isinstance(city, str) or not city.strip():
        return {"error": "City name is required."}
    try:
        geo = geocode_city(city.strip())
    except Exception as e:
        return {"error": f"Geocoding failed: {e}"}
    if not geo:
        return {"error": f"City not found: {city}"}

    try:
        r = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "current_weather": "true",
                "timezone": "auto",
            },
            headers=_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        cw = (r.json() or {}).get("current_weather") or {}
    except Exception as e:
        return {"error": f"Forecast fetch failed: {e}"}

    code = cw.get("weathercode")
    return {
        "city": geo["name"],
        "country": geo["country"],
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "temperature_c": cw.get("temperature"),
        "wind_kmh": cw.get("windspeed"),
        "wind_direction_deg": cw.get("winddirection"),
        "weather_code": code,
        "description": _WMO.get(code, "unknown"),
        "observed_at": cw.get("time"),
    }


SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get current weather (temperature, wind, conditions) for a "
            "city. Use whenever the user asks about weather conditions, "
            "temperature, rain, snow, or whether it is good to go "
            "outside."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name in any language, e.g. 'Warsaw', 'Paryż'.",
                },
            },
            "required": ["city"],
        },
    },
}
