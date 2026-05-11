"""Shared command parsing and formatting helpers."""

import re
import shlex
import traceback


def parse_params(text):
    """Parse key=value command parameters.

    Supports unquoted values, double/single quoted values and falls back
    to the legacy regex parser when the input contains an unfinished
    quote.
    """
    text = text or ""
    params = {}
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = []

    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            params[key.lower()] = value

    if params:
        return params

    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))', text):
        key = match.group(1).lower()
        value = next(
            group for group in match.groups()[1:]
            if group is not None
        )
        params[key] = value
    return params


def extract_quoted_args(text, count):
    """Extract exactly `count` quoted arguments from text."""
    matches = re.findall(r'"([^"]*)"', text)
    return matches if len(matches) == count else None


def log_error(context, error):
    print(f"[ERROR] {context}: {type(error).__name__}: {error}")
    print(traceback.format_exc())


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def truncate(text, max_len=200):
    """Truncate `text` to at most `max_len` characters with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
