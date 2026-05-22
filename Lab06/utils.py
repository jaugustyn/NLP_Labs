"""Shared command parsing and formatting helpers."""

import logging
import re
import shlex


_PARAM_PATTERN = re.compile(r"(?<!\w)([A-Za-z_]\w*)\s*=")
_QUOTED_PATTERN = re.compile(r'"([^"]*)"|\'([^\']*)\'')


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

    pos = 0
    while True:
        match = _PARAM_PATTERN.search(text, pos)
        if not match:
            break
        key = match.group(1).lower()
        value, pos = _read_param_value(text, match.end())
        params[key] = value
    return params


def _read_param_value(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return "", start
    if text[start] in ("\"", "'"):
        return _read_quoted_value(text, start)

    next_param = _PARAM_PATTERN.search(text, start)
    end = next_param.start() if next_param else len(text)
    return text[start:end].strip(), end


def _read_quoted_value(text, start):
    quote = text[start]
    chars = []
    escaped = False
    index = start + 1
    while index < len(text):
        char = text[index]
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return "".join(chars), index + 1
        else:
            chars.append(char)
        index += 1
    if escaped:
        chars.append("\\")
    return "".join(chars), len(text)


def extract_param_value(text, key):
    """Return a command parameter value, tolerating an unfinished quote."""
    if not key:
        return None
    pattern = re.compile(
        rf"(?<!\w){re.escape(key)}\s*=",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text or "")
    if not match:
        return None
    value, _ = _read_param_value(text or "", match.end())
    return value.strip()


def extract_first_quoted(text):
    """Return the first single- or double-quoted value from text."""
    match = _QUOTED_PATTERN.search(text or "")
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").strip()


def extract_quoted_args(text, count):
    """Extract exactly `count` quoted arguments from text."""
    matches = re.findall(r'"([^"]*)"', text)
    return matches if len(matches) == count else None


def log_error(context, error):
    exc_info = None
    if isinstance(error, BaseException):
        exc_info = (type(error), error, error.__traceback__)
    logging.getLogger("nlp_labs").error(
        "%s: %s: %s",
        context,
        type(error).__name__,
        error,
        exc_info=exc_info,
    )


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
