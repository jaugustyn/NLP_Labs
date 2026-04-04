import re
import traceback


def parse_params(text):
    """Parse key=value and key="quoted value" from command text."""
    params = {}
    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', text):
        key = match.group(1).lower()
        value = match.group(2) if match.group(2) is not None else match.group(3)
        params[key] = value
    return params


def extract_quoted_args(text, count):
    """Extract exactly `count` quoted arguments from text."""
    matches = re.findall(r'"([^"]*)"', text)
    return matches if len(matches) == count else None


def log_error(context, error):
    """Log error with traceback to console."""
    print(f"[ERROR] {context}: {type(error).__name__}: {error}")
    print(traceback.format_exc())


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"
