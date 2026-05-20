"""JSON persistence helpers for Lab 1 examples."""

import json
import os
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_DIR, "sentences.json")


def _backup_data_file(reason):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DATA_FILE}.{reason}_{timestamp}.bak"
    try:
        os.replace(DATA_FILE, backup_path)
    except OSError:
        pass


def _is_valid_record(record):
    return (
        isinstance(record, dict)
        and set(record) == {"text", "class"}
        and isinstance(record.get("text"), str)
        and isinstance(record.get("class"), str)
    )


def load_records():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            records = json.loads(content)
    except json.JSONDecodeError:
        _backup_data_file("corrupt")
        return []

    if not isinstance(records, list) or not all(_is_valid_record(r) for r in records):
        _backup_data_file("invalid")
        return []

    return records


def save_record(text, text_class):
    records = load_records()

    new_entry = {
        "text": text,
        "class": text_class,
    }

    records.append(new_entry)

    tmp_path = f"{DATA_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DATA_FILE)
