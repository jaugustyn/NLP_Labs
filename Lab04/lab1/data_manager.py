import json
import os
from datetime import datetime

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_DIR, "sentences.json")

def load_records():
    if not os.path.exists(DATA_FILE):
        return []
    
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        # Keep malformed file as backup to avoid silent data loss
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{DATA_FILE}.corrupt_{timestamp}.bak"
        try:
            os.replace(DATA_FILE, backup_path)
        except OSError:
            pass
        return []

def save_record(text, text_class):
    records = load_records()
    
    new_entry = {
        "text": text,
        "class": text_class
    }
    
    records.append(new_entry)
    
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
