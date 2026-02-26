import json
import os
from datetime import datetime, timezone

from config import HISTORY_PATH


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


def load_history(limit: int = 20) -> list[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        return []
    return data[-limit:][::-1]


def save_history_entry(entry: dict) -> None:
    _ensure_parent_dir(HISTORY_PATH)
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            loaded = json.load(file)
            if isinstance(loaded, list):
                history = loaded

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    history.append(payload)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=True, indent=2)
