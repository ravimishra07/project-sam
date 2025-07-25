import json
import os
from typing import Dict, Any

def load_logs(directory: str = "CleanedDaily") -> Dict[str, Any]:
    """Load all JSON logs from the given directory.

    Each log is stored in a dictionary keyed by the ISO date
    extracted from the timestamp field.
    """
    logs = {}
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            timestamp = data.get("timestamp")
            if timestamp:
                date_str = timestamp.split("T")[0]
            else:
                # Fallback to filename without extension
                date_str = os.path.splitext(filename)[0]
            logs[date_str] = data
        except (json.JSONDecodeError, OSError):
            # Skip malformed files but continue loading others
            continue
    return logs

if __name__ == "__main__":
    # Simple manual test
    logs = load_logs()
    print(f"Loaded {len(logs)} logs")
