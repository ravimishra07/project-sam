import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from RAGScripts.load_logs import load_logs

# Load logs once at import time
LOGS: Dict[str, Any] = load_logs()


def get_log_by_date(date_str: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Return log(s) exactly matching the given ISO date string."""
    data = LOGS.get(date_str)
    return [(date_str, data)] if data else []


def _fuzzy_contains(needle: str, haystack: str, threshold: float = 0.6) -> bool:
    haystack = haystack.lower()
    needle = needle.lower()
    if needle in haystack:
        return True
    ratio = SequenceMatcher(None, needle, haystack).ratio()
    return ratio >= threshold


def search_logs_by_keyword(keyword: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Search logs for a keyword across summary, insights, and tags."""
    results = []
    for date, log in LOGS.items():
        parts = [log.get("summary", "")]
        insights = log.get("insights", {})
        for key in ("wins", "losses", "ideas"):
            parts.extend(insights.get(key, []))
        parts.extend(log.get("tags", []))
        text = " ".join(parts)
        if _fuzzy_contains(keyword, text):
            results.append((date, log))
    return results


def filter_logs_by_mood_below(threshold: float) -> List[Tuple[str, Dict[str, Any]]]:
    """Return logs where moodLevel is below the given threshold."""
    results = []
    for date, log in LOGS.items():
        try:
            mood = float(log.get("status", {}).get("moodLevel", 0))
            if mood < threshold:
                results.append((date, log))
        except (TypeError, ValueError):
            continue
    return results


def search_logs_by_tag(tag: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Return logs that contain the given tag (fuzzy match)."""
    results = []
    for date, log in LOGS.items():
        for existing_tag in log.get("tags", []):
            if _fuzzy_contains(tag, existing_tag):
                results.append((date, log))
                break
    return results
