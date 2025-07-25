import os
import re
import sys
from typing import List, Tuple, Dict

# Ensure parent directory is on sys.path for package imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from RAGScripts.retrieve import (
    get_log_by_date,
    search_logs_by_keyword,
    filter_logs_by_mood_below,
)
from RAGScripts.format_prompt import format_prompt
from RAGScripts.openai_client import query_openai

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _detect_date(question: str) -> str:
    """Try to extract a date in YYYY-MM-DD from the question."""
    match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", question)
    if match:
        day, month, year = match.groups()
        year = int(year)
        if year < 100:
            year += 2000
        return f"{year:04d}-{int(month):02d}-{int(day):02d}"

    match = re.search(r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{2,4})?",
                       question, re.IGNORECASE)
    if match:
        day, month_str, year = match.groups()
        month = MONTH_MAP[month_str[:3].lower()]
        if year:
            year = int(year)
            if year < 100:
                year += 2000
        else:
            year = 2025
        return f"{year:04d}-{month:02d}-{int(day):02d}"
    return ""


def handle_question(question: str) -> str:
    lower_q = question.lower()
    logs: List[Tuple[str, Dict]] = []

    date = _detect_date(question)
    if date:
        logs = get_log_by_date(date)
    elif "mood" in lower_q and re.search(r"below\s*(\d+)", lower_q):
        threshold = float(re.search(r"below\s*(\d+)", lower_q).group(1))
        logs = filter_logs_by_mood_below(threshold)
    elif any(kw in lower_q for kw in ["suicidal", "burnout"]):
        for kw in ["suicidal", "burnout"]:
            if kw in lower_q:
                logs = search_logs_by_keyword(kw)
                break
    else:
        logs = search_logs_by_keyword(question)

    prompt = format_prompt(question, logs)
    try:
        return query_openai(prompt)
    except Exception as e:
        return f"Error querying language model: {e}"


def main():
    print("Personal Log Reflection CLI. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        answer = handle_question(question)
        print(answer)


if __name__ == "__main__":
    main()
