from typing import Dict, List, Tuple


def format_single_log(date: str, log: Dict) -> str:
    status = log.get("status", {})
    insights = log.get("insights", {})
    return (
        f"Date: {date}\n"
        f"Summary: {log.get('summary', '')}\n"
        f"Mood: {status.get('moodLevel', '')}\n"
        f"Energy: {status.get('energyLevel', '')}\n"
        f"Tags: {', '.join(log.get('tags', []))}\n"
        f"Insights:\n"
        f"- Wins: {', '.join(insights.get('wins', []))}\n"
        f"- Losses: {', '.join(insights.get('losses', []))}\n"
        f"- Ideas: {', '.join(insights.get('ideas', []))}\n"
    )


def format_prompt(question: str, logs: List[Tuple[str, Dict]]) -> str:
    if not logs:
        return f"User asked: {question}\n\nNo matching logs were found."

    formatted_logs = "\n".join(format_single_log(date, log) for date, log in logs)
    prompt = (
        f"User asked: {question}\n\n"
        f"Here are the matching logs:\n{formatted_logs}"
    )
    return prompt
