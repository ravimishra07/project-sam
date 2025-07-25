import json
import os
from pathlib import Path

INPUT_DIR = Path('CleanedDaily')
OUTPUT_FILE = 'cleaned_chat_format.jsonl'


def _read_json(path: Path) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load {path}: {exc}")
        return {}


def _join_list(items):
    if not isinstance(items, list):
        return ''
    return ', '.join(str(x) for x in items)


def convert_file(path: Path) -> dict:
    data = _read_json(path)

    status = data.get('status', {})
    mood = status.get('moodLevel', '')
    energy = status.get('energyLevel', '')
    sleep_dur = status.get('sleepDuration', '')
    sleep_qual = status.get('sleepQuality', '')
    stability = status.get('stabilityScore', '')

    user_prompt = (
        "Here is today's mental log data:"\
        f"\nMood: {mood}"\
        f"\nEnergy: {energy}"\
        f"\nSleep Duration: {sleep_dur} hrs"\
        f"\nSleep Quality: {sleep_qual}"\
        f"\nStability Score: {stability}"
    )

    summary = data.get('summary', '')

    insights = data.get('insights', {})
    wins = _join_list(insights.get('wins', []))
    losses = _join_list(insights.get('losses', []))
    ideas = _join_list(insights.get('ideas', []))
    goals = _join_list(data.get('goals', []))
    tags = _join_list(data.get('tags', []))
    symptoms = _join_list(data.get('symptomChecklist', []))
    triggers = _join_list(data.get('triggerEvents', []))

    detailed_response = (
        f"Wins: [{wins}]"\
        f"\nLosses: [{losses}]"\
        f"\nIdeas: [{ideas}]"\
        f"\nGoals: [{goals}]"\
        f"\nTags: [{tags}]"\
        f"\nSymptoms: [{symptoms}]"\
        f"\nTriggers: [{triggers}]"
    )

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": summary},
            {"role": "user", "content": "Now extract insights, goals, and tags."},
            {"role": "assistant", "content": detailed_response},
        ]
    }


def main():
    paths = sorted(p for p in INPUT_DIR.glob('*.json'))
    with open(OUTPUT_FILE, 'w') as out_f:
        for path in paths:
            entry = convert_file(path)
            json.dump(entry, out_f)
            out_f.write('\n')
    print(f"Wrote {len(paths)} entries to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
