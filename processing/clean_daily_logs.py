import json
import os
from datetime import datetime

DAILY_DIR = os.path.join('logs', 'daily')
OUTPUT_DIR = 'CleanedDaily'

STANDARD_TEMPLATE = {
    "timestamp": "",
    "summary": "",
    "status": {
        "moodLevel": "",
        "sleepQuality": "",
        "sleepDuration": "",
        "energyLevel": "",
        "stabilityScore": ""
    },
    "insights": {
        "wins": [],
        "losses": [],
        "ideas": []
    },
    "goals": [],
    "tags": [],
    "triggerEvents": [],
    "symptomChecklist": []
}


def _parse_timestamp(raw_ts: str) -> datetime:
    """Return datetime object from a timestamp string and fix year if needed."""
    ts = raw_ts.strip()
    if ts.endswith('Z'):
        ts = ts[:-1]
    dt = datetime.fromisoformat(ts)
    if dt.year != 2025:
        dt = dt.replace(year=2025)
    return dt


def _get_status(data: dict) -> dict:
    status = data.get('status', {})
    return {
        'moodLevel': status.get('moodLevel') or status.get('mood_level', ''),
        'sleepQuality': status.get('sleepQuality') or status.get('sleep_quality', ''),
        'sleepDuration': status.get('sleepDuration') or status.get('sleep_duration', ''),
        'energyLevel': status.get('energyLevel') or status.get('energy_level', ''),
        'stabilityScore': status.get('stabilityScore') or status.get('stability_score', ''),
    }


def _get_insights(data: dict) -> dict:
    insights = data.get('insights', {})
    return {
        'wins': insights.get('wins', []) or [],
        'losses': insights.get('losses', []) or [],
        'ideas': insights.get('ideas', []) or []
    }


def clean_logs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name_counts = {}
    for fname in os.listdir(DAILY_DIR):
        if not fname.endswith('.json'):
            continue
        if fname == 'daily_now.json':
            # remove this file if present
            os.remove(os.path.join(DAILY_DIR, fname))
            continue
        path = os.path.join(DAILY_DIR, fname)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f'[WARN] Could not read {path}: {e}')
            continue
        timestamp = data.get('timestamp') or data.get('timeStamp')
        if not timestamp:
            print(f'[WARN] Missing timestamp in {fname}')
            continue
        try:
            dt = _parse_timestamp(timestamp)
        except Exception as e:
            print(f'[WARN] Invalid timestamp in {fname}: {e}')
            continue
        base_name = f"{dt.day}-{dt.month}-{dt.year % 100}"
        count = name_counts.get(base_name, 0)
        if count:
            new_filename = f"{base_name}_{count+1}.json"
        else:
            new_filename = f"{base_name}.json"
        name_counts[base_name] = count + 1
        clean = STANDARD_TEMPLATE.copy()
        clean['timestamp'] = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        clean['summary'] = data.get('summary', '')
        clean['status'] = _get_status(data)
        clean['insights'] = _get_insights(data)
        clean['goals'] = data.get('goals', []) or []
        clean['tags'] = data.get('tags', []) or []
        clean['triggerEvents'] = data.get('triggerEvents') or data.get('trigger_events', []) or []
        clean['symptomChecklist'] = data.get('symptomChecklist') or data.get('symptom_checklist', []) or []

        out_path = os.path.join(OUTPUT_DIR, new_filename)
        with open(out_path, 'w') as f:
            json.dump(clean, f, indent=2)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    clean_logs()
