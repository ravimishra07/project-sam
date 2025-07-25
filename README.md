# Daily Log Cleanup

This repository contains personal log files under `logs/`.
The script `processing/clean_daily_logs.py` normalizes entries in
`logs/daily/` and writes them to `CleanedDaily/`.

## Usage

Ensure Python 3 is installed and run:

```bash
python processing/clean_daily_logs.py
```

The script will:

1. Create `CleanedDaily/` if it does not exist.
2. Read each JSON file in `logs/daily/` (excluding `daily_now.json`).
3. Rename the file based on the `timestamp` value and correct any year set
   incorrectly (e.g., `2925` -> `2025`).
4. Output a cleaned JSON structure with the following format:

```json
{
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
```

Cleaned files are saved in `CleanedDaily/` using names like
`15-5-25.json`. If multiple logs exist for the same day, a numeric
suffix is appended (e.g., `15-5-25_2.json`).

## Export chat format

To create a single JSONL file for LoRA fine-tuning, run:

```bash
python processing/convert_to_chat.py
```

This reads all files in `CleanedDaily/` and writes `cleaned_chat_format.jsonl`.
Each line in this file is a conversation with user prompts summarizing the
log's status and a follow-up containing insights, goals and other details.
