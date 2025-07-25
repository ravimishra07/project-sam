import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer

CLEANED_DIR = 'CleanedDaily'
OUTPUT_FILE = 'cleaned_log_embeddings.jsonl'
MODEL_NAME = 'all-mpnet-base-v2'

model = SentenceTransformer(MODEL_NAME)

def extract_date(ts: str, fname: str) -> str:
    if ts:
        try:
            if ts.endswith('Z'):
                ts = ts[:-1]
            dt = datetime.fromisoformat(ts)
            return f"{dt.day}-{dt.month}-{dt.year % 100}"
        except Exception as e:
            print(f'[WARN] Failed to parse timestamp in {fname}: {e}')
    return fname.replace('.json', '')

def build_text(data: dict) -> str:
    parts = []
    summary = data.get('summary')
    if summary:
        parts.append(str(summary))
    tags = data.get('tags') or []
    if isinstance(tags, list):
        parts.extend(map(str, tags))
    else:
        parts.append(str(tags))
    insights = data.get('insights') or {}
    for key in ['wins', 'losses', 'ideas']:
        val = insights.get(key)
        if isinstance(val, list):
            parts.extend(map(str, val))
        elif val:
            parts.append(str(val))
    triggers = data.get('triggerEvents') or []
    if isinstance(triggers, list):
        parts.extend(map(str, triggers))
    elif triggers:
        parts.append(str(triggers))
    return ' '.join(parts)

def process_file(fname: str):
    path = os.path.join(CLEANED_DIR, fname)
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
    except Exception as e:
        print(f'[ERROR] Could not read {path}: {e}')
        return None
    text = build_text(data)
    try:
        emb = model.encode([text])[0]
    except Exception as e:
        print(f'[ERROR] Could not encode {fname}: {e}')
        return None
    date = extract_date(data.get('timestamp', ''), fname)
    return {'date': date, 'embedding': emb.tolist()}

def main():
    files = [f for f in os.listdir(CLEANED_DIR) if f.endswith('.json')]
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_file, f) for f in files]
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for item in results:
            json.dump(item, out)
            out.write('\n')
    print(f'Embedded {len(results)} logs.')

if __name__ == '__main__':
    main()
