import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_FILE = 'cleaned_log_embeddings.jsonl'
LOG_DIR = 'CleanedDaily'
RESULTS_FILE = 'semantic_results.json'


def load_embeddings(path: str):
    dates = []
    vectors = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dates.append(data['date'])
            vectors.append(np.array(data['embedding'], dtype=float))
    vecs = np.vstack(vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-10, None)
    return dates, vecs


def load_log(date: str):
    fname = os.path.join(LOG_DIR, f'{date}.json')
    if not os.path.exists(fname):
        print(f'[WARN] Missing log file {fname}')
        return None
    with open(fname, 'r') as f:
        return json.load(f)


def pretty_print(date: str, entry: dict):
    timestamp = entry.get('timestamp', '')
    print(f'Date: {timestamp[:10] or date}')
    print(f'Summary: {entry.get("summary", "")}')
    mood = entry.get('status', {}).get('moodLevel')
    print(f'Mood: {mood}')
    tags = ', '.join(entry.get('tags', []))
    print(f'Tags: {tags}')
    insights = entry.get('insights', {})
    wins = '; '.join(insights.get('wins', []))
    losses = '; '.join(insights.get('losses', []))
    print(f'Wins: {wins}')
    print(f'Losses: {losses}')
    print('-' * 40)


def main():
    if not os.path.exists(EMB_FILE):
        print(f'Embeddings file not found: {EMB_FILE}')
        return

    dates, embeddings = load_embeddings(EMB_FILE)
    model = SentenceTransformer('all-mpnet-base-v2')

    query = input('Enter search query: ').strip()
    if not query:
        print('Empty query. Exiting.')
        return

    q_vec = model.encode([query])
    q_vec = q_vec / np.linalg.norm(q_vec)

    sims = embeddings.dot(q_vec.squeeze())
    top_idx = np.argsort(-sims)[:5]

    results = []
    for idx in top_idx:
        date = dates[idx]
        entry = load_log(date)
        if entry is None:
            continue
        pretty_print(date, entry)
        results.append({'date': date, 'score': float(sims[idx]), 'entry': entry})

    save = input('Save results to semantic_results.json? (y/N): ').strip().lower()
    if save == 'y':
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote {RESULTS_FILE}')


if __name__ == '__main__':
    main()
