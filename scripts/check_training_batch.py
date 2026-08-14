import os
import csv
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

md5_to_family = {r['md5']: r['avclass_family'] for r in rows}

reports_dir = 'data/training_reports'
results = []
failed = []

for i, row in enumerate(rows, 1):
    md5 = row['md5']
    path = os.path.join(reports_dir, f'{md5}.json')
    try:
        events = parse_cape_json(path)
        G = build_behavior_graph(events)
        results.append({
            'md5': md5,
            'family': row['avclass_family'],
            'events': len(events),
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
        })
        print(f"[{i}/{len(rows)}] {md5} ({row['avclass_family']}): events={len(events)} nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    except Exception as exc:
        failed.append((md5, row['avclass_family'], str(exc)))
        print(f"[{i}/{len(rows)}] {md5} ({row['avclass_family']}): FAILED -- {type(exc).__name__}: {exc}")

print('')
print(f'Succeeded: {len(results)} / {len(rows)}')
print(f'Failed: {len(failed)}')
if failed:
    print('Failures:')
    for md5, fam, err in failed:
        print(f'  {md5} ({fam}): {err}')