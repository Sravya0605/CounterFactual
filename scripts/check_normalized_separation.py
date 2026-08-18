import csv
import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.utils.graph_features import graph_to_normalized_api_features

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f) if r['avclass_family'].strip().lower() in ('agenttesla', 'qbot')]

results = {'agenttesla': [], 'qbot': []}
for row in rows:
    family = row['avclass_family'].strip().lower()
    md5 = row['md5']
    path = os.path.join('data/training_reports', f'{md5}.json')
    events = parse_cape_json(path)
    G = build_behavior_graph(events)
    norm = graph_to_normalized_api_features(G)
    ratio = norm.get('ratio_createtoolhelp32snapshot', 0)
    results[family].append((md5, ratio))

for family in ('agenttesla', 'qbot'):
    vals = [r for _, r in results[family]]
    print(f'{family} (n={len(vals)}): min={min(vals):.6f} max={max(vals):.6f} mean={sum(vals)/len(vals):.6f}')

misclassified_ratio = next(r for m, r in results['qbot'] if m == '2f46ceec4c08f19c162c692b9cb5ce3a')
agenttesla_max = max(r for _, r in results['agenttesla'])
print(f'Misclassified sample ratio: {misclassified_ratio:.6f}')
print(f'agenttesla max ratio: {agenttesla_max:.6f}')
print(f'Does misclassified ratio exceed agenttesla max? {misclassified_ratio > agenttesla_max}')