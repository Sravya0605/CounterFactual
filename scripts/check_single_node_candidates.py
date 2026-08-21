import csv
import random
import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch

random.seed(42)
with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
random.shuffle(rows)
split = int(len(rows) * 0.8)
test_rows = rows[split:]

TARGET_MD5 = '81aca65079b5c5f2c5456a9355cdcbd6'
path = os.path.join('data/training_reports', f'{TARGET_MD5}.json')
events = parse_cape_json(path)
G = build_behavior_graph(events)

harness = ClassifierHarness(backend='lgbm', model_path='models/emotet_binary_holdout_lgbm.pkl')
search = CounterfactualSearch(classifier=harness, graph=G)

single_node_candidates = []
n164_appearances = []
count = 0
for candidate in search.propose():
    count += 1
    deletes = candidate.get('delete_nodes', [])
    if len(deletes) == 1:
        single_node_candidates.append(deletes[0])
    if 'n164' in deletes:
        n164_appearances.append(candidate)
    if count >= 2000:
        break

print(f'Total candidates checked: {count}')
print(f'Single-node-deletion candidates found: {len(single_node_candidates)}')
print(f'First 20 single-node candidates: {single_node_candidates[:20]}')
print(f'\nAny candidate mentioning n164: {len(n164_appearances)}')
for c in n164_appearances[:10]:
    print(f'  {c}')