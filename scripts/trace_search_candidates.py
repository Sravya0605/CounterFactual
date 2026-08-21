import csv
import random
import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.feasibility import validate_candidate, candidate_cost

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

from src.counterfactual.search import CounterfactualSearch
search = CounterfactualSearch(classifier=harness, graph=G)

print('Calling propose() directly to see every candidate generated:')
count = 0
for candidate in search.propose():
    count += 1
    valid = validate_candidate(G, candidate)
    cost = candidate_cost(candidate)
    if candidate.get('delete_nodes') == ['n164'] or valid:
        print(f'  #{count}: {candidate} valid={valid} cost={cost}')
    if count >= 60:
        print('  ...(stopping trace at 60 candidates)')
        break
print(f'\nTotal candidates seen: {count}')