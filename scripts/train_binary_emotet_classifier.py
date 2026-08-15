import csv
import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

reports_dir = 'data/training_reports'
graphs = []
labels = []
md5s = []

for i, row in enumerate(rows, 1):
    md5 = row['md5']
    path = os.path.join(reports_dir, f'{md5}.json')
    events = parse_cape_json(path)
    G = build_behavior_graph(events)
    graphs.append(G)
    label = 1 if row['avclass_family'].strip().lower() == 'emotet' else 0
    labels.append(label)
    md5s.append(md5)
    print(f'[{i}/{len(rows)}] {md5} ({row["avclass_family"]}) -> label={label}')

print('')
print(f'Positive (emotet) samples: {sum(labels)}')
print(f'Negative (not emotet) samples: {len(labels) - sum(labels)}')

harness = ClassifierHarness(backend='lgbm', model_path='models/emotet_binary_lgbm.pkl')
harness.train(graphs, labels)

# Sanity-check: not degenerate
tree_info = harness.model.dump_model()['tree_info']
print('')
print(f'Number of trees: {len(tree_info)}')
leaf_counts = [t['num_leaves'] for t in tree_info]
print(f'Leaves per tree: min={min(leaf_counts)}, max={max(leaf_counts)}, mean={sum(leaf_counts)/len(leaf_counts):.1f}')

importance = harness.model.feature_importance(importance_type='gain')
nonzero = sum(1 for v in importance if v > 0)
print(f'Features with nonzero gain: {nonzero} / {len(importance)}')

# In-sample check on the exact sample we plan to run the counterfactual search against
TEST_MD5 = '951df57c534b6acaeecced8cc6e898c7'
idx = md5s.index(TEST_MD5)
test_graph = graphs[idx]
prob = harness.predict_proba([test_graph])[0]
print('')
print(f'Test sample {TEST_MD5} (true label: emotet) -> predicted P(emotet) = {prob:.4f}')