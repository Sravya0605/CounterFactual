import csv
import os
import random
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch

random.seed(42)

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

random.shuffle(rows)
split = int(len(rows) * 0.8)
train_rows, test_rows = rows[:split], rows[split:]
print(f'Train: {len(train_rows)}, Held-out test: {len(test_rows)}')

reports_dir = 'data/training_reports'

def load_graphs(rows):
    graphs, labels, md5s = [], [], []
    for row in rows:
        md5 = row['md5']
        path = os.path.join(reports_dir, f'{md5}.json')
        events = parse_cape_json(path)
        G = build_behavior_graph(events)
        graphs.append(G)
        labels.append(1 if row['avclass_family'].strip().lower() == 'emotet' else 0)
        md5s.append(md5)
    return graphs, labels, md5s

train_graphs, train_labels, train_md5s = load_graphs(train_rows)
test_graphs, test_labels, test_md5s = load_graphs(test_rows)

harness = ClassifierHarness(backend='lgbm', model_path='models/emotet_binary_holdout_lgbm.pkl')
harness.train(train_graphs, train_labels, params={
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 4,
    'max_depth': 2,
    'min_data_in_leaf': 5,
    'learning_rate': 0.05,
}, rounds=15)

print('')
print('Held-out predictions:')
test_probs = harness.predict_proba(test_graphs)
for md5, label, prob in zip(test_md5s, test_labels, test_probs):
    print(f'  {md5} (true label={label}): P(emotet) = {prob:.4f}')

# Find the held-out sample closest to the decision boundary
distances = [abs(p - 0.5) for p in test_probs]
best_idx = distances.index(min(distances))
print('')
print(f'Most borderline held-out sample: {test_md5s[best_idx]} P(emotet)={test_probs[best_idx]:.4f} true_label={test_labels[best_idx]}')

# Run the counterfactual search on it
G_test = test_graphs[best_idx]
search = CounterfactualSearch(graph=G_test)
result = search.generate()
print('')
print('--- SEARCH RESULT on held-out sample ---')
print('status:', result.get('status'))
print('candidate:', result.get('candidate'))
if result.get('edited_graph') is not None:
    new_prob = harness.predict_proba([result['edited_graph']])[0]
    print(f'New P(emotet) after edit = {new_prob:.4f}')