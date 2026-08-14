import csv
import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.utils.graph_features import graph_list_to_bow, build_feature_vocab

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
    labels.append(row['avclass_family'])
    md5s.append(md5)
    print(f'[{i}/{len(rows)}] {md5} ({row["avclass_family"]}) -- graph built, {G.number_of_nodes()} nodes')

print('')
print('Building feature vocabulary...')
vocab = build_feature_vocab(graphs)
print(f'Vocabulary size: {len(vocab)} features')

print('Building feature matrix...')
X = graph_list_to_bow(graphs, vocab=vocab)

print('')
print(f'Feature matrix shape: {X.shape}')
print(f'Labels: {len(labels)}')

from collections import Counter
print('Label distribution:', Counter(labels))

X.to_csv('data/feature_matrix.csv', index=False)
with open('data/labels.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['md5', 'label'])
    for m, l in zip(md5s, labels):
        writer.writerow([m, l])

print('')
print('Saved data/feature_matrix.csv and data/labels.csv')