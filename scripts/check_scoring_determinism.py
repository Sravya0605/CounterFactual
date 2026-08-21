import os
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness

MD5 = '83a3340793cec4cb51526c5f4d6711b9'
MODEL_PATH = 'models/full_dataset_agenttesla_vs_qbot_lgbm.pkl'

print('PYTHONHASHSEED env var:', os.environ.get('PYTHONHASHSEED', '(not set -- randomized per process)'))

harness = ClassifierHarness(backend='lgbm', model_path=MODEL_PATH)

path = os.path.join('data/training_reports', f'{MD5}.json')

# Build the graph and score it THREE separate times, all within this one process
for i in range(3):
    events = parse_cape_json(path)
    G = build_behavior_graph(events)
    prob = harness.predict_proba([G])[0]
    print(f'Run {i+1} (same process): nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, prob={prob!r}')