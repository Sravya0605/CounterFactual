import time
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.feasibility import candidate_cost

MD5 = '83a3340793cec4cb51526c5f4d6711b9'
MODEL_PATH = 'models/full_dataset_agenttesla_vs_qbot_lgbm.pkl'

events = parse_cape_json(f'data/training_reports/{MD5}.json')
G = build_behavior_graph(events)

harness = ClassifierHarness(backend='lgbm', model_path=MODEL_PATH)

t0 = time.time()
search = CounterfactualSearch(classifier=harness, graph=G)
candidates = search.propose()
insertion_candidates = [c for c in candidates if c.get('insert_nodes')]
print(f'propose(): {len(candidates)} total candidates, {len(insertion_candidates)} are insertion-type')

result = search.run()
elapsed = time.time() - t0

print(f'run(): {elapsed:.1f}s')
print('status:', result.get('status'))
cand = result.get('candidate')
print('candidate:', cand)
if cand is not None:
    print('candidate_cost:', candidate_cost(cand))
print('orig_prob:', result.get('orig_prob'))
print('new_prob:', result.get('new_prob'))