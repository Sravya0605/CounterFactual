import pickle
import pandas as pd
from collections import Counter
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.feasibility import validate_candidate, candidate_cost
from src.utils.graph_features import graph_to_api_counts, graph_to_ngram_features, graph_to_edge_features, graph_to_entropy_features
from src.classifier.lgbm_model import predict_proba as lgbm_predict_proba

with open('models/full_dataset_agenttesla_vs_qbot_lgbm.pkl', 'rb') as f:
    saved = pickle.load(f)
vocab = saved['feature_vocab']
model = saved['model']

def extract_feature_counter(G):
    features = Counter()
    features.update(graph_to_api_counts(G))
    features.update({f"ngram_{'_'.join(gram)}": v for gram, v in graph_to_ngram_features(G).items()})
    features.update(graph_to_edge_features(G))
    features.update(graph_to_entropy_features(G))
    return features

def predict(graph):
    counter = extract_feature_counter(graph)
    row = [[counter.get(tok, 0) for tok in vocab]]
    X = pd.DataFrame(row, columns=vocab)
    return lgbm_predict_proba(model, X)[0]

# One of the confirmed-correct agenttesla predictions from the held-out set
# (P=0.9273, true label=1/agenttesla) -- pick the first one from last
# session's test md5 list that ISN'T the misclassified sample.
TARGET_MD5 = 'e46305d2488f5b2f21ec03cc0484d1c8'  # true agenttesla, predicted 0.9273 correctly
path = f'data/training_reports/{TARGET_MD5}.json'
events = parse_cape_json(path)
G = build_behavior_graph(events)

orig_prob = predict(G)
print(f'Sample {TARGET_MD5}: original P(agenttesla) = {orig_prob:.4f}')

# Constrained (feasibility-checked) search
search = CounterfactualSearch(graph=G)
best_cost = None
best_cand = None
best_prob = None
checked = 0
for cand in search.propose():
    checked += 1
    if not validate_candidate(G, cand):
        continue
    edited = search._apply_candidate(cand)
    new_prob = predict(edited)
    if new_prob < 0.5:
        cost = candidate_cost(cand)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_cand = cand
            best_prob = new_prob

print(f'Checked {checked} candidates (feasibility-constrained)')
if best_cand is None:
    print('CONSTRAINED RESULT: no_flip_found')
else:
    print(f'CONSTRAINED RESULT: FLIP FOUND, cost={best_cost}, new_prob={best_prob:.4f}')
    print(f'Candidate: {best_cand}')
    for node_id in best_cand.get('delete_nodes', []):
        if node_id in G.nodes:
            print(f'  Deleted node {node_id}: api={G.nodes[node_id].get("api")}, entity_type={G.nodes[node_id].get("entity_type")}')