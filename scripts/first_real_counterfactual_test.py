from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch

TEST_MD5 = '951df57c534b6acaeecced8cc6e898c7'
path = f'data/training_reports/{TEST_MD5}.json'

events = parse_cape_json(path)
G = build_behavior_graph(events)
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

harness = ClassifierHarness(backend='lgbm', model_path='models/emotet_binary_lgbm.pkl')
prob = harness.predict_proba([G])[0]
print(f'Original P(emotet) = {prob:.4f}')

search = CounterfactualSearch(classifier=harness, graph=G)
result = search.run()

print('')
print('--- SEARCH RESULT ---')
print('status:', result.get('status'))
print('candidate:', result.get('candidate'))

if result.get('edited_graph') is not None:
    edited = result['edited_graph']
    new_prob = harness.predict_proba([edited])[0]
    print(f'New P(emotet) after edit = {new_prob:.4f}')