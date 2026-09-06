from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.counterfactual.search import CounterfactualSearch
import os

sample = os.path.join('tests', 'sample_cape.json')
evts = parse_cape_json(sample)
G = build_behavior_graph(evts)
print('nodes', G.number_of_nodes(), 'edges', G.number_of_edges())
for n, d in list(G.nodes(data=True)):
    if d.get('resources'):
        print('node', n, 'api', d.get('api'), 'resources', d.get('resources'))
print('--- candidates ---')
cs = CounterfactualSearch(graph=G)
for i, c in enumerate(cs.propose()[:10]):
    ok = cs.validate(c)
    print(i, c, ok)
    if ok:
        break
print('generated', cs.generate())
