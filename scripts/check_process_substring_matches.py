from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.counterfactual.insertions import is_anchor_plausible

MD5 = '83a3340793cec4cb51526c5f4d6711b9'

events = parse_cape_json(f'data/training_reports/{MD5}.json')
G = build_behavior_graph(events)

# Find every DISTINCT real API name in this graph that contains "process"
# as a substring (case-insensitive) -- i.e. everything the PR's loose
# ANCHOR_FAMILY["process"] token would match, using real data instead of guesses.
matching_apis = set()
for _, data in G.nodes(data=True):
    api = data.get('api', '') or ''
    if 'process' in api.lower():
        matching_apis.add(api)

print(f'Distinct real APIs in this graph containing "process" as a substring: {len(matching_apis)}')
for api in sorted(matching_apis):
    print(f'  {api!r}')

print('')
print('For each, does it justify inserting CreateRemoteThread per the PR\'s rule?')
false_positives = 0
for _, data in G.nodes(data=True):
    api = data.get('api', '') or ''
    if 'process' in api.lower():
        node_id = [n for n, d in G.nodes(data=True) if d is data][0]
        plausible = is_anchor_plausible(G, node_id, 'CreateRemoteThread')
        if plausible:
            false_positives += 1

print(f'{false_positives} real nodes in this one sample justify a CreateRemoteThread insertion via this rule')