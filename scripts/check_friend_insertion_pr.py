from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.insertions import is_anchor_plausible, ANCHOR_FAMILY

MD5 = '83a3340793cec4cb51526c5f4d6711b9'  # our known reference sample

events = parse_cape_json(f'data/training_reports/{MD5}.json')
G = build_behavior_graph(events)

print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
print('')

# --- Check 1: does propose() exceed max_candidates=200? ---
search = CounterfactualSearch(classifier=None, graph=G)
candidates = search.propose()
print(f'Check 1 -- propose() returned {len(candidates)} candidates (max_candidates={search.max_candidates})')
print(f'  VIOLATION: {len(candidates) > search.max_candidates}')
print('')

# --- Check 2: does the loose substring match over-approve real insertions? ---
suspect_apis = ['GetCurrentProcess', 'TerminateProcess', 'GetCurrentProcessId', 'NtQueryInformationProcess']
false_positive_count = 0
checked = 0
for node, data in G.nodes(data=True):
    api = data.get('api', '') or ''
    if any(sus.lower() == api.lower() for sus in suspect_apis):
        checked += 1
        # would this node justify inserting CreateRemoteThread, per the PR's own rule?
        plausible = is_anchor_plausible(G, node, 'CreateRemoteThread')
        if plausible:
            false_positive_count += 1
            print(f'  FALSE POSITIVE: node with api={api!r} justifies inserting CreateRemoteThread (via ANCHOR_FAMILY substring match)')

print('')
print(f'Check 2 -- checked {checked} real nodes with commonplace, non-injection-related process APIs')
print(f'  {false_positive_count}/{checked} incorrectly justify a CreateRemoteThread insertion')
print('')
print('ANCHOR_FAMILY tokens in use:', list(ANCHOR_FAMILY.keys()))