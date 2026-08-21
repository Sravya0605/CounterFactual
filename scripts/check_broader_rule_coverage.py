import csv
import os

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph

REPORTS_DIR = 'data/training_reports'
RESULTS_PATH = 'data/held_out_search_results.csv'

BROADER_FAMILY = {
    'createtoolhelp32snapshot', 'process32first', 'process32next',
    'module32first', 'module32next', 'getmodulehandlea', 'getprocaddress',
    'virtualqueryex', 'readprocessmemory',
}

with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
    failing = [r['md5'] for r in csv.DictReader(f) if r['status'] == 'no_flip_found']

print(f'Checking {len(failing)} previously-failing samples against the broader rule...')
print('')

covered = 0
for md5 in failing:
    path = os.path.join(REPORTS_DIR, f'{md5}.json')
    events = parse_cape_json(path)
    G = build_behavior_graph(events)

    process_has_family_call = {}
    for _, data in G.nodes(data=True):
        pid = data.get('process_id')
        api = str(data.get('api') or '').lower()
        if pid is not None and api in BROADER_FAMILY:
            process_has_family_call[pid] = process_has_family_call.get(pid, set())
            process_has_family_call[pid].add(api)

    is_covered = len(process_has_family_call) > 0
    if is_covered:
        covered += 1
    apis_seen = set()
    for s in process_has_family_call.values():
        apis_seen |= s
    print(f'{md5:<34} covered={is_covered!s:<5} apis_seen={sorted(apis_seen)}')

print('')
print(f'{covered}/{len(failing)} previously-failing samples would now generate at least one insertion candidate under the broader rule')