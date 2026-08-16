import csv
import os
import random
import warnings

warnings.filterwarnings('ignore')

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.feasibility import validate_candidate, candidate_cost, apply_candidate

random.seed(42)

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

rows = [
    row for row in rows
    if row['avclass_family'].strip().lower() in {'emotet', 'agenttesla'}
]
print('SCOPE: this evaluation covers emotet vs agenttesla only. qbot excluded (smallest qbot sample 96.3 MB exceeds memory cap; requires separate evaluation with streaming JSON parsing, tracked as follow-up work).')

random.shuffle(rows)
split = int(len(rows) * 0.8)
train_rows, test_rows = rows[:split], rows[split:]
print(f'Train: {len(train_rows)}, Held-out test: {len(test_rows)}')

reports_dir = 'data/training_reports'


def load_graphs(rows):
    graphs, labels, md5s = [], [], []
    skipped = 0
    for row in rows:
        md5 = row['md5']
        family = row['avclass_family'].strip()
        path = os.path.join(reports_dir, f'{md5}.json')
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > 60:
            print(f'SKIPPED {md5} ({family}): {size_mb:.1f} MB exceeds 60 MB cap')
            skipped += 1
            continue
        events = parse_cape_json(path)
        G = build_behavior_graph(events)
        graphs.append(G)
        labels.append(1 if family.lower() == 'emotet' else 0)
        md5s.append(md5)
    return graphs, labels, md5s, skipped

train_graphs, train_labels, _, train_skipped = load_graphs(train_rows)
test_graphs, test_labels, test_md5s, test_skipped = load_graphs(test_rows)
print(f'SKIPPED TRAIN: {train_skipped}')
print(f'SKIPPED TEST: {test_skipped}')


def survivors_after_cap(rows):
    counts = {'emotet': 0, 'agenttesla': 0}
    for row in rows:
        md5 = row['md5']
        family = row['avclass_family'].strip().lower()
        path = os.path.join(reports_dir, f'{md5}.json')
        if family not in counts:
            continue
        if os.path.getsize(path) <= 60 * 1024 * 1024:
            counts[family] += 1
    return counts


print('FAMILY COUNTS AFTER SIZE FILTERING (TRAIN):')
train_survivors = survivors_after_cap(train_rows)
for family in ['emotet', 'agenttesla']:
    print(f'  {family}: {train_survivors[family]}')
    if train_survivors[family] < 8:
        print(f'WARNING: {family} has only {train_survivors[family]} samples after size filtering, class balance may be skewed')

print('FAMILY COUNTS AFTER SIZE FILTERING (TEST):')
test_survivors = survivors_after_cap(test_rows)
for family in ['emotet', 'agenttesla']:
    print(f'  {family}: {test_survivors[family]}')

harness = ClassifierHarness(backend='lgbm', model_path='models/emotet_binary_holdout_lgbm.pkl')
harness.train(
    train_graphs,
    train_labels,
    params={
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 4,
        'max_depth': 2,
        'min_data_in_leaf': 5,
        'learning_rate': 0.05,
    },
    rounds=15,
)


# Bounded minimality check: no combinatorial subset explosions.
def minimality_check(graph, candidate, search):
    ops = []
    for node in candidate.get('delete_nodes', []) or []:
        ops.append(('delete_node', node))
    for edge in candidate.get('delete_edges', []) or []:
        edge_tuple = tuple(edge) if isinstance(edge, (list, tuple)) else edge
        ops.append(('delete_edge', edge_tuple))
    for key, val in (candidate.get('substitute', {}) or {}).items():
        ops.append(('substitute', key, val))

    if len(ops) <= 1:
        return True

    if len(ops) == 2:
        for op in ops:
            sub = {'delete_nodes': [], 'delete_edges': [], 'substitute': {}}
            kind = op[0]
            if kind == 'delete_node':
                sub['delete_nodes'] = [op[1]]
            elif kind == 'delete_edge':
                sub['delete_edges'] = [list(op[1])]
            elif kind == 'substitute':
                sub['substitute'][op[1]] = op[2]
            if validate_candidate(graph, sub):
                edited = apply_candidate(graph, sub)
                if float(harness.predict_proba([edited])[0]) < search.threshold:
                    return False
        return True

    # 3+ operations: intentionally not checked because combinatorial testing
    # is too expensive and hangs the held-out evaluation.
    return None


print('')
print('sample|orig_prob|status|candidate_cost|new_prob|feasible|minimal')
records = []
constrained_completed = 0
for md5, label, graph in zip(test_md5s, test_labels, test_graphs):
    prob = float(harness.predict_proba([graph])[0])
    search = CounterfactualSearch(classifier=harness, graph=graph)
    result = search.run()
    status = result.get('status')
    cand = result.get('candidate')
    if cand is None:
        records.append({'md5': md5, 'orig_prob': prob, 'status': status, 'candidate_cost': None, 'new_prob': None, 'feasible': None, 'minimal': None})
        print(f'{md5}|{prob:.4f}|{status}|NA|NA|NA|None')
        continue
    edited = search._apply_candidate(cand)
    new_prob = float(harness.predict_proba([edited])[0])
    cost = candidate_cost(cand)
    feasible = bool(validate_candidate(graph, cand))
    minimal = minimality_check(graph, cand, search) if feasible else False
    if status == 'completed':
        constrained_completed += 1
    records.append({'md5': md5, 'orig_prob': prob, 'status': status, 'candidate_cost': cost, 'new_prob': new_prob, 'feasible': feasible, 'minimal': minimal})
    print(f'{md5}|{prob:.4f}|{status}|{cost}|{new_prob:.4f}|{feasible}|{minimal}')

print('')
print('sample|unconstrained_cost|unconstrained_new_prob|unconstrained_feasible|constrained_cost|smaller_than_constrained')
base_records = []
base_infeasible = 0
count_smaller = 0
for md5, graph in zip(test_md5s, test_graphs):
    search = CounterfactualSearch(classifier=harness, graph=graph)
    best_cost = None
    best_cand = None
    best_new_prob = None
    orig_prob = float(harness.predict_proba([graph])[0])
    constrained_cost = next((r['candidate_cost'] for r in records if r['md5'] == md5), None)
    if orig_prob < search.threshold:
        print(f'{md5}|SKIPPED (orig already not_malicious)|NA|NA|{constrained_cost}|NA')
        base_records.append({'md5': md5, 'unconstrained_cost': None, 'new_prob': None, 'infeasible': None, 'smaller': None, 'constrained_cost': constrained_cost})
        continue
    for i, cand in enumerate(search.propose()):
        if i >= 30:
            break
        edited = search._apply_candidate(cand)
        new_prob = float(harness.predict_proba([edited])[0])
        if new_prob < search.threshold:
            cost = candidate_cost(cand)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_cand = cand
                best_new_prob = new_prob
    if best_cand is None:
        print(f'{md5}|NA|NA|NA|{constrained_cost}|NA')
        base_records.append({'md5': md5, 'unconstrained_cost': None, 'new_prob': None, 'infeasible': None, 'smaller': None, 'constrained_cost': constrained_cost})
        continue
    feasible = validate_candidate(graph, best_cand)
    smaller = constrained_cost is not None and best_cost < constrained_cost
    if feasible is False:
        base_infeasible += 1
    if smaller:
        count_smaller += 1
    base_records.append({'md5': md5, 'unconstrained_cost': best_cost, 'new_prob': best_new_prob, 'infeasible': not feasible, 'smaller': smaller, 'constrained_cost': constrained_cost})
    print(f'{md5}|{best_cost}|{best_new_prob:.4f}|{not feasible}|{constrained_cost}|{smaller}')

print('')
print('summary')
constrained_costs = [r['candidate_cost'] for r in records if r['status'] == 'completed' and r['candidate_cost'] is not None]
base_costs = [r['unconstrained_cost'] for r in base_records if r['unconstrained_cost'] is not None]
heldout_total_before_size_filter = len(test_rows)
heldout_total_evaluated = len(test_md5s)
print(f'heldout_total_before_size_filter={heldout_total_before_size_filter}')
print(f'heldout_total_evaluated={heldout_total_evaluated}')
print(f'completed_flips={constrained_completed}')
if heldout_total_evaluated:
    print(f'feasibility_rate={(constrained_completed / heldout_total_evaluated) * 100:.2f}%')
else:
    print('feasibility_rate=NA')
if constrained_costs:
    print(f'mean_edit_size_constrained={sum(constrained_costs) / len(constrained_costs):.4f}')
else:
    print('mean_edit_size_constrained=NA')
if base_costs:
    print(f'mean_edit_size_unconstrained={sum(base_costs) / len(base_costs):.4f}')
else:
    print('mean_edit_size_unconstrained=NA')
if base_costs:
    print(f'percent_unconstrained_flips_infeasible={(base_infeasible / len(base_costs) * 100):.2f}%')
else:
    print('percent_unconstrained_flips_infeasible=NA')
print(f'count_unconstrained_cheaper_than_constrained={count_smaller}')
print('note=unconstrained baseline checked first 30 candidates only, not full search space.')
