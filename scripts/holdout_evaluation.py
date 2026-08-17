import csv
import os
import random
import warnings
from statistics import mean

warnings.filterwarnings('ignore')

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.feasibility import validate_candidate, candidate_cost

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

rows = [
    row for row in rows
    if row['avclass_family'].strip().lower() in {'emotet', 'agenttesla'}
]
print('SCOPE: this evaluation covers emotet vs agenttesla only. qbot excluded (smallest qbot sample 96.3 MB exceeds memory cap; requires separate evaluation with streaming JSON parsing, tracked as follow-up work).')

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

def run_single_seed(seed):
    seeded_rows = list(rows)
    random.Random(seed).shuffle(seeded_rows)
    split = int(len(seeded_rows) * 0.8)
    train_rows, test_rows = seeded_rows[:split], seeded_rows[split:]

    print('')
    print(f'SEED={seed}')
    print(f'Train: {len(train_rows)}, Held-out test: {len(test_rows)}')

    train_graphs, train_labels, _, train_skipped = load_graphs(train_rows)
    test_graphs, _, test_md5s, test_skipped = load_graphs(test_rows)
    print(f'SKIPPED TRAIN: {train_skipped}')
    print(f'SKIPPED TEST: {test_skipped}')

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

    records = []
    constrained_completed = 0
    for md5, graph in zip(test_md5s, test_graphs):
        prob = float(harness.predict_proba([graph])[0])
        search = CounterfactualSearch(classifier=harness, graph=graph)
        result = search.run()
        status = result.get('status')
        cand = result.get('candidate')
        cost = candidate_cost(cand) if cand is not None else None
        records.append({'md5': md5, 'candidate_cost': cost, 'orig_prob': prob, 'status': status})
        if status == 'completed':
            constrained_completed += 1

    # Keep unconstrained baseline in the pipeline, but do not print per-sample lines.
    for md5, graph in zip(test_md5s, test_graphs):
        search = CounterfactualSearch(classifier=harness, graph=graph)
        orig_prob = float(harness.predict_proba([graph])[0])
        if orig_prob < search.threshold:
            continue
        best_cost = None
        best_cand = None
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
        if best_cand is not None:
            validate_candidate(graph, best_cand)

    heldout_total_evaluated = len(test_md5s)
    if heldout_total_evaluated:
        feasibility_rate = (constrained_completed / heldout_total_evaluated) * 100
    else:
        feasibility_rate = 0.0

    print(f'heldout_total_evaluated={heldout_total_evaluated}')
    print(f'completed_flips={constrained_completed}')
    print(f'feasibility_rate={feasibility_rate:.2f}%')

    return {
        'seed': seed,
        'heldout_total_evaluated': heldout_total_evaluated,
        'completed_flips': constrained_completed,
        'feasibility_rate': feasibility_rate,
    }


seed_results = []
for seed in [0, 1, 2, 3, 4]:
    seed_results.append(run_single_seed(seed))

rates = [r['feasibility_rate'] for r in seed_results]
total_completed_flips = sum(r['completed_flips'] for r in seed_results)
total_heldout_evaluated = sum(r['heldout_total_evaluated'] for r in seed_results)

print('')
print('aggregate_summary')
print(f'feasibility_rate_min={min(rates):.2f}%')
print(f'feasibility_rate_max={max(rates):.2f}%')
print(f'feasibility_rate_mean={mean(rates):.2f}%')
print(f'total_completed_flips_all_seeds={total_completed_flips}')
print(f'total_heldout_total_evaluated_all_seeds={total_heldout_evaluated}')
