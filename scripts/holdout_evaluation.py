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

# Use the full training batch (streamed) and process one report at a time.
CSV_PATH = 'data/training_batch_redline_vbclone.csv'
SOURCE_CSV_PATH = 'data/training_batch.csv'
POSITIVE_FAMILY = 'redline'
reports_dir = 'data/training_reports'


def ensure_pair_csv():
    if os.path.exists(CSV_PATH):
        return

    with open(SOURCE_CSV_PATH, newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))

    pair_families = {'redline', 'vbclone'}
    pair_rows = [
        row for row in rows
        if row.get('avclass_family', '').strip().lower() in pair_families
    ]
    if not pair_rows:
        raise FileNotFoundError(
            f'No redline/vbclone rows found in {SOURCE_CSV_PATH}'
        )

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)
    print(f'Created {CSV_PATH} from {SOURCE_CSV_PATH} ({len(pair_rows)} rows)')


def stream_graphs(rows):
    """Yield one parsed graph and metadata at a time to avoid large memory usage."""
    for row in rows:
        md5 = row['md5']
        family = row['avclass_family'].strip()
        path = os.path.join(reports_dir, f'{md5}.json')
        if not os.path.exists(path):
            yield None, None, md5, family, f'file missing: {path}'
            continue
        try:
            events = parse_cape_json(path)
            G = build_behavior_graph(events)
            yield G, (1 if family.lower() == POSITIVE_FAMILY else 0), md5, family, None
        except Exception as exc:
            yield None, None, md5, family, f'parse error: {exc}'

def minimality_check(candidate, graph, harness):
    """Simple minimality check: returns True if no strict subset of the
    candidate's edits also flips the classifier. This is conservative and
    only checks single-node subsets for efficiency.
    """
    if candidate is None:
        return False
    base_cost = candidate_cost(candidate)
    # Check single-node deletions
    del_nodes = candidate.get("delete_nodes", []) or []
    for n in del_nodes:
        sub = {"delete_nodes": [n], "substitute": {}}
        if not validate_candidate(graph, sub):
            continue
        edited = CounterfactualSearch(graph=graph)._apply_candidate(sub)
        new_prob = float(harness.predict_proba([edited])[0])
        if new_prob < CounterfactualSearch(graph=graph).threshold:
            return False
    # Check single substitutions
    subs = (candidate.get("substitute", {}) or {}).items()
    for node, api in subs:
        sub = {"delete_nodes": [], "substitute": {node: api}}
        if not validate_candidate(graph, sub):
            continue
        edited = CounterfactualSearch(graph=graph)._apply_candidate(sub)
        new_prob = float(harness.predict_proba([edited])[0])
        if new_prob < CounterfactualSearch(graph=graph).threshold:
            return False
    return True


def run_single_seed(seed):
    # Stream CSV rows rather than loading everything into memory.
    ensure_pair_csv()
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    random.Random(seed).shuffle(all_rows)
    split = int(len(all_rows) * 0.8)
    train_rows, test_rows = all_rows[:split], all_rows[split:]

    print('')
    print(f'SEED={seed}')
    print(f'Train: {len(train_rows)}, Held-out test: {len(test_rows)}')

    # Train on streamed graphs (but here we still collect feature counters
    # similar to the full-train script). For simplicity reuse ClassifierHarness
    # training path: parse train graphs one at a time and accumulate.
    train_graphs = []
    train_labels = []
    for G, label, md5, family, err in stream_graphs(train_rows):
        if err or G is None:
            continue
        train_graphs.append(G)
        train_labels.append(label)

    harness = ClassifierHarness(backend='lgbm', model_path='models/redline_vbclone_holdout_lgbm.pkl')
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
    unconstrained_completed = 0
    # Evaluate on test set streaming one graph at a time.
    for G, label, md5, family, err in stream_graphs(test_rows):
        if err or G is None:
            records.append({'md5': md5, 'candidate_cost': None, 'orig_prob': None, 'status': 'skipped', 'error': err})
            continue
        prob = float(harness.predict_proba([G])[0])
        search = CounterfactualSearch(graph=G)
        # Generate first; classifier scoring remains an explicit outer step.
        result = search.generate()
        status = result.get('status')
        cand = result.get('candidate')
        cost = candidate_cost(cand) if cand is not None else None
        is_minimal = minimality_check(cand, G, harness)
        records.append({'md5': md5, 'candidate_cost': cost, 'orig_prob': prob, 'status': status, 'is_minimal': is_minimal})
        if status == 'completed' and cand is not None:
            edited = search._apply_candidate(cand)
            new_prob = float(harness.predict_proba([edited])[0])
            records[-1]['new_prob'] = new_prob
            if prob >= search.threshold and new_prob < search.threshold:
                constrained_completed += 1

        # Unconstrained baseline: same proposer but no feasibility checks.
        search_uncon = CounterfactualSearch(graph=G)
        search_uncon.enforce_feasibility = False
        best_cost = None
        best_cand = None
        for i, cand in enumerate(search_uncon.propose()):
            if i >= 30:
                break
            edited = search_uncon._apply_candidate(cand)
            new_prob = float(harness.predict_proba([edited])[0])
            if new_prob < search_uncon.threshold:
                cost = candidate_cost(cand)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_cand = cand
        if best_cand is not None:
            unconstrained_completed += 1

    heldout_total_evaluated = len([r for r in records if r.get('status') != 'skipped'])
    feasibility_rate = (constrained_completed / heldout_total_evaluated) * 100 if heldout_total_evaluated else 0.0

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
