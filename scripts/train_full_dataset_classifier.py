import csv
import gc
import os
import pickle
import random

import pandas as pd
import psutil

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.utils.graph_features import (
    graph_to_api_counts,
    graph_to_ngram_features,
    graph_to_edge_features,
    graph_to_entropy_features,
)
from src.classifier.lgbm_model import train_lgbm, predict_proba as lgbm_predict_proba

REPORTS_DIR = 'data/training_reports'
CSV_PATH = 'data/training_batch.csv'
MODEL_OUT_PATH = 'models/full_dataset_emotet_binary_lgbm.pkl'
SPLIT_SEED = 0  # matches holdout_evaluation.py's first seed, for a fair comparison
# Family framing for this run. Set via env var so the same script/pipeline
# can be reused for different binary framings without duplicating code.
# 'emotet' -> emotet=1 vs {agenttesla,qbot}=0 (original step 3/4 framing)
# 'agenttesla_vs_qbot' -> agenttesla=1 vs qbot=0, emotet excluded entirely
FRAMING = os.environ.get('FRAMING', 'emotet')
if FRAMING == 'agenttesla_vs_qbot':
    MODEL_OUT_PATH = 'models/full_dataset_agenttesla_vs_qbot_lgbm.pkl'
# No size cap: unlike holdout_evaluation.py's 60MB cutoff, this script is
# meant to include every sample regardless of file size, since it never
# holds more than one parsed graph in memory at a time.

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 3,
    'learning_rate': 0.05,
    'lambda_l2': 1.0,
}
LGBM_ROUNDS = 50

process = psutil.Process(os.getpid())


def rss_mb():
    return process.memory_info().rss / (1024 * 1024)


def extract_feature_counter(G):
    from collections import Counter
    features = Counter()
    features.update(graph_to_api_counts(G))
    features.update({f"ngram_{'_'.join(gram)}": v for gram, v in graph_to_ngram_features(G).items()})
    features.update(graph_to_edge_features(G))
    features.update(graph_to_entropy_features(G))
    return features


def stream_build_features(rows):
    """Process one CAPE report at a time. Only small per-graph feature
    Counters are accumulated -- never a list of full graph objects."""
    feature_counters = []
    labels = []
    md5s = []
    skipped = []

    for i, row in enumerate(rows, 1):
        md5 = row['md5']
        family = row['avclass_family'].strip().lower()
        path = os.path.join(REPORTS_DIR, f'{md5}.json')

        if not os.path.exists(path):
            skipped.append((md5, 'file not present in this environment'))
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        events = parse_cape_json(path)
        G = build_behavior_graph(events)
        counter = extract_feature_counter(G)

        feature_counters.append(counter)
        labels.append(family)
        md5s.append(md5)

        del events, G, counter
        gc.collect()

        processed_count = len(feature_counters)
        # Print every sample's RSS here (this sandbox only has a handful of
        # files present, so "every 10" would give zero visibility); on the
        # full 90-file run this still satisfies "at least every 10 samples".
        print(f'[row {i}/{len(rows)}, processed #{processed_count}] {md5} ({family}, {size_mb:.1f} MB) -- RSS={rss_mb():.1f} MB')

    return feature_counters, labels, md5s, skipped


def build_matrix(feature_counters):
    vocab = sorted(set().union(*[c.keys() for c in feature_counters])) if feature_counters else []
    rows = [[c.get(tok, 0) for tok in vocab] for c in feature_counters]
    return pd.DataFrame(rows, columns=vocab), vocab


if __name__ == '__main__':
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    print(f'Total rows in {CSV_PATH}: {len(all_rows)}')
    print(f'Starting RSS: {rss_mb():.1f} MB')
    print('')

    feature_counters, labels, md5s, skipped = stream_build_features(all_rows)

    print('')
    print(f'Processed: {len(feature_counters)}')
    print(f'Skipped: {len(skipped)}')
    for md5, reason in skipped:
        print(f'  SKIPPED {md5}: {reason}')
    print(f'Final RSS: {rss_mb():.1f} MB')
    print(f'Peak RSS since process start: not available on Windows (resource module is Unix-only); using psutil peak instead where tracked separately if needed')
    
    # ---- Step 3: binary reframe, 80/20 split ----
    if FRAMING == 'agenttesla_vs_qbot':
        keep = [i for i, fam in enumerate(labels) if fam in ('agenttesla', 'qbot')]
        print(f'FRAMING=agenttesla_vs_qbot: excluding emotet entirely, keeping {len(keep)}/{len(labels)} samples')
        feature_counters = [feature_counters[i] for i in keep]
        binary_labels = [1 if labels[i] == 'agenttesla' else 0 for i in keep]
        md5s = [md5s[i] for i in keep]
    else:
        binary_labels = [1 if fam == 'emotet' else 0 for fam in labels]

    indices = list(range(len(feature_counters)))
    random.Random(SPLIT_SEED).shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    train_counters = [feature_counters[i] for i in train_idx]
    train_labels = [binary_labels[i] for i in train_idx]
    test_counters = [feature_counters[i] for i in test_idx]
    test_labels = [binary_labels[i] for i in test_idx]
    test_md5s = [md5s[i] for i in test_idx]

    print('')
    print(f'Train: {len(train_idx)}, Held-out test: {len(test_idx)}')
    print(f'Train positive count: {sum(train_labels)}')
    print(f'Test positive count: {sum(test_labels)}')

    vocab = sorted(set().union(*[c.keys() for c in train_counters]))
    print(f'Vocab size (from train set only): {len(vocab)}')

    X_train = pd.DataFrame([[c.get(tok, 0) for tok in vocab] for c in train_counters], columns=vocab)
    X_test = pd.DataFrame([[c.get(tok, 0) for tok in vocab] for c in test_counters], columns=vocab)

    print('')
    print(f'LGBM params: {LGBM_PARAMS}, rounds={LGBM_ROUNDS}')
    model = train_lgbm(X_train, train_labels, params=LGBM_PARAMS, num_boost_round=LGBM_ROUNDS)

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    with open(MODEL_OUT_PATH, 'wb') as f:
        pickle.dump({'backend': 'lgbm', 'model': model, 'feature_vocab': vocab, 'api_vocab': None}, f)
    print(f'Model saved to {MODEL_OUT_PATH}')

    # ---- Step 4: diagnostic probability spread on held-out test set ----
    probs = lgbm_predict_proba(model, X_test)
    print('')
    print('Test probabilities:', [round(p, 4) for p in probs])
    print('Unique values:', len(set(round(p, 4) for p in probs)))
    print('Test md5s (same order):', test_md5s)
    print('Test true labels (same order):', test_labels)