"""Utilities to convert a behavior graph into feature vectors.

The baseline now includes API counts, simple n-gram features over the event
sequence, edge-type counts, and a small entropy signal so it is more faithful
to the design document than a pure bag-of-APIs representation.
"""
from collections import Counter
from typing import Any, List

import pandas as pd


def _node_api_sequence(G) -> List[str]:
    nodes = sorted(G.nodes(), key=lambda nid: (G.nodes[nid].get("timestamps") or [0])[0])
    return [str(G.nodes[nid].get("api") or "unknown") for nid in nodes]


def graph_to_api_counts(G) -> Counter:
    counts: Counter = Counter()
    for _, data in G.nodes(data=True):
        api = data.get("api") or "unknown"
        counts[api] += int(data.get("count", 1))
        counts[api.lower()] += int(data.get("count", 1))
    return counts


def graph_to_ngram_features(G, n: int = 2) -> Counter:
    seq = _node_api_sequence(G)
    grams: Counter = Counter()
    for i in range(len(seq) - n + 1):
        gram = tuple(part.lower() for part in seq[i : i + n])
        grams[gram] += 1
    return grams


def graph_to_edge_features(G) -> Counter:
    counts: Counter = Counter()
    for _, _, data in G.edges(data=True):
        edge_type = data.get("type", "unknown")
        counts[f"edge_{edge_type}"] += 1
    return counts


def graph_to_entropy_features(G) -> Counter:
    counts: Counter = Counter()
    for _, data in G.nodes(data=True):
        api = str(data.get("api") or "unknown")
        counts[f"entity_{data.get('entity_type', 'unknown')}"] += 1
        if data.get("attack_id"):
            counts[f"attack_{data.get('attack_id')}"] += 1
    return counts


def graph_list_to_bow(graphs: List[Any]) -> pd.DataFrame:
    vocab = set()
    feature_rows = []
    for G in graphs:
        features = Counter()
        features.update(graph_to_api_counts(G))
        features.update({f"ngram_{'_'.join(gram)}": value for gram, value in graph_to_ngram_features(G).items()})
        features.update(graph_to_edge_features(G))
        features.update(graph_to_entropy_features(G))
        feature_rows.append(features)
        vocab.update(features.keys())

    vocab = sorted(vocab)
    rows = []
    for features in feature_rows:
        rows.append([features.get(token, 0) for token in vocab])
    return pd.DataFrame(rows, columns=vocab)
