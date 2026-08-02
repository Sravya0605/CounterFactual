"""Utilities to convert a behavior graph into feature vectors."""
import math
from collections import Counter
from typing import Any, List

import pandas as pd


def _node_api_sequence(G) -> List[str]:
    nodes = sorted(G.nodes(), key=lambda nid: (G.nodes[nid].get("timestamps") or [0])[0])
    return [str(G.nodes[nid].get("api") or "unknown") for nid in nodes]


def graph_to_api_counts(G) -> Counter:
    counts: Counter = Counter()
    for _, data in G.nodes(data=True):
        api = str(data.get("api") or "unknown").lower()
        counts[api] += int(data.get("count", 1))
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


def _shannon_entropy(values: List[Any]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def graph_to_entropy_features(G) -> Counter:
    counts: Counter = Counter()
    entity_types = [str(data.get("entity_type", "unknown")) for _, data in G.nodes(data=True)]
    attack_ids = [str(data.get("attack_id")) for _, data in G.nodes(data=True) if data.get("attack_id")]
    counts["entropy_entity_type"] = round(_shannon_entropy(entity_types), 4)
    counts["entropy_attack_id"] = round(_shannon_entropy(attack_ids), 4)
    return counts


def build_feature_vocab(graphs: List[Any]) -> List[str]:
    vocab = set()
    for G in graphs:
        features = Counter()
        features.update(graph_to_api_counts(G))
        features.update({f"ngram_{'_'.join(gram)}": value for gram, value in graph_to_ngram_features(G).items()})
        features.update(graph_to_edge_features(G))
        features.update(graph_to_entropy_features(G))
        vocab.update(features.keys())
    return sorted(vocab)


def graph_list_to_bow(graphs: List[Any], vocab: List[str] = None) -> pd.DataFrame:
    vocab = list(vocab) if vocab is not None else build_feature_vocab(graphs)
    feature_rows = []
    for G in graphs:
        features = Counter()
        features.update(graph_to_api_counts(G))
        features.update({f"ngram_{'_'.join(gram)}": value for gram, value in graph_to_ngram_features(G).items()})
        features.update(graph_to_edge_features(G))
        features.update(graph_to_entropy_features(G))
        feature_rows.append(features)

    rows = []
    for features in feature_rows:
        rows.append([features.get(token, 0) for token in vocab])
    return pd.DataFrame(rows, columns=vocab)
