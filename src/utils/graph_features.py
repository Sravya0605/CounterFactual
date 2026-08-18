"""Utilities to convert a behavior graph into feature vectors."""
import math
from collections import Counter
from typing import Any, List
from datetime import datetime

import pandas as pd


def _node_api_sequence(G) -> List[str]:
    from src.utils.timestamps import normalize_timestamp

    def timestamp_key(nid):
        timestamps = G.nodes[nid].get("timestamps") or []
        if not timestamps:
            return float("inf")
        try:
            return normalize_timestamp(timestamps[0])
        except ValueError:
            # Explicit, visible fallback -- an unparseable timestamp sorts
            # this node last rather than crashing feature extraction, but
            # unlike the old version this is a deliberate, named decision
            # at one call site, not a silently duplicated implementation.
            return float("inf")

    nodes = sorted(G.nodes(), key=timestamp_key)
    return [str(G.nodes[nid].get("api") or "unknown") for nid in nodes]


def graph_to_api_counts(G) -> Counter:
    counts: Counter = Counter()
    for _, data in G.nodes(data=True):
        api = str(data.get("api") or "unknown").lower()
        counts[api] += int(data.get("count", 1))
    return counts

def graph_to_normalized_api_features(G) -> Counter:
    """Return each API's share of total API call volume, instead of a raw
    count. Raw counts let a single high-volume feature dominate a shallow
    classifier's decisions regardless of what fraction of behavior it
    actually represents (confirmed directly: 2026-08-18 findings --
    createtoolhelp32snapshot's raw count became the sole decision feature
    for an agenttesla-vs-qbot classifier, with no correctly-classified
    sample anywhere near its learned threshold). Normalizing by total
    volume gives the model a graded signal that isn't just "how active
    was this sample overall."
    """
    raw_counts = graph_to_api_counts(G)
    total = sum(raw_counts.values())
    normalized: Counter = Counter()
    if total == 0:
        return normalized
    for api, count in raw_counts.items():
        normalized[f"ratio_{api}"] = round(count / total, 6)
    return normalized


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
