"""Utilities to convert a coalesced behavior graph into feature vectors.

Current simple representation: bag-of-API counts (per-graph). This is a
lightweight baseline feature extractor for LightGBM training and experiments.
"""
from typing import List, Any
import pandas as pd
from collections import Counter


def graph_to_api_counts(G) -> Counter:
    c = Counter()
    for n, d in G.nodes(data=True):
        api = d.get("api") or "unknown"
        c[api] += int(d.get("count", 1))
    return c


def graph_list_to_bow(graphs: List[Any]) -> pd.DataFrame:
    # build vocabulary
    vocab = set()
    counts = []
    for G in graphs:
        c = graph_to_api_counts(G)
        counts.append(c)
        vocab.update(c.keys())
    vocab = sorted(vocab)
    rows = []
    for c in counts:
        rows.append([c.get(tok, 0) for tok in vocab])
    df = pd.DataFrame(rows, columns=vocab)
    return df
