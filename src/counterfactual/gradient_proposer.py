"""Gradient-based proposer for GNN models (CF-style importance scores).

Computes node importance via gradients on a PyG GNN model's input features,
ranks nodes by L2 gradient norm, and returns deletion/substitution candidates
focused on the top-ranked nodes.
"""
from typing import Any, Dict, List

import networkx as nx
import torch

from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data


def node_importance_via_gradients(model: Any, G: nx.DiGraph, api_vocab: Any = None) -> List[float]:
    """Return a list of importance scores (one per node in G) using gradients."""
    vocab = api_vocab if api_vocab is not None else build_api_vocab([G])
    data = graph_to_pyg_data(G, vocab)
    x = data.x.clone().detach()
    x.requires_grad = True

    model.eval()
    with torch.enable_grad():
        out = model(x, data.edge_index)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        loss = torch.sigmoid(out).sum()
        loss.backward()
        grads = x.grad
        scores = grads.norm(p=2, dim=1).tolist()
    return scores


def propose_from_gradients(model: Any, G: nx.DiGraph, top_k: int = 10, api_vocab: Any = None) -> List[Dict]:
    """Produce candidate edits based on top-k gradient-ranked nodes.

    Returns a list of candidate dicts compatible with `search.CounterfactualSearch`.
    For now, candidates are single-node deletions for top-k nodes, plus pairwise
    deletions among the top `min(5, top_k)` nodes.
    """
    scores = node_importance_via_gradients(model, G, api_vocab=api_vocab)
    nodes = list(G.nodes())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cands = []
    for idx in ranked[:top_k]:
        n = nodes[idx]
        cands.append({"delete_nodes": [n], "substitute": {}})

    # pairwise combinations among top-5
    pair_count = min(5, top_k)
    top_nodes = [nodes[i] for i in ranked[:pair_count]]
    for i in range(len(top_nodes)):
        for j in range(i + 1, len(top_nodes)):
            cands.append({"delete_nodes": [top_nodes[i], top_nodes[j]], "substitute": {}})

    return cands
