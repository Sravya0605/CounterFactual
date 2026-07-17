"""Gradient-based proposer for GNN models (CF-style importance scores).

Computes node importance via gradients on a PyG GNN model's input features,
ranks nodes by L2 gradient norm, and returns deletion/substitution candidates
focused on the top-ranked nodes.
"""
from typing import List, Dict, Any
import torch
import networkx as nx
from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data


def node_importance_via_gradients(model: Any, G: nx.DiGraph) -> List[float]:
    """Return a list of importance scores (one per node in G) using gradients.

    Higher score = more important. Assumes `model` is a PyG-compatible
    torch `nn.Module` that maps node features/edges -> graph logit.
    """
    vocab = build_api_vocab([G])
    data = graph_to_pyg_data(G, vocab)
    x = data.x.clone().detach()
    x.requires_grad = True

    model.eval()
    with torch.enable_grad():
        out = model(x, data.edge_index)
        # if out is a scalar per graph, ensure shape
        if out.dim() == 0:
            out = out.unsqueeze(0)
        # take the graph logit (assumes single graph in data)
        logit = out
        loss = torch.sigmoid(logit).sum()
        loss.backward()
        grads = x.grad
        # importance = L2 norm per node
        scores = grads.norm(p=2, dim=1).tolist()
    return scores


def propose_from_gradients(model: Any, G: nx.DiGraph, top_k: int = 10) -> List[Dict]:
    """Produce candidate edits based on top-k gradient-ranked nodes.

    Returns a list of candidate dicts compatible with `search.CounterfactualSearch`.
    For now, candidates are single-node deletions for top-k nodes, plus pairwise
    deletions among the top `min(5, top_k)` nodes.
    """
    scores = node_importance_via_gradients(model, G)
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
