"""Edge-mask (CF-style) proposer using gradient signals.

This module ranks edges by the importance of their endpoint nodes' gradients
and returns candidates that delete high-scoring edges. It reuses
`node_importance_via_gradients` to compute node-level scores.
"""
from typing import List, Dict, Any, Tuple
import networkx as nx
from src.counterfactual.gradient_proposer import node_importance_via_gradients


def propose_edge_deletions(model: Any, G: nx.DiGraph, top_k: int = 10) -> List[Dict]:
    """Return a list of candidate dicts that delete top-k important edges.

    Each candidate has the form {"delete_edges": [(u,v), ...], "delete_nodes": [], "substitute": {}}
    """
    scores = node_importance_via_gradients(model, G)
    nodes = list(G.nodes())
    edge_scores: List[Tuple[Tuple[str,str], float]] = []
    for u, v in G.edges():
        try:
            iu = nodes.index(u)
            iv = nodes.index(v)
            s = float(scores[iu] + scores[iv])
        except ValueError:
            s = 0.0
        edge_scores.append(((u, v), s))

    edge_scores.sort(key=lambda x: x[1], reverse=True)
    cands = []
    for (u, v), _ in edge_scores[:top_k]:
        cands.append({"delete_edges": [(u, v)], "delete_nodes": [], "substitute": {}})
    return cands
