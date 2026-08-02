"""Simple evaluation metrics for counterfactual explanations."""
from typing import Any, Dict, List

import networkx as nx


def compute_metrics(original: nx.DiGraph, edited: nx.DiGraph, candidate: Dict[str, Any]) -> Dict[str, float]:
    """Compute a few cheap, interpretable metrics for a candidate edit."""
    deleted_nodes = len(set(candidate.get("delete_nodes", []) or []))
    deleted_edges = len(candidate.get("delete_edges", []) or [])
    substitutions = len((candidate.get("substitute", {}) or {}).keys())

    return {
        "edit_size": float(deleted_nodes + deleted_edges + substitutions),
        "nodes_removed": float(deleted_nodes),
        "edges_removed": float(deleted_edges),
        "substitutions": float(substitutions),
        "node_delta": float(edited.number_of_nodes() - original.number_of_nodes()),
        "edge_delta": float(edited.number_of_edges() - original.number_of_edges()),
    }


def detect_decoy_flips(
    original: nx.DiGraph,
    edited: nx.DiGraph,
    candidate: Dict[str, Any] | None = None,
    *,
    threshold: float = 0.5,
    orig_prob: float | None = None,
    new_prob: float | None = None,
) -> List[Dict[str, Any]]:
    """Flag single-node deletions that genuinely flip the classifier verdict."""
    if candidate is None:
        return []

    deleted_nodes = list(set(candidate.get("delete_nodes", []) or []))
    deleted_edges = len(candidate.get("delete_edges", []) or [])
    substitutions = len((candidate.get("substitute", {}) or {}).keys())
    edit_size = len(deleted_nodes) + deleted_edges + substitutions
    node_delta = edited.number_of_nodes() - original.number_of_nodes()

    if edit_size != 1 or node_delta != -1:
        return []

    if orig_prob is None or new_prob is None:
        return []

    if orig_prob < threshold or new_prob >= threshold:
        return []

    return [{"node": deleted_nodes[0], "reason": "single_node_flip"}]
