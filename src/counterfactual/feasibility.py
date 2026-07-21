"""Tier-1 feasibility checker for structural validity.

The checker now validates resource-producer relationships more carefully and
rejects candidates that would leave a node without a viable upstream producer
or introduce an obviously inconsistent dependency closure.
"""
from typing import Dict

import networkx as nx

from src.counterfactual.substitutions import get_substitutes


def apply_candidate(G: nx.DiGraph, candidate: Dict) -> nx.DiGraph:
    """Return a copy of `G` with the candidate edits applied."""
    G2 = nx.DiGraph(G)
    delete_nodes = set(candidate.get("delete_nodes", []))
    substitutes = candidate.get("substitute", {}) or {}

    for node in delete_nodes:
        if node in G2:
            G2.remove_node(node)

    delete_edges = candidate.get("delete_edges", []) or []
    for edge in delete_edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            u, v = edge
            if G2.has_edge(u, v):
                G2.remove_edge(u, v)

    for node, api in substitutes.items():
        if node in G2:
            G2.nodes[node]["api"] = api
    return G2


def validate_candidate(G: nx.DiGraph, candidate: Dict) -> bool:
    """Validate candidate edits against dependency closure and substitution rules."""
    delete_nodes = set(candidate.get("delete_nodes", []))
    if not delete_nodes and not candidate.get("delete_edges") and not candidate.get("substitute"):
        return False

    G2 = apply_candidate(G, candidate)
    if G2.number_of_nodes() == 0:
        return False

    meaningful = any((data.get("api") and data.get("api") != "unknown") for _, data in G2.nodes(data=True))
    if not meaningful:
        return False

    for node, data in G2.nodes(data=True):
        resources = data.get("resources", []) or []
        if not resources:
            continue
        for resource in resources:
            has_producer = False
            for predecessor in G2.predecessors(node):
                edge_data = G2.get_edge_data(predecessor, node) or {}
                if edge_data.get("type") != "resource":
                    continue
                predecessor_resources = G2.nodes[predecessor].get("resources", []) or []
                if resource in predecessor_resources:
                    has_producer = True
                    break
            if not has_producer and resource not in (data.get("resources", []) or []):
                return False

    for node, api in (candidate.get("substitute", {}) or {}).items():
        original_api = G.nodes[node].get("api") if node in G.nodes else None
        if original_api is None:
            return False
        if api not in get_substitutes(original_api):
            return False

    return True
