"""Tier-1 feasibility checker for structural validity."""
from typing import Dict

import networkx as nx

from src.counterfactual.substitutions import get_substitutes


def apply_candidate(G: nx.DiGraph, candidate: Dict) -> nx.DiGraph:
    """Return a copy of ``G`` with the candidate edits applied."""
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
            original_resources = list(G.nodes[node].get("resources", []) or []) if node in G.nodes else []
            from src.counterfactual.substitutions import update_resources_for_substitution

            G2.nodes[node]["resources"] = update_resources_for_substitution(api, original_resources)
    return G2


def candidate_cost(candidate: Dict) -> int:
    """Return a simple edit-distance proxy for ranking candidates."""
    delete_nodes = len(set(candidate.get("delete_nodes", []) or []))
    delete_edges = len(candidate.get("delete_edges", []) or [])
    substitutions = len((candidate.get("substitute", {}) or {}).keys())
    return delete_nodes + delete_edges + substitutions


def validate_candidate(G: nx.DiGraph, candidate: Dict) -> bool:
    """Validate candidate edits against dependency closure and substitution rules."""
    delete_nodes = set(candidate.get("delete_nodes", []))
    G2 = apply_candidate(G, candidate)
    if G2.number_of_nodes() == 0:
        return False

    meaningful = any((data.get("api") and data.get("api") != "unknown") for _, data in G2.nodes(data=True))
    if not meaningful:
        return False

    resource_roots = set()
    for node, data in G.nodes(data=True):
        for resource in data.get("resources", []) or []:
            has_original_producer = False
            for predecessor in G.predecessors(node):
                edge_data = G.get_edge_data(predecessor, node) or {}
                if edge_data.get("type") != "resource":
                    continue
                predecessor_resources = G.nodes[predecessor].get("resources", []) or []
                if resource in predecessor_resources:
                    has_original_producer = True
                    break
            if not has_original_producer:
                resource_roots.add((node, resource))

    for node, data in G2.nodes(data=True):
        for resource in data.get("resources", []) or []:
            has_producer = False
            for predecessor in G2.predecessors(node):
                edge_data = G2.get_edge_data(predecessor, node) or {}
                if edge_data.get("type") != "resource":
                    continue
                predecessor_resources = G2.nodes[predecessor].get("resources", []) or []
                if resource in predecessor_resources:
                    has_producer = True
                    break
            if has_producer or (node, resource) in resource_roots:
                continue
            return False

    for node, api in (candidate.get("substitute", {}) or {}).items():
        original_api = G.nodes[node].get("api") if node in G.nodes else None
        if original_api is None:
            return False
        if api not in get_substitutes(original_api):
            return False

    process_nodes = [node for node, data in G2.nodes(data=True) if data.get("entity_type") == "process"]
    if process_nodes:
        for node, data in G2.nodes(data=True):
            if data.get("entity_type") == "process":
                continue
            has_process_parent = any(
                G2.nodes[predecessor].get("entity_type") == "process"
                for predecessor in G2.predecessors(node)
            )
            if not has_process_parent:
                return False

    return True
