"""Tier-1 feasibility checker: structural validity for candidate edits.

This module provides lightweight, conservative checks that a proposed graph
edit (node deletions and node substitutions) does not create obvious
dependency-orphaning under the heuristic `resource` edges produced by the
graph builder.

The checks are intentionally simple: they flag as invalid any candidate that
removes a source node for a `resource` edge while leaving the target node
in place (an orphaned dependency). More sophisticated checks (taint-based
verification) belong in Tier-2 and are out of scope for this module.
"""
from typing import Dict, List
import networkx as nx
from src.counterfactual.substitutions import get_substitutes


def apply_candidate(G: nx.DiGraph, candidate: Dict) -> nx.DiGraph:
    """Return a copy of `G` with the candidate edits applied.

    Candidate format:
      {"delete_nodes": [node_id, ...],
       "substitute": {node_id: substitute_api, ...}}
    """
    G2 = nx.DiGraph(G)
    delete_nodes = set(candidate.get("delete_nodes", []))
    substitutes = candidate.get("substitute", {}) or {}

    # apply deletions
    for n in delete_nodes:
        if n in G2:
            G2.remove_node(n)

    # apply edge deletions
    delete_edges = candidate.get("delete_edges", []) or []
    for e in delete_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            u, v = e
            if G2.has_edge(u, v):
                G2.remove_edge(u, v)

    # apply substitutions
    for n, api in substitutes.items():
        if n in G2:
            G2.nodes[n]["api"] = api

    return G2


def validate_candidate(G: nx.DiGraph, candidate: Dict) -> bool:
    """Validate candidate edits against resource-edge orphaning.

    Returns True if valid, False if the edit would orphan resource dependents.
    """
    delete_nodes = set(candidate.get("delete_nodes", []))

    # If a deleted node is the source of a resource edge to a node that remains,
    # that's an orphaned dependency and we reject the candidate.
    for u, v, d in G.edges(data=True):
        if d.get("type") == "resource":
            if u in delete_nodes and v not in delete_nodes:
                return False

    # Edge deletions are allowed; they may remove dependency edges but do not
    # create orphaned nodes by themselves. No extra checks necessary here.

    # substitution validity: any substituted api must come from the curated library
    substitutes = candidate.get("substitute", {}) or {}
    for n, new_api in substitutes.items():
        orig_api = G.nodes[n].get("api") if n in G.nodes else None
        if orig_api is None:
            return False
        allowed = get_substitutes(orig_api)
        if new_api not in allowed:
            return False

    # basic sanity: resulting graph must have at least one meaningful node
    G2 = apply_candidate(G, candidate)
    if G2.number_of_nodes() == 0:
        return False
    # require at least one node with a non-unknown api
    meaningful = any((d.get("api") and d.get("api") != "unknown") for _, d in G2.nodes(data=True))
    if not meaningful:
        return False

    # temporal/dependency prerequisites: for each node that references resources,
    # check that at least one producer of that resource exists and precedes it.
    for n, d in G2.nodes(data=True):
        resources = d.get("resources", []) or []
        if not resources:
            continue
        # for each resource, ensure there is at least one predecessor in G2 that
        # has the same resource (producer), or the node itself is considered a producer
        for r in resources:
            has_producer = False
            # check predecessors via resource edges
            for pred in G2.predecessors(n):
                ed = G2.get_edge_data(pred, n) or {}
                if ed.get("type") == "resource":
                    pred_resources = G2.nodes[pred].get("resources", []) or []
                    if r in pred_resources:
                        has_producer = True
                        break
            if not has_producer:
                # allow if this node itself lists the resource (self-producer)
                if r in (d.get("resources") or []):
                    has_producer = True
            if not has_producer:
                return False

    return True
