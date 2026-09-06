"""Insertion proposer and anchor-family plausibility rules.

This module encodes simple heuristics to decide when inserting a synthetic
API call into an existing trace is plausible. The rule implemented here is
conservative: we only propose inserting calls that are anchored to a process
node that already exhibits related enumeration/introspection activity.
"""
from typing import Dict, List, Any
import networkx as nx

# Anchor family: mapping of anchor token -> insertable APIs. The keys are
# substrings we look for in nearby APIs to justify insertion. Values are
# candidate APIs to insert when an anchor is found.
ANCHOR_FAMILY = {
    "enum": ["OpenProcess", "QueryProcess", "ReadProcessMemory"],
    "query": ["OpenProcess", "QueryInformation"],
    "openprocess": ["CreateRemoteThread", "OpenProcess"],
}


def _node_matches_anchor(node_api: str, anchor_token: str) -> bool:
    if not node_api:
        return False
    return anchor_token.lower() in node_api.lower()


def propose_insertions(G: nx.DiGraph, top_k: int = 20, api_vocab: Any = None) -> List[Dict]:
    """Emit simple insertion candidate dicts anchored to process nodes.

    Each candidate uses the schema: {"insert_nodes": [ {"anchor": node, "api": api} ],
    "delete_nodes": [], "substitute": {}}. These are intentionally small
    single-event insertions; the search/evaluation pipeline will assign cost
    and run feasibility checks.
    """
    candidates: List[Dict] = []
    nodes = list(G.nodes())
    for n in nodes:
        api = G.nodes[n].get("api", "")
        # Only consider anchors that appear to be process-introspection
        # or enumeration calls. This keeps the synthetic insertion plausible.
        for anchor_token, insertable in ANCHOR_FAMILY.items():
            if _node_matches_anchor(api, anchor_token):
                for ins_api in insertable:
                    candidates.append({"insert_nodes": [{"anchor": n, "api": ins_api}], "delete_nodes": [], "substitute": {}})
                    if len(candidates) >= top_k:
                        return candidates
    return candidates


def is_anchor_plausible(G: nx.DiGraph, anchor_node: str, inserted_api: str) -> bool:
    """Decision rule used by feasibility checking: return True if the
    anchor node (or nearby nodes in the same process) have evidence that
    justifies inserting `inserted_api`.
    """
    if anchor_node not in G:
        return False
    allowed = {
        candidate
        for candidates in ANCHOR_FAMILY.values()
        for candidate in candidates
    }
    if inserted_api not in allowed:
        return False
    # Look at neighbor APIs in the same process lineage.
    # If any nearby API contains tokens that match ANCHOR_FAMILY keys,
    # accept the insertion. Otherwise reject conservatively.
    for pred in list(G.predecessors(anchor_node)) + list(G.successors(anchor_node)):
        api = G.nodes[pred].get("api", "")
        for token, candidates in ANCHOR_FAMILY.items():
            if inserted_api in candidates and token in (api or "").lower():
                return True
    # Also check the anchor node itself.
    api = G.nodes[anchor_node].get("api", "")
    for token, candidates in ANCHOR_FAMILY.items():
        if inserted_api in candidates and token in (api or "").lower():
            return True
    return False
