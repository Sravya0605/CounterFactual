"""Tier-1 feasibility checker for structural validity."""
from typing import Dict
from datetime import datetime

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
    # Handle synthetic insertions: create new nodes anchored to an existing
    # anchor node. Insertions use the schema: candidate["insert_nodes"] = [
    # {"anchor": node_id, "api": api_name, "entity_type": ..., "resources": [...]}, ...]
    insertions = candidate.get("insert_nodes", []) or []
    ins_idx = 0
    for ins in insertions:
        anchor = ins.get("anchor")
        api = ins.get("api")
        entity_type = ins.get("entity_type", "api_call")
        resources = ins.get("resources", []) or []
        timestamps = ins.get("timestamps", None)
        # Create a stable new node id that won't collide with existing ones.
        new_node = f"__insert__{anchor}__{ins_idx}"
        ins_idx += 1
        # Timestamp: if anchor has timestamps, place the insertion shortly
        # after the anchor to preserve temporal ordering.
        if timestamps is None and anchor in G2:
            anchor_ts = G2.nodes[anchor].get("timestamps") or []
            timestamps = anchor_ts[:1] if anchor_ts else []
        G2.add_node(new_node, api=api, entity_type=entity_type, resources=list(resources), timestamps=timestamps)
        # Add a temporal edge from anchor -> new_node
        if anchor in G2:
            G2.add_edge(anchor, new_node, type="temporal")
    return G2


def candidate_cost(candidate: Dict) -> int:
    """Return a simple edit-distance proxy for ranking candidates."""
    delete_nodes = len(set(candidate.get("delete_nodes", []) or []))
    delete_edges = len(candidate.get("delete_edges", []) or [])
    substitutions = len((candidate.get("substitute", {}) or {}).keys())
    insertions = len(candidate.get("insert_nodes", []) or [])
    return delete_nodes + delete_edges + substitutions + insertions


OPENING_API_TOKENS = {"createfile", "createprocess", "regcreatekey", "socket", "createservice"}
CLOSING_API_TOKENS = {"closehandle", "deletefile", "regdeletekey", "regdeletevalue", "terminateprocess", "closesocket"}


def _matches_any(api: str, tokens: set) -> bool:
    normalized = str(api or "").lower()
    return any(token in normalized for token in tokens)


def _node_timestamp(data: Dict) -> float:
    """Return the earliest node timestamp as a comparable float."""
    from src.utils.timestamps import normalize_timestamp

    values = [t for t in (data.get("timestamps") or []) if t is not None]
    if not values:
        return 0.0
    return min(normalize_timestamp(t) for t in values)


def _check_resource_lifetime(G2: nx.DiGraph) -> bool:
    """Reject graphs where a resource is used after being closed/freed,
    without an intervening re-open. Modeled as a per-resource open/close
    state machine over timestamp-ordered events, not a single permanent
    close -- resources are legitimately opened, closed, and reopened
    within one trace (e.g. the same file path handled across two separate
    handles), and a naive "no use after the first close" rule would
    reject that common, entirely valid pattern.
    """
    events_by_resource: Dict[str, list] = {}
    for _, data in G2.nodes(data=True):
        api = data.get("api")
        ts = _node_timestamp(data)
        for resource in data.get("resources", []) or []:
            events_by_resource.setdefault(resource, []).append((ts, api))

    for resource, events in events_by_resource.items():
        events.sort(key=lambda pair: pair[0])
        is_open = True  # implicitly open until we see a close, absent evidence otherwise
        for ts, api in events:
            if _matches_any(api, OPENING_API_TOKENS):
                is_open = True
            elif _matches_any(api, CLOSING_API_TOKENS):
                is_open = False
            else:
                if not is_open:
                    return False
    return True


def _check_temporal_order(G2: nx.DiGraph) -> bool:
    """Reject graphs where a temporal edge points backward in time.

    Honest caveat: with the CURRENT edit vocabulary (node/edge deletion,
    API substitution) this check is effectively vacuous today -- no
    existing candidate operation reorders events or changes timestamps,
    so nothing can violate it yet. It's added now, with its own test that
    constructs a violation by hand, for two reasons: (1) it's part of the
    feasibility constraint as formally claimed in the survey/design doc,
    so the paper's methodology section should be true when it says this
    is enforced; (2) it starts enforcing automatically the moment the
    proposer gains any reordering or insertion move, without anyone
    having to remember to add it later.
    """
    for u, v, edge_data in G2.edges(data=True):
        if edge_data.get("type") != "temporal":
            continue
        if _node_timestamp(G2.nodes[u]) > _node_timestamp(G2.nodes[v]):
            return False
    return True


def validate_candidate(G: nx.DiGraph, candidate: Dict) -> bool:
    """Validate candidate edits against dependency closure and substitution rules."""
    delete_nodes = set(candidate.get("delete_nodes", []))
    G2 = apply_candidate(G, candidate)
    if G2.number_of_nodes() == 0:
        return False

    meaningful = any((data.get("api") and data.get("api") != "unknown") for _, data in G2.nodes(data=True))
    if not meaningful:
        return False

    # If the original trace had at least one process node, the edit must not
    # remove every one of them — a behavior graph stripped of all process
    # context can no longer represent a real execution. This is a transition
    # check (did the edit destroy process presence that existed?), not an
    # absolute one, so it correctly leaves graphs that never modeled a
    # process node (e.g. isolated resource-dependency test fixtures) alone.
    original_has_process = any(data.get("entity_type") == "process" for _, data in G.nodes(data=True))
    edited_has_process = any(data.get("entity_type") == "process" for _, data in G2.nodes(data=True))
    if original_has_process and not edited_has_process:
        return False

    if not _check_resource_lifetime(G2):
        return False

    if not _check_temporal_order(G2):
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
            # Allow resources that originate from synthetic insertions if
            # the insertion was explicitly anchored and judged plausible.
            insertions = candidate.get("insert_nodes", []) or []
            inserted_node_names = set()
            for i, ins in enumerate(insertions):
                anchor = ins.get("anchor")
                inserted_node_names.add(f"__insert__{anchor}__{i}")
            if node in inserted_node_names:
                # Ensure the anchor plausibility rule holds.
                try:
                    from src.counterfactual.insertions import is_anchor_plausible

                    # Anchor id encoded in the synthetic node name
                    parts = node.split("__")
                    # pattern: ['', 'insert', anchor, idx]
                    anchor = parts[2] if len(parts) > 2 else None
                    if anchor and is_anchor_plausible(G, anchor, G2.nodes[node].get("api")):
                        continue
                except Exception:
                    pass
            return False

    for node, api in (candidate.get("substitute", {}) or {}).items():
        original_api = G.nodes[node].get("api") if node in G.nodes else None
        if original_api is None:
            return False
        if api not in get_substitutes(original_api):
            return False

    # Validate insertions: anchors must exist and be plausible according to
    # the insertion heuristics.
    insertions = candidate.get("insert_nodes", []) or []
    if insertions:
        try:
            from src.counterfactual.insertions import is_anchor_plausible
        except Exception:
            return False
        for ins in insertions:
            anchor = ins.get("anchor")
            api = ins.get("api")
            if anchor not in G.nodes:
                return False
            if not is_anchor_plausible(G, anchor, api):
                return False

    process_nodes = [node for node, data in G2.nodes(data=True) if data.get("entity_type") == "process"]
    if process_nodes:
        for node, data in G2.nodes(data=True):
            if data.get("entity_type") == "process":
                continue
            # Walk backward through predecessors (not just immediate ones) to
            # find a process ancestor. Immediate-predecessor-only checking
            # breaks for any node reachable from a process only through
            # intermediate nodes -- e.g. a resource-lifetime node whose only
            # predecessor is an API-call event, not the process itself.
            visited = set()
            stack = list(G2.predecessors(node))
            has_process_ancestor = False
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if G2.nodes[current].get("entity_type") == "process":
                    has_process_ancestor = True
                    break
                stack.extend(G2.predecessors(current))
            if not has_process_ancestor:
                return False

    return True