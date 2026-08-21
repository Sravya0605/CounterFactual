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

    insert_nodes = candidate.get("insert_nodes", []) or []
    for spec in insert_nodes:
        api = spec.get("api")
        process_id = spec.get("target_process_id")
        if api is None or process_id is None:
            continue
        proc_node_id = f"proc:{process_id}"
        if proc_node_id not in G2:
            continue
        tail_node = _process_chain_tail(G2, process_id)
        tail_data = G2.nodes[tail_node]
        tail_ts = [t for t in (tail_data.get("timestamps") or []) if t is not None]
        tail_seqs = [s for s in (tail_data.get("sequences") or []) if s is not None]
        next_seq = (max(tail_seqs) + 1) if tail_seqs else 1

        new_node_id = _next_insertion_id(G2)
        G2.add_node(
            new_node_id,
            api=api,
            entity_type="file",
            process_id=process_id,
            resources=[],
            count=1,
            timestamps=[tail_ts[0]] if tail_ts else [],
            sequences=[next_seq],
        )
        G2.add_edge(proc_node_id, new_node_id, type="process")
        G2.add_edge(tail_node, new_node_id, type="temporal")

    return G2


def candidate_cost(candidate: Dict) -> int:
    """Return a simple edit-distance proxy for ranking candidates."""
    delete_nodes = len(set(candidate.get("delete_nodes", []) or []))
    delete_edges = len(candidate.get("delete_edges", []) or [])
    substitutions = len((candidate.get("substitute", {}) or {}).keys())
    insertions = len(candidate.get("insert_nodes", []) or [])
    return delete_nodes + delete_edges + substitutions + insertions

INSERTABLE_APIS = {"createtoolhelp32snapshot"}

ANCHOR_FAMILY = {
    "createtoolhelp32snapshot", "process32first", "process32next",
    "module32first", "module32next", "getmodulehandlea", "getprocaddress",
    "virtualqueryex", "readprocessmemory",
}


def _check_insertion_plausibility(G: nx.DiGraph, candidate: Dict) -> bool:
    """An inserted API call is only plausible if (a) it's one of the specific
    APIs we allow inserting at all, and (b) the target process already shows
    at least one call from the broader process/module-introspection anchor
    family somewhere in its OWN observed timeline. The anchor family is
    deliberately broader than INSERTABLE_APIS -- e.g. a process that already
    calls ReadProcessMemory (the standard second step in process-injection
    workflows, after enumeration) is treated as plausibly capable of also
    having enumerated processes, even if CreateToolhelp32Snapshot itself
    isn't already present. This is a judgment call about API semantic
    relatedness, not a purely mechanical family match -- documented here
    because it should be stated exactly this way in any writeup, not glossed
    as a narrower "same API family" rule than it actually is in practice.
    Checked against the ORIGINAL graph G, not the edited one, since this is
    a claim about pre-existing behavior.
    """
    insert_nodes = candidate.get("insert_nodes", []) or []
    if not insert_nodes:
        return True
    for spec in insert_nodes:
        api = str(spec.get("api") or "").lower()
        process_id = spec.get("target_process_id")
        if api not in INSERTABLE_APIS:
            return False
        has_anchor_call = any(
            data.get("process_id") == process_id
            and str(data.get("api") or "").lower() in ANCHOR_FAMILY
            for _, data in G.nodes(data=True)
        )
        if not has_anchor_call:
            return False
    return True


def _next_insertion_id(G2: nx.DiGraph) -> str:
    i = 0
    while f"n_ins_{i}" in G2:
        i += 1
    return f"n_ins_{i}"


def _process_chain_tail(G2: nx.DiGraph, process_id) -> str:
    """Return the node with the max sequence number among this process's
    event nodes in G2, or the process node itself if it has none yet. In
    practice the latter never occurs for candidates that pass
    _check_insertion_plausibility, since that check requires at least one
    existing event on the target process already.
    """
    proc_node = f"proc:{process_id}"
    ranked = []
    for node, data in G2.nodes(data=True):
        if data.get("process_id") == process_id and data.get("entity_type") != "process":
            seqs = [s for s in (data.get("sequences") or []) if s is not None]
            if seqs:
                ranked.append((max(seqs), node))
    if not ranked:
        return proc_node
    ranked.sort()
    return ranked[-1][1]

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
    if not _check_insertion_plausibility(G, candidate):
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