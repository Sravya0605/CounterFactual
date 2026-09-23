"""Tier-1 feasibility checker for structural validity."""
from typing import Dict
from datetime import datetime

import networkx as nx

from src.counterfactual.substitutions import get_substitutes
from src.counterfactual.insertions import ANCHOR_FAMILY, is_anchor_plausible


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
        entity_type = ins.get("entity_type") or _entity_type_for_api(api)
        resources = ins.get("resources", []) or []
        timestamps = ins.get("timestamps", None)
        # Create a stable new node id that won't collide with existing ones.
        new_node = f"__insert__{ins_idx}"
        ins_idx += 1
        # Timestamp: if anchor has timestamps, place the insertion shortly
        # after the anchor to preserve temporal ordering.
        if timestamps is None and anchor in G2:
            anchor_ts = G2.nodes[anchor].get("timestamps") or []
            timestamps = anchor_ts[:1] if anchor_ts else []
        anchor_data = G2.nodes[anchor] if anchor in G2 else {}
        G2.add_node(
            new_node,
            api=api,
            entity_type=entity_type,
            process_id=anchor_data.get("process_id"),
            resources=list(resources),
            timestamps=timestamps,
            insertion_anchor=anchor,
            insertion_index=ins_idx,
        )
        # Add a temporal edge from anchor -> new_node
        if anchor in G2:
            G2.add_edge(anchor, new_node, type="temporal")
    return G2


def _entity_type_for_api(api: str) -> str:
    normalized = str(api or "").lower()
    if "reg" in normalized:
        return "registry"
    if "service" in normalized or "task" in normalized:
        return "persistence"
    if "socket" in normalized or "connect" in normalized:
        return "network"
    if "process" in normalized or "thread" in normalized:
        return "process"
    if "file" in normalized:
        return "file"
    return "unclassified_event"


def candidate_cost(candidate: Dict) -> int:
    """Return a simple edit-distance proxy for ranking candidates."""
    delete_nodes = len(set(candidate.get("delete_nodes", []) or []))
    delete_edges = len(candidate.get("delete_edges", []) or [])
    substitutions = len((candidate.get("substitute", {}) or {}).keys())
    insertions = len(candidate.get("insert_nodes", []) or [])
    return delete_nodes + delete_edges + substitutions + insertions


OPENING_API_TOKENS = {
    "createfile", "openfile", "createprocess", "openprocess",
    "regcreatekey", "regopenkey", "ntopenkey", "ntcreatekey",
    "createsection", "opensection", "createmutant", "openmutant",
    "socket", "createservice", "openscmanager", "loadlibrary", "ldrloaddll",
    "open", "create", "socket", "connect",
}
CLOSING_API_TOKENS = {
    "closehandle", "ntclose", "regclosekey",
    "deletefile", "regdeletekey", "regdeletevalue", "terminateprocess",
    "closesocket",
}
# NOTE: NtReleaseMutant is deliberately NOT here. Releasing a mutex is an
# unlock, not a close -- the handle stays fully valid and can be waited on,
# released, and re-waited on any number of times for the rest of the
# process's life (the normal pattern for a mutex used as a lock). Treating
# it as a closing token caused _check_resource_lifetime to reject the very
# first NtWaitForSingleObject after any NtReleaseMutant, which in turn
# rejected the unmodified, ground-truth graph itself on real samples that
# use a mutex as a singleton/critical-section lock -- and since this check
# runs over the whole graph rather than just the edited region, that one
# false violation was enough to fail nearly every proposed candidate,
# regardless of what the candidate actually touched.


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


def _is_stable_resource_identifier(resource) -> bool:
    """Return False for resource strings that aren't a reliable identity to
    track a lifecycle against.

    Raw numeric handle values (e.g. "0x00000088") are not stable
    identifiers: Windows recycles handle numbers constantly, so the same
    numeric value can legitimately refer to a completely different object
    later in the same trace once the original was closed. Treating the
    number itself as "the resource" conflates unrelated objects and
    produces false use-after-close violations. Well-known pseudo-handle
    constants (GetCurrentProcess's -1 / 0xffffffff / 0xffffffffffffffff)
    are excluded outright -- they are never genuinely opened or closed,
    they're a compile-time sentinel every self-referencing call
    (VirtualAlloc, VirtualProtect, NtMapViewOfSection on your own process,
    ...) shares, which shows up in almost any sample that unpacks or
    injects into itself.

    File paths, registry keys, domains, and similar semantic identifiers
    are unaffected by this and are still tracked normally.
    """
    r = str(resource or "").strip().lower()
    if not r:
        return False
    if r in {"-1", "0xffffffff", "0xfffffffe", "0xffffffffffffffff"}:
        return False
    if r.startswith("0x"):
        return False
    return True


def _check_resource_lifetime(G2: nx.DiGraph) -> bool:
    """Reject graphs where a resource is used after being closed/freed,
    without an intervening re-open. Modeled as a per-resource open/close
    state machine over timestamp-ordered events, not a single permanent
    close -- resources are legitimately opened, closed, and reopened
    within one trace (e.g. the same file path handled across two separate
    handles), and a naive "no use after the first close" rule would
    reject that common, entirely valid pattern.

    Only tracked for stable resource identifiers (see
    _is_stable_resource_identifier) -- raw numeric handle values are
    excluded since the OS can and does recycle them for unrelated objects.
    """
    events_by_resource: Dict[str, list] = {}
    for _, data in G2.nodes(data=True):
        api = data.get("api")
        ts = _node_timestamp(data)
        for resource in data.get("resources", []) or []:
            if not _is_stable_resource_identifier(resource):
                continue
            events_by_resource.setdefault(resource, []).append((ts, api))

    for resource, events in events_by_resource.items():
        events.sort(key=lambda pair: pair[0])
        is_open = True  # implicitly open until we see a close, absent evidence otherwise
        for ts, api in events:
            if _matches_any(api, CLOSING_API_TOKENS):
                is_open = False
            elif _matches_any(api, OPENING_API_TOKENS):
                is_open = True
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
    original_has_process = any(str(data.get("api", "")).lower() == "process" for _, data in G.nodes(data=True))
    edited_has_process = any(str(data.get("api", "")).lower() == "process" for _, data in G2.nodes(data=True))
    if original_has_process and not edited_has_process:
        return False

    if not _check_resource_lifetime(G2):
        return False

    if not _check_temporal_order(G2):
        return False

    if not _check_argument_data_flow(G, delete_nodes, G2):
        return False

    if not _check_lifetime_entities(G2):
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
            if G2.nodes[node].get("insertion_anchor") is not None:
                # Ensure the anchor plausibility rule holds.
                try:
                    anchor = G2.nodes[node].get("insertion_anchor")
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

    # Validate insertions: anchors and inserted APIs must be plausible according to
    # the insertion heuristics.
    insertions = candidate.get("insert_nodes", []) or []
    if insertions:
        for ins in insertions:
            anchor = ins.get("anchor")
            api = ins.get("api")
            if anchor not in G.nodes:
                return False
            if api not in {candidate_api for values in ANCHOR_FAMILY.values() for candidate_api in values}:
                return False
            if not is_anchor_plausible(G, anchor, api):
                return False

    # NOTE: identify the process ANCHOR nodes by their literal api value, not
    # by entity_type=="process" -- real event nodes (OpenProcess,
    # CreateRemoteThread, ...) also carry entity_type=="process" (see
    # graph_builder._entity_type_for_api), since that label describes the
    # API category, not "this is an anchor". Filtering by entity_type here
    # let a deleted child process's own injection *events* survive, orphaned
    # from any process, by mistakenly treating them as exempt anchors.
    process_nodes = [node for node, data in G2.nodes(data=True) if str(data.get("api", "")).lower() == "process"]
    if process_nodes:
        for node in process_nodes:
            child_pid = G2.nodes[node].get("process_id")
            parent_id = G2.nodes[node].get("parent_process_id")
            if parent_id is None:
                continue
            parent_node = f"proc:{parent_id}"
            # Only check parent process and creation events if the parent process was modeled in the trace
            if parent_node in G:
                if parent_node not in G2:
                    return False
                # The creation event is issued by the PARENT process, not the child.
                # Search for any API-call node whose process_id equals the parent PID
                # and whose API name is a known process-creation primitive.
                _creation_apis = {"createprocessw", "createprocessa", "ntcreateuserprocess",
                                  "zwcreateuserprocess", "createprocesswithlogonw",
                                  "createprocesswithtokenw"}
                creation_events = [
                    event for event, data in G2.nodes(data=True)
                    if str(data.get("process_id", "")) in {str(parent_id), str(child_pid)}
                    and str(data.get("api", "")).lower() in _creation_apis
                ]
                if not creation_events:
                    return False
        # Every non-process node must be reachable from SOME process node via
        # a forward path (process -> ... -> node). Originally this walked
        # backward from every single node independently, which redoes a full
        # graph traversal per node -- O(V*(V+E)) for the whole check. Doing
        # one multi-source forward traversal starting from all process nodes
        # at once computes the exact same "has a process ancestor" set for
        # every node simultaneously, in O(V+E) total.
        reachable_from_process = set()
        stack=list(process_nodes)
        while stack:
            current = stack.pop()
            if current in reachable_from_process:
                continue
            reachable_from_process.add(current)
            stack.extend(G2.successors(current))
        for node, data in G2.nodes(data=True):
            if str(data.get("api", "")).lower() == "process":
                continue
            if node not in reachable_from_process:
                return False
    return True


def _argument_values(data: Dict) -> set:
    raw = data.get("arguments") or data.get("args") or []
    values = set()

    def _collect(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                _collect(sub)
        elif isinstance(item, dict):
            if "value" in item and item["value"] is not None:
                values.add(str(item["value"]))
            else:
                for v in item.values():
                    _collect(v)
        else:
            values.add(str(item))

    _collect(raw)
    return values


def _check_argument_data_flow(G: nx.DiGraph, delete_nodes: set, G2: nx.DiGraph) -> bool:
    """Ensure that deleting nodes does not leave downstream consumers with dangling arguments.

    A surviving node in G2 is invalid if it consumes an argument or handle that was
    produced by a deleted node, with no surviving producer to supply it.
    """
    for u in delete_nodes:
        if u not in G:
            continue
        u_vals = set(str(r) for r in G.nodes[u].get("resources", []) or []) | _argument_values(G.nodes[u])
        pred_vals = set()
        for p in G.predecessors(u):
            pred_vals.update(str(r) for r in G.nodes[p].get("resources", []) or [])
            pred_vals.update(_argument_values(G.nodes[p]))
        # Values uniquely introduced/produced by u (not inherited from predecessors)
        newly_produced = u_vals - pred_vals
        if not newly_produced:
            continue
        # Check if any surviving downstream node in G2 references a value produced by u
        for succ in G.successors(u):
            if succ in G2:
                succ_vals = set(str(r) for r in G2.nodes[succ].get("resources", []) or []) | _argument_values(G2.nodes[succ])
                if succ_vals & newly_produced:
                    return False
    return True


def _check_lifetime_entities(G2: nx.DiGraph) -> bool:
    for node, data in G2.nodes(data=True):
        if data.get("entity_type") != "resource":
            continue
        if not list(G2.predecessors(node)):
            return False
        if data.get("state") == "released" and not list(G2.successors(node)):
            return False
    return True