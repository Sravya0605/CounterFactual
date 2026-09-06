"""Build a process-aware behavior graph from parsed CAPE events.

The graph uses entity nodes (process, file, registry, network, persistence,
resource) rather than treating every event as a node. This makes the
counterfactual search more faithful to the design document.
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from datetime import datetime

import networkx as nx


def _coalesce_events(events: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    """Collapse repeated identical events while retaining process context."""
    key_to_idx: Dict[Tuple[Any, Any, Any], int] = {}
    coalesced: List[Dict] = []
    id_map: Dict[str, str] = {}

    for evt in events:
        key = (evt.get("process_id"), evt.get("api"), tuple(sorted(evt.get("resources", []))))
        if key not in key_to_idx:
            idx = len(coalesced)
            key_to_idx[key] = idx
            coalesced.append(
                {
                    "id": f"n{idx}",
                    "api": evt.get("api"),
                    "process_id": evt.get("process_id"),
                    "event_type": evt.get("event_type", "system"),
                    "attack_id": evt.get("attack_id"),
                    "resources": evt.get("resources", []),
                    "arguments": [evt.get("args", {})],
                    "count": 1,
                    "timestamps": [evt.get("timestamp")],
                    "sequences": [evt.get("sequence")],
                    "event_ids": [evt.get("id")],
                }
            )
        else:
            idx = key_to_idx[key]
            coalesced[idx]["count"] += 1
            coalesced[idx]["timestamps"].append(evt.get("timestamp"))
            coalesced[idx]["sequences"].append(evt.get("sequence"))
            coalesced[idx]["event_ids"].append(evt.get("id"))
            coalesced[idx]["arguments"].append(evt.get("args", {}))
        id_map[evt.get("id")] = coalesced[idx]["id"]
    return coalesced, id_map


def _first_timestamp(node: Dict) -> float:
    from src.utils.timestamps import normalize_timestamp

    values = [t for t in node.get("timestamps", []) if t is not None]
    if not values:
        return 0.0
    return min(normalize_timestamp(t) for t in values)


def build_behavior_graph(
    events: List[Dict],
    lifetimes: List[Dict] = None,
    active_resources: List[Dict] = None,
) -> nx.DiGraph:
    """Create a process-aware behavior graph with resource lifetimes.

    The graph contains:
      - process -> event edges
      - temporal event -> event edges
      - resource-reference edges
      - acquisition -> resource -> release lifetime relationships

    ``lifetimes`` should contain the reconstructed resource lifetime records
    produced by ``match_resource_lifetimes()``.
    """
    G = nx.DiGraph()

    if not events:
        return G

    if lifetimes is None:
        lifetimes = []

    if active_resources is None:
        active_resources = []

    coalesced, _ = _coalesce_events(events)

    # ------------------------------------------------------------------
    # 1. Process nodes
    # ------------------------------------------------------------------

    process_ids = sorted(
        {
            evt.get("process_id")
            for evt in events
            if evt.get("process_id") is not None
        }
    )

    for process_id in process_ids:
        proc_node_id = f"proc:{process_id}"
        process_events = [evt for evt in events if evt.get("process_id") == process_id]
        parent_id = next((evt.get("parent_process_id") for evt in process_events if evt.get("parent_process_id") is not None), None)

        G.add_node(
            proc_node_id,
            api="process",
            entity_type="process",
            process_id=process_id,
            parent_process_id=parent_id,
            resources=[],
            count=1,
            timestamps=[],
            attack_id=None,
        )

        if parent_id is not None and f"proc:{parent_id}" in G:
            G.add_edge(f"proc:{parent_id}", proc_node_id, type="process_creation")

    # ------------------------------------------------------------------
    # 2. Coalesced event nodes
    # ------------------------------------------------------------------

    for node in coalesced:
        entity_type = "unclassified_event"

        api_name = str(node.get("api") or "").lower()

        if node.get("event_type") == "persistence":
            entity_type = "persistence"
        elif node.get("event_type") == "registry":
            entity_type = "registry"
        elif node.get("event_type") == "network":
            entity_type = "network"
        elif node.get("event_type") == "file":
            entity_type = "file"
        elif node.get("event_type") == "process":
            entity_type = "process"
        elif any(
            token in api_name
            for token in {"dll", "module", "library"}
        ):
            entity_type = "module"

        G.add_node(
            node["id"],
            api=node.get("api"),
            entity_type=entity_type,
            process_id=node.get("process_id"),
            resources=node.get("resources", []),
            arguments=node.get("arguments", []),
            count=node.get("count", 1),
            timestamps=node.get("timestamps", []),
            sequences=node.get("sequences", []),
            event_ids=node.get("event_ids", []),
            attack_id=node.get("attack_id"),
        )

        if node.get("process_id") is not None:
            G.add_edge(
                f"proc:{node['process_id']}",
                node["id"],
                type="process",
            )

    # ------------------------------------------------------------------
    # 3. Temporal edges
    # ------------------------------------------------------------------

    node_by_id = {
        node["id"]: node
        for node in coalesced
    }

    for proc_id in process_ids:
        proc_nodes = [
            node["id"]
            for node in coalesced
            if node.get("process_id") == proc_id
        ]

        proc_nodes.sort(
            key=lambda nid: min(
                s
                for s in node_by_id[nid].get("sequences", [])
                if s is not None
            )
        )

        for i in range(len(proc_nodes) - 1):
            source = proc_nodes[i]
            target = proc_nodes[i + 1]

            G.add_edge(
                source,
                target,
                type="temporal",
            )

    # ------------------------------------------------------------------
    # 4. Existing resource-reference edges
    # ------------------------------------------------------------------

    resource_to_nodes: Dict[
        Tuple[str, str],
        List[str]
    ] = defaultdict(list)

    for node in coalesced:
        process_id = node.get("process_id") or "global"

        for resource in node.get("resources", []):
            resource_to_nodes[
                (str(process_id), str(resource))
            ].append(node["id"])

    for (_, _), nodes in resource_to_nodes.items():
        nodes_sorted = sorted(
            nodes,
            key=lambda nid: min(
                s
                for s in node_by_id[nid].get("sequences", [])
                if s is not None
            )
        )

        for i in range(len(nodes_sorted) - 1):
            source = nodes_sorted[i]
            target = nodes_sorted[i + 1]

            G.add_edge(
                source,
                target,
                type="resource",
            )

    # ------------------------------------------------------------------
    # 5. Build sequence -> graph-node mapping
    # ------------------------------------------------------------------

    sequence_to_node: Dict[Tuple[str, int], str] = {}

    for node in coalesced:
        for sequence in node.get("sequences", []):
            if sequence is not None:
                sequence_to_node[(str(node.get("process_id")), int(sequence))] = node["id"]

    # ------------------------------------------------------------------
    # 6. Add reconstructed resource lifetime edges
    # ------------------------------------------------------------------

    lifetime_edges = 0

    for lifetime in lifetimes:
        acquisition_sequence = lifetime.get(
            "acquisition_sequence"
        )
        release_sequence = lifetime.get(
            "release_sequence"
        )

        if acquisition_sequence is None:
            continue

        acquisition_node = sequence_to_node.get(
            (str(lifetime.get("process_id")), int(acquisition_sequence))
        )

        release_node = None

        if release_sequence is not None:
            release_node = sequence_to_node.get(
                (str(lifetime.get("process_id")), int(release_sequence))
            )

        # We need the acquisition event to exist in the graph.
        if acquisition_node is None:
            continue

        # Create a unique resource entity for this lifetime.
        resource_id = (
            f"resource:"
            f"{lifetime.get('resource_type', 'unknown')}:"
            f"{lifetime.get('handle', 'unknown')}:"
            f"{acquisition_sequence}"
        )

        G.add_node(
            resource_id,
            entity_type="resource",
            resource_type=lifetime.get("resource_type"),
            handle=lifetime.get("handle"),
            process_id=lifetime.get("process_id"),
            acquisition_api=lifetime.get("acquisition_api"),
            release_api=lifetime.get("release_api"),
            acquisition_sequence=acquisition_sequence,
            release_sequence=release_sequence,
            acquisition_timestamp=lifetime.get(
                "acquisition_timestamp"
            ),
            release_timestamp=lifetime.get(
                "release_timestamp"
            ),
            state="released" if release_node else "active",
        )

        # Acquisition event -> resource
        G.add_edge(
            acquisition_node,
            resource_id,
            type="resource",
            relation="acquires",
        )

        lifetime_edges += 1

        # Resource -> release event
        if release_node is not None:
            G.add_edge(
                resource_id,
                release_node,
                type="resource",
                relation="releases",
            )

    # ------------------------------------------------------------------
    # 7. Add resources that remain active
    # ------------------------------------------------------------------

    for resource in active_resources:
        acquisition_sequence = resource.get("sequence")

        if acquisition_sequence is None:
            continue

        acquisition_node = sequence_to_node.get(
            (str(resource.get("process_id")), int(acquisition_sequence))
        )

        if acquisition_node is None:
            continue

        resource_id = (
            f"resource:"
            f"{resource.get('resource_type', 'unknown')}:"
            f"{resource.get('handle', 'unknown')}:"
            f"{acquisition_sequence}"
        )

        G.add_node(
            resource_id,
            entity_type="resource",
            resource_type=resource.get("resource_type"),
            handle=resource.get("handle"),
            process_id=resource.get("process_id"),
            acquisition_api=resource.get("api"),
            acquisition_sequence=acquisition_sequence,
            acquisition_timestamp=resource.get("timestamp"),
            state="active",
        )

        G.add_edge(
            acquisition_node,
            resource_id,
            type="resource",
            relation="acquires",
        )

    return G

if __name__ == "__main__":
    import sys

    from src.ingestion.parser import parse_cape_json

    if len(sys.argv) < 2:
        print("Usage: graph_builder.py /path/to/cape_report.json")
        sys.exit(1)
    events = parse_cape_json(sys.argv[1])
    graph = build_behavior_graph(events)
    print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
