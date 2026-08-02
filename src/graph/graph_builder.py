"""Build a process-aware behavior graph from parsed CAPE events.

The graph uses entity nodes (process, file, registry, network, persistence,
resource) rather than treating every event as a node. This makes the
counterfactual search more faithful to the design document.
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

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
                    "count": 1,
                    "timestamps": [evt.get("timestamp")],
                }
            )
        else:
            idx = key_to_idx[key]
            coalesced[idx]["count"] += 1
            coalesced[idx]["timestamps"].append(evt.get("timestamp"))
        id_map[evt.get("id")] = coalesced[idx]["id"]
    return coalesced, id_map


def _first_timestamp(node: Dict) -> float:
    values = [t for t in node.get("timestamps", []) if t is not None]
    if not values:
        return 0.0
    return float(min(values))


def build_behavior_graph(events: List[Dict]) -> nx.DiGraph:
    """Create a process-aware behavior graph.

    Nodes are created for processes and the resources they touch; edges are added
    for temporal ordering and causal resource dependencies, scoped by process.
    """
    G = nx.DiGraph()
    if not events:
        return G

    coalesced, _ = _coalesce_events(events)

    process_ids = sorted({evt.get("process_id") for evt in events if evt.get("process_id") is not None})
    for process_id in process_ids:
        proc_node_id = f"proc:{process_id}"
        G.add_node(
            proc_node_id,
            api="process",
            entity_type="process",
            process_id=process_id,
            resources=[],
            count=1,
            timestamps=[],
            attack_id=None,
        )

    for node in coalesced:
        entity_type = "resource"
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
        elif any(token in api_name for token in {"dll", "module", "library"}):
            entity_type = "module"

        G.add_node(
            node["id"],
            api=node.get("api"),
            entity_type=entity_type,
            process_id=node.get("process_id"),
            resources=node.get("resources", []),
            count=node.get("count", 1),
            timestamps=node.get("timestamps", []),
            attack_id=node.get("attack_id"),
        )
        if node.get("process_id") is not None:
            G.add_edge(f"proc:{node['process_id']}", node["id"], type="process")

    for proc_id in process_ids:
        proc_nodes = [node["id"] for node in coalesced if node.get("process_id") == proc_id]
        proc_nodes.sort(key=lambda nid: _first_timestamp(next(n for n in coalesced if n["id"] == nid)))
        for i in range(len(proc_nodes) - 1):
            G.add_edge(proc_nodes[i], proc_nodes[i + 1], type="temporal")

    resource_to_nodes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for node in coalesced:
        process_id = node.get("process_id") or "global"
        for resource in node.get("resources", []):
            resource_to_nodes[(str(process_id), resource)].append(node["id"])

    for (process_id, resource), nodes in resource_to_nodes.items():
        nodes_sorted = sorted(nodes, key=lambda nid: _first_timestamp(next(n for n in coalesced if n["id"] == nid)))
        for i in range(len(nodes_sorted) - 1):
            G.add_edge(nodes_sorted[i], nodes_sorted[i + 1], type="resource")

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
