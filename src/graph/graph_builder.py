"""Behavior graph builder with simple coalescing and heuristic dependency edges

Exports `build_behavior_graph(events)` which returns a `networkx.DiGraph` where
nodes represent coalesced events/resources and edges represent temporal and
resource-dependency relationships.
"""
from typing import List, Dict, Tuple
import networkx as nx
from collections import defaultdict


def _coalesce_events(events: List[Dict]) -> Tuple[List[Dict], dict]:
    """Coalesce identical events (by api + resource set) and return new list and mapping.

    Returns (coalesced_events, id_map) where id_map maps original event id -> coalesced id.
    """
    key_to_idx = {}
    coalesced = []
    id_map = {}
    for evt in events:
        key = (evt.get("api"), tuple(sorted(evt.get("resources", []))))
        if key not in key_to_idx:
            idx = len(coalesced)
            key_to_idx[key] = idx
            coalesced.append({"id": f"n{idx}", "api": evt.get("api"), "resources": evt.get("resources", []), "count": 1, "timestamps": [evt.get("timestamp")]})
        else:
            idx = key_to_idx[key]
            coalesced[idx]["count"] += 1
            coalesced[idx]["timestamps"].append(evt.get("timestamp"))
        id_map[evt.get("id")] = coalesced[idx]["id"]
    return coalesced, id_map


def build_behavior_graph(events: List[Dict]) -> nx.DiGraph:
    """Build a behavior graph from parsed events.

    Node attributes: `api`, `resources`, `count`, `timestamps`.
    Edge types: `temporal` and `resource` stored as edge attribute `type`.
    """
    G = nx.DiGraph()
    if not events:
        return G

    coalesced, id_map = _coalesce_events(events)

    # Add nodes
    for node in coalesced:
        G.add_node(node["id"], api=node["api"], resources=node["resources"], count=node["count"], timestamps=node["timestamps"])

    # Temporal edges: order coalesced nodes by first timestamp
    order = sorted(coalesced, key=lambda n: min([t for t in n.get("timestamps") if t is not None] or [0]))
    for i in range(len(order) - 1):
        s = order[i]["id"]
        t = order[i + 1]["id"]
        G.add_edge(s, t, type="temporal")

    # Resource edges: connect nodes that share a resource string
    resource_to_nodes = defaultdict(list)
    for node in coalesced:
        for r in node.get("resources", []):
            resource_to_nodes[r].append(node["id"])

    for r, nodes in resource_to_nodes.items():
        # sort nodes by earliest timestamp so edges point from earlier->later
        def first_ts(nid):
            ts = G.nodes[nid].get("timestamps") or []
            vals = [t for t in ts if t is not None]
            return min(vals) if vals else 0

        nodes_sorted = sorted(nodes, key=lambda nid: first_ts(nid))
        for i in range(len(nodes_sorted)):
            for j in range(i + 1, len(nodes_sorted)):
                a = nodes_sorted[i]
                b = nodes_sorted[j]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, type="resource")

    return G


if __name__ == "__main__":
    import json, sys
    from src.ingestion.parser import parse_cape_json
    if len(sys.argv) < 2:
        print("Usage: graph_builder.py /path/to/cape_report.json")
        sys.exit(1)
    evts = parse_cape_json(sys.argv[1])
    G = build_behavior_graph(evts)
    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
