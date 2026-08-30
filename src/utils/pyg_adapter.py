"""Convert NetworkX behavior graphs into PyG Data objects with richer node and edge encodings."""
import math
from typing import Any, Dict, List

import networkx as nx

try:
    import torch
    from torch_geometric.data import Data
except Exception:
    torch = None
    Data = None


def build_api_vocab(graphs: List[nx.DiGraph]) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for G in graphs:
        for _, data in G.nodes(data=True):
            api = data.get("api", "unknown")
            if api not in vocab:
                vocab[api] = idx
                idx += 1
    return vocab


def _node_feature_vector(node_id: str, G: nx.DiGraph, api_vocab: Dict[str, int]) -> List[float]:
    api = str(G.nodes[node_id].get("api", "unknown"))
    idx = api_vocab.get(api, api_vocab.get("<UNK>", 1))
    count = float(G.nodes[node_id].get("count", 1))
    entity_type = str(G.nodes[node_id].get("entity_type", "unknown")).lower()
    resource_count = len(G.nodes[node_id].get("resources", []) or [])
    feature_dim = max(api_vocab.values()) + 1
    vec = [0.0] * feature_dim
    vec[idx] = 1.0
    vec.append(math.log1p(count))
    vec.append(float(1 if "file" in entity_type else 0))
    vec.append(float(1 if "process" in entity_type else 0))
    vec.append(float(1 if "registry" in entity_type else 0))
    vec.append(float(resource_count))
    return vec


def graph_to_pyg_data(G: nx.DiGraph, api_vocab: Dict[str, int]) -> Any:
    """Convert a single NetworkX DiGraph to a PyG Data object."""
    if torch is None or Data is None:
        raise ImportError("torch and torch_geometric required for PyG adapter")

    node_list = list(G.nodes())
    id_to_idx = {node: index for index, node in enumerate(node_list)}

    x = []
    for node in node_list:
        x.append(_node_feature_vector(node, G, api_vocab))

    x = torch.tensor(x, dtype=torch.float)

    edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attrs = []
    for u, v in G.edges():
        edge_data = G[u][v]
        edge_type = str(edge_data.get("type", "unknown")).lower()
        weight = float(edge_data.get("weight", edge_data.get("count", 1.0)))
        delay = float(edge_data.get("delay", 0.0))
        edge_attrs.append([
            1.0 if edge_type == "temporal" else 0.0,
            1.0 if edge_type == "resource" else 0.0,
            1.0 if edge_type == "process" else 0.0,
            math.log1p(max(weight, 1.0)),
            delay,
        ])
    if edge_attrs:
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_attr = torch.empty((0, 5), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
