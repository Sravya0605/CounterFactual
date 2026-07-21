"""Convert NetworkX behavior graphs into PyG Data objects.

The adapter now preserves edge-type information and uses a stable vocabulary so
train-time and inference-time encodings remain consistent.
"""
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


def graph_to_pyg_data(G: nx.DiGraph, api_vocab: Dict[str, int]) -> Any:
    """Convert a single NetworkX DiGraph to a PyG Data object."""
    if torch is None or Data is None:
        raise ImportError("torch and torch_geometric required for PyG adapter")

    node_list = list(G.nodes())
    id_to_idx = {node: index for index, node in enumerate(node_list)}

    feature_dim = max(api_vocab.values()) + 1
    x = []
    for node in node_list:
        api = G.nodes[node].get("api", "unknown")
        idx = api_vocab.get(api, api_vocab.get("<UNK>", 1))
        vec = [0.0] * feature_dim
        vec[idx] = 1.0
        x.append(vec)

    x = torch.tensor(x, dtype=torch.float)

    edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attrs = []
    for u, v in G.edges():
        edge_type = G[u][v].get("type", "unknown")
        edge_attrs.append([1.0 if edge_type == "temporal" else 0.0, 1.0 if edge_type == "resource" else 0.0])
    if edge_attrs:
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
