"""Convert coalesced NetworkX behavior graphs into PyG Data objects.

Provides a simple API vocabulary-based one-hot encoding for node features.
"""
from typing import Tuple, List, Dict, Any
import networkx as nx

try:
    import torch
    from torch_geometric.data import Data
except Exception:
    torch = None
    Data = None


def build_api_vocab(graphs: List[nx.DiGraph]) -> Dict[str, int]:
    vocab = {"<PAD>": 0}
    idx = 1
    for G in graphs:
        for _, d in G.nodes(data=True):
            api = d.get("api", "unknown")
            if api not in vocab:
                vocab[api] = idx
                idx += 1
    return vocab


def graph_to_pyg_data(G: nx.DiGraph, api_vocab: Dict[str, int]) -> Any:
    """Convert a single NetworkX DiGraph to a PyG Data object.

    Node features are one-hot encodings of the `api` attribute using `api_vocab`.
    """
    if torch is None or Data is None:
        raise ImportError("torch and torch_geometric required for PyG adapter")

    node_list = list(G.nodes())
    id_to_idx = {n: i for i, n in enumerate(node_list)}

    # build feature matrix
    x = []
    for n in node_list:
        api = G.nodes[n].get("api", "unknown")
        idx = api_vocab.get(api, 0)
        vec = [0] * (max(api_vocab.values()) + 1)
        vec[idx] = 1
        x.append(vec)

    x = torch.tensor(x, dtype=torch.float)

    # edges
    edges = [ (id_to_idx[u], id_to_idx[v]) for u, v in G.edges() ]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data
