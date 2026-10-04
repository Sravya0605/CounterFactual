"""GNN training harness using PyG."""
from typing import Any, List

import torch
from torch_geometric.loader import DataLoader

from src.classifier.gnn_model import SimpleGCN
from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data


def train_gnn(
    graphs: List[Any],
    labels: List[int],
    epochs: int = 10,
    batch_size: int = 16,
    pooling: str = "mean_max",
) -> Any:
    vocab = build_api_vocab(graphs)
    data_list = [graph_to_pyg_data(G, vocab) for G in graphs]
    max_edge_dim = max((d.edge_attr.size(-1) for d in data_list if d.edge_attr.numel() > 0), default=5)
    for i, d in enumerate(data_list):
        d.y = torch.tensor([labels[i]], dtype=torch.float)

    loader = DataLoader(data_list, batch_size=batch_size)
    in_channels = data_list[0].x.size(-1)
    model = SimpleGCN(
        in_channels=in_channels,
        hidden=32,
        edge_dim=max_edge_dim,
        pooling=pooling,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optim.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch, edge_attr=getattr(batch, "edge_attr", None))
            if out.dim() == 0:
                out = out.unsqueeze(0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()
            optim.step()
    return model


def predict_gnn_proba(model: Any, graphs: List[Any], api_vocab: Any) -> List[float]:
    if api_vocab is None:
        api_vocab = build_api_vocab(graphs)
    data_list = [graph_to_pyg_data(G, api_vocab) for G in graphs]
    with torch.no_grad():
        outputs = []
        for data in data_list:
            batch = torch.zeros(data.x.size(0), dtype=torch.long)
            out = model(data.x, data.edge_index, batch=batch, edge_attr=data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr.numel() > 0 else None)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            outputs.append(float(torch.sigmoid(out).item()))
    return outputs


class GNNDeletionScorer:
    """Score deletion-only candidates by slicing a cached base PyG graph."""

    def __init__(self, model: Any, graph: Any, api_vocab: Any):
        self.model = model
        self.graph = graph
        self.api_vocab = api_vocab
        self.base_data = graph_to_pyg_data(graph, api_vocab)
        self.node_to_index = {
            node: index for index, node in enumerate(graph.nodes())
        }
        self.edge_to_index = {
            edge: index for index, edge in enumerate(graph.edges())
        }

    def predict_proba(self, candidate: Any) -> float:
        delete_nodes = set(candidate.get("delete_nodes", []) or [])
        deleted_node_indices = [
            self.node_to_index[node]
            for node in delete_nodes
            if node in self.node_to_index
        ]
        deleted_edge_indices = [
            self.edge_to_index[(edge[0], edge[1])]
            for edge in (candidate.get("delete_edges", []) or [])
            if isinstance(edge, (list, tuple))
            and len(edge) == 2
            and (edge[0], edge[1]) in self.edge_to_index
        ]

        node_keep = None
        old_to_new = None
        x = self.base_data.x
        if deleted_node_indices:
            node_keep = torch.ones(x.size(0), dtype=torch.bool)
            node_keep[deleted_node_indices] = False
            if not node_keep.any():
                raise ValueError("Cannot score a candidate that removes every graph node")
            old_to_new = torch.full((x.size(0),), -1, dtype=torch.long)
            old_to_new[node_keep] = torch.arange(int(node_keep.sum()))
            x = x[node_keep]

        edge_keep = None
        if node_keep is not None or deleted_edge_indices:
            edge_keep = torch.ones(
                self.base_data.edge_index.size(1), dtype=torch.bool
            )
            if node_keep is not None:
                edge_keep &= (
                    node_keep[self.base_data.edge_index[0]]
                    & node_keep[self.base_data.edge_index[1]]
                )
            if deleted_edge_indices:
                edge_keep[deleted_edge_indices] = False
            edge_index = self.base_data.edge_index[:, edge_keep]
            if old_to_new is not None:
                edge_index = old_to_new[edge_index]
            edge_attr = self.base_data.edge_attr[edge_keep]
        else:
            edge_index = self.base_data.edge_index
            edge_attr = self.base_data.edge_attr

        batch = torch.zeros(x.size(0), dtype=torch.long)
        with torch.no_grad():
            out = self.model(
                x,
                edge_index,
                batch=batch,
                edge_attr=edge_attr if edge_attr.numel() > 0 else None,
            )
            if out.dim() == 0:
                out = out.unsqueeze(0)
            return float(torch.sigmoid(out).item())
