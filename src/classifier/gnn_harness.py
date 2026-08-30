"""GNN training harness using PyG."""
from typing import Any, List

import torch
from torch_geometric.loader import DataLoader

from src.classifier.gnn_model import SimpleGCN
from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data


def train_gnn(graphs: List[Any], labels: List[int], epochs: int = 10, batch_size: int = 16) -> Any:
    vocab = build_api_vocab(graphs)
    data_list = [graph_to_pyg_data(G, vocab) for G in graphs]
    max_edge_dim = max((d.edge_attr.size(-1) for d in data_list if d.edge_attr.numel() > 0), default=5)
    for i, d in enumerate(data_list):
        d.y = torch.tensor([labels[i]], dtype=torch.float)

    loader = DataLoader(data_list, batch_size=batch_size)
    model = SimpleGCN(in_channels=max(vocab.values()) + 1 + 5, hidden=32, edge_dim=max_edge_dim)
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
