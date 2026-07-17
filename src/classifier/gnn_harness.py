"""GNN training harness using PyG.

This is a lightweight training loop for `SimpleGCN` that accepts a list of
NetworkX graphs and labels, converts them via `pyg_adapter`, and trains.
"""
from typing import List, Any
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.classifier.gnn_model import SimpleGCN
from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data


def train_gnn(graphs: List[Any], labels: List[int], epochs: int = 10, batch_size: int = 16) -> Any:
    vocab = build_api_vocab(graphs)
    data_list = [graph_to_pyg_data(G, vocab) for G in graphs]
    for i, d in enumerate(data_list):
        d.y = torch.tensor([labels[i]], dtype=torch.float)

    loader = DataLoader(data_list, batch_size=batch_size)
    model = SimpleGCN(in_channels=max(vocab.values()) + 1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optim.zero_grad()
            out = model(batch.x, batch.edge_index)
            # out is graph-level logits; ensure shape
            if out.dim() == 0:
                out = out.unsqueeze(0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, batch.y)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
        # print per-epoch loss
        print(f"Epoch {ep+1}/{epochs} loss={total_loss:.4f}")
    return model
