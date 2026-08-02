"""PyG GNN model stub. Defines a simple GCN if torch_geometric is available.

This module is intentionally light: training loops and dataset adapters belong in
`harness.py`. If PyG or torch isn't installed, importing this module raises
an informative error.
"""
try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:
    raise ImportError("PyTorch/PyG required for GNN support: install torch and torch-geometric")

from torch import nn


class SimpleGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        _ = edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)
