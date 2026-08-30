"""DMalNet-inspired GNN wrapper with edge-aware graph summary pooling.

This deliberately keeps the public API thin while upgrading the prototype from
raw adjacency-only GCNs to a richer graph encoder with mean/max pooling.
"""
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool, global_max_pool, global_mean_pool
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch/PyG required for GNN support: install torch and torch-geometric") from exc

from torch import nn


class SimpleGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 1, edge_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.edge_dim = edge_dim
        self.conv2 = GATv2Conv(hidden, hidden, heads=2, concat=False, edge_dim=edge_dim if edge_dim is not None else None)
        self.norm = nn.LayerNorm(hidden)
        self.lin = Linear(hidden * 3, out_channels)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        if edge_attr is not None and edge_attr.numel() > 0 and self.edge_dim is not None:
            x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        else:
            x = F.relu(self.conv2(x, edge_index))
        x = self.norm(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        x = torch.cat([mean_pool, max_pool, add_pool], dim=-1)
        return self.lin(x).squeeze(-1)
