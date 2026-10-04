"""DMalNet-inspired GNN wrapper with edge-aware graph summary pooling.

Mean/max pooling is the default; attention pooling is available for controlled
experiments without changing existing model behavior.
"""
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Linear
    from torch_geometric.nn import (
        GATv2Conv,
        GCNConv,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )
    from torch_geometric.utils import softmax
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch/PyG required for GNN support: install torch and torch-geometric") from exc

from torch import nn


class SimpleGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        out_channels: int = 1,
        edge_dim: Optional[int] = None,
        pooling: str = "mean_max",
    ):
        super().__init__()
        if pooling not in {"mean_max", "attention"}:
            raise ValueError(f"Unsupported graph pooling method: {pooling}")
        self.conv1 = GCNConv(in_channels, hidden)
        self.edge_dim = edge_dim
        self.conv2 = GATv2Conv(hidden, hidden, heads=2, concat=False, edge_dim=edge_dim if edge_dim is not None else None)
        self.norm = nn.LayerNorm(hidden)
        self.pooling = pooling
        self.pooling_gate = Linear(hidden, 1) if pooling == "attention" else None
        pooled_channels = hidden if pooling == "attention" else hidden * 2
        self.lin = Linear(pooled_channels, out_channels)

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index))
        if edge_attr is not None and edge_attr.numel() > 0 and self.edge_dim is not None:
            x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        else:
            x = F.relu(self.conv2(x, edge_index))
        x = self.norm(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if getattr(self, "pooling", "mean_max") == "attention":
            scores = self.pooling_gate(x).squeeze(-1)
            weights = softmax(scores, batch)
            x = global_add_pool(x * weights.unsqueeze(-1), batch)
        else:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            x = torch.cat([mean_pool, max_pool], dim=-1)
        return self.lin(x).squeeze(-1)