"""Implementation of SparseCINConv layer optimized for MUTAG experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d as BN
from complex import CochainMessagePassingParams
from torch_scatter import scatter_add


class SparseCINCochainConv(nn.Module):
    """Single dimension CIN message passing layer."""
    
    def __init__(
        self,
        dim: int,
        hidden_channels: int = 64,
        eps: float = 0.,
        train_eps: bool = True,
    ):
        super().__init__()
        self.dim = dim
        
        # Message networks (new)
        self.msg_up_nn = Sequential(
            Linear(hidden_channels * 2, hidden_channels),  # Takes concatenated features
            BN(hidden_channels),
            nn.ReLU()
        )
        
        self.msg_boundaries_nn = Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            BN(hidden_channels),
            nn.ReLU()
        )
        
        # Update networks
        self.update_up = Sequential(
            Linear(hidden_channels, hidden_channels),
            BN(hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
            BN(hidden_channels),
            nn.ReLU()
        )
        
        self.update_boundary = Sequential(
            Linear(hidden_channels, hidden_channels),
            BN(hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
            BN(hidden_channels),
            nn.ReLU()
        )
        
        # Combine network
        self.combine = Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            BN(hidden_channels),
            nn.ReLU()
        )
        
        # GIN-style epsilon parameter
        self.eps1 = nn.Parameter(torch.Tensor([eps])) if train_eps else eps
        self.eps2 = nn.Parameter(torch.Tensor([eps])) if train_eps else eps
        
    def forward(self, cochain: CochainMessagePassingParams):
        """Forward pass of SparseCINCochainConv."""
        x = cochain.x
        
        # Upper adjacency messages
        if cochain.up_index is not None:
            # Process each message individually before aggregation
            source_features = x[cochain.up_index[1]]  # j features
            target_features = x[cochain.up_index[0]]  # i features
            messages = self.msg_up_nn(torch.cat([target_features, source_features], dim=-1))
            
            out_up = scatter_add(
                src=messages,
                index=cochain.up_index[0],
                dim=0,
                dim_size=x.size(0)
            )
            out_up = out_up + (1 + self.eps1) * x
            out_up = self.update_up(out_up)
        else:
            out_up = torch.zeros_like(x)
        
        # Boundary messages
        if cochain.boundary_index is not None:
            # Process each message individually before aggregation
            source_features = cochain.boundary_attr[cochain.boundary_index[0]]  # boundary features
            target_features = x[cochain.boundary_index[1]]  # current cell features
            messages = self.msg_boundaries_nn(torch.cat([target_features, source_features], dim=-1))
            
            out_boundary = scatter_add(
                src=messages,
                index=cochain.boundary_index[1],
                dim=0,
                dim_size=x.size(0)
            )
            out_boundary = out_boundary + (1 + self.eps2) * x
            out_boundary = self.update_boundary(out_boundary)
        else:
            out_boundary = torch.zeros_like(x)
        
        # Combine messages
        return self.combine(torch.cat([out_up, out_boundary], dim=-1))


class SparseCINConv(nn.Module):
    """Multi-dimensional CIN layer optimized for MUTAG experiments."""
    
    def __init__(
        self,
        hidden_channels: int = 64,
        max_dim: int = 2,
        eps: float = 0.,
        train_eps: bool = True,
    ):
        super().__init__()
        
        # Create a CochainConv for each dimension
        self.mp_levels = nn.ModuleList([
            SparseCINCochainConv(
                dim=dim,
                hidden_channels=hidden_channels,
                eps=eps,
                train_eps=train_eps
            ) for dim in range(max_dim + 1)
        ])
        
    def forward(self, *cochain_params: CochainMessagePassingParams):
        """
        Forward pass of SparseCINConv.
        
        Parameters
        ----------
        cochain_params : CochainMessagePassingParams
            Parameters for message passing on each dimension
        """
        out = []
        for dim, mp in enumerate(self.mp_levels):
            if dim >= len(cochain_params):
                break
            out.append(mp(cochain_params[dim]))
        return out