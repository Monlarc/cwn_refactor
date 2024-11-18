"""Implementation of SparseCIN optimized for MUTAG experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d as BN, Embedding
from torch_scatter import scatter_add
from complex import ComplexBatch, CochainMessagePassingParams
from sparse_cin_conv import SparseCINConv
from typing import Optional, Tuple
from torch_geometric.nn import JumpingKnowledge, global_add_pool, global_mean_pool

def get_nonlinearity(nonlinearity, return_module=False):
    if nonlinearity == 'relu':
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == 'elu':
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == 'id':
        module = torch.nn.Identity
        function = lambda x: x
    elif nonlinearity == 'sigmoid':
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == 'tanh':
        module = torch.nn.Tanh
        function = torch.tanh
    else:
        raise NotImplementedError('Nonlinearity {} is not currently supported.'.format(nonlinearity))
    if return_module:
        return module
    return function

def get_pooling_fn(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    else:
        raise NotImplementedError('Readout {} is not currently supported.'.format(readout))

def pool_complex(xs, data, max_dim, readout_type):
        pooling_fn = get_pooling_fn(readout_type)
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(max_dim+1, batch_size, xs[0].size(-1),
                                device=batch_size.device)
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
        return pooled_xs

class EmbedABR(nn.Module):
    """Embeds vertices and initializes higher-dim cell features through boundary aggregation."""
    def __init__(
        self,
        v_embed_layer: Embedding,
    ):
        super().__init__()
        self.v_embed_layer = v_embed_layer
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert 1 <= len(cochain_params) <= 3
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        c_params = cochain_params[2] if len(cochain_params) == 3 else None

        # Debug prints
        # print(f"\nProcessing batch:")
        # print(f"Nodes: {v_params.x.shape}")
        # if e_params is not None:
        #     print(f"Edges: {e_params.x.shape}")
        #     print(f"Edge boundary index shape: {e_params.boundary_index.shape}")
        #     print(f"Edge boundary index max values: {e_params.boundary_index.max(dim=1).values}")
        # if c_params is not None:
        #     print(f"Rings: {c_params.x.shape}")
        #     if c_params.boundary_index is not None:
        #         print(f"Ring boundary index shape: {c_params.boundary_index.shape}")
        #         print(f"Ring boundary index max values: {c_params.boundary_index.max(dim=1).values}")

        # Convert one-hot encoded features to indices
        assert v_params.x is not None
        node_type = torch.argmax(v_params.x, dim=1).to(dtype=torch.long)
        
        # Embed vertices (0-cells)
        vx = self.v_embed_layer(node_type)
        out = [vx]
        
        # Process edges if they exist
        if e_params is not None and e_params.boundary_index is not None:
            # Verify indices are within bounds
            assert e_params.boundary_index[0].max() < v_params.x.size(0), \
                f"Node index {e_params.boundary_index[0].max()} is out of bounds (max should be {v_params.x.size(0)-1})"
            
            ex = torch.zeros(
                e_params.x.size(0),  # number of edges
                vx.size(1),          # embedding dimension
                device=vx.device
            )
            
            # Aggregate node features to edges
            ex = scatter_add(
                vx[e_params.boundary_index[0]],  # source node features
                e_params.boundary_index[1],      # target edge indices
                dim=0,
                out=ex
            )
            out.append(ex)
            
            # Process rings if they exist
            if c_params is not None and c_params.boundary_index is not None:
                # Verify indices are within bounds
                assert c_params.boundary_index[0].max() < e_params.x.size(0), \
                    f"Edge index {c_params.boundary_index[0].max()} is out of bounds (max should be {e_params.x.size(0)-1})"
                
                # Initialize ring features
                cx = torch.zeros(
                    c_params.x.size(0),  # number of rings
                    ex.size(1),          # embedding dimension
                    device=ex.device
                )
                
                # Aggregate edge features to rings
                cx = scatter_add(
                    ex[c_params.boundary_index[0]],  # source edge features
                    c_params.boundary_index[1],      # target ring indices
                    dim=0,
                    out=cx
                ) / 2.0
                out.append(cx)
        
        return out

class EmbedSparseCIN(nn.Module):
    """Main model for molecular property prediction using cell complex features."""
    def __init__(
        self,
        atom_types: int,
        out_size: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        max_dim: int = 2,  # 0: atoms, 1: bonds, 2: rings
        dropout: float = 0.5,
        jump_mode: Optional[str] = None,
        nonlinearity: str = 'relu',
        readout: str = 'sum',
        final_readout: str = 'sum',
        apply_dropout_before: str = 'lin1',
        readout_dims: Optional[Tuple[int, ...]] = (0, 1, 2),
        final_hidden_multiplier: int = 2
    ):
        super().__init__()
        
        # Set dimensions to use
        self.max_dim = max_dim
        self.readout_dims = readout_dims
        
        # Initial embeddings
        self.v_embed = Embedding(atom_types, hidden_channels)
        self.abr_embed = EmbedABR(self.v_embed)
        
        # Model parameters
        self.dropout_rate = dropout
        self.apply_dropout_before = apply_dropout_before
        self.jump_mode = jump_mode
        self.readout = readout
        self.final_readout = final_readout

        self.act = get_nonlinearity(nonlinearity)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            SparseCINConv(
                hidden_channels=hidden_channels,
                max_dim=max_dim
            ) for _ in range(num_layers)
        ])
        
        # Jumping knowledge
        self.jump = JumpingKnowledge(jump_mode) if jump_mode is not None else None
        
        # Output layers
        self.lin1s = nn.ModuleList()
        for _ in range(max_dim + 1):
            if jump_mode == 'cat':
                self.lin1s.append(Linear(num_layers * hidden_channels, 
                                       final_hidden_multiplier * hidden_channels,
                                       bias=False))
            else:
                self.lin1s.append(Linear(hidden_channels, 
                                       final_hidden_multiplier * hidden_channels))
        
        self.lin2 = Linear(final_hidden_multiplier * hidden_channels, out_size)
        # self.sigmoid = nn.Sigmoid()
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.abr_embed.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()
    
    def jump_complex(self, jump_xs):
        """Apply jumping knowledge to each level of the complex."""
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs
    
    def forward(self, complex_batch: ComplexBatch):
        
        # Initial embeddings
        params = complex_batch.get_all_cochain_params()
        xs = self.abr_embed(*params)
        complex_batch.set_xs(xs)
        
        # Track features for jumping knowledge
        jump_xs = None
        if self.jump_mode is not None:
            jump_xs = [[] for _ in xs]
        
        # Message passing
        for conv in self.convs:
            params = complex_batch.get_all_cochain_params()
            xs = conv(*params)
            complex_batch.set_xs(xs)
            
            # Collect features for jumping knowledge
            if self.jump_mode is not None:
                for i, x in enumerate(xs):
                    jump_xs[i].append(x)
        
        # Apply jumping knowledge if used
        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)
        
        # Pool each dimension
        xs = pool_complex(xs, complex_batch, self.max_dim, self.readout)
        
        # Select specified dimensions
        xs = [xs[i] for i in self.readout_dims]
        
        # Apply MLPs to each dimension
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(self.act(self.lin1s[self.readout_dims[i]](x)))
        
        # Stack and apply final readout
        x = torch.stack(new_xs, dim=0)
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        else:
            raise NotImplementedError
            
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return self.lin2(x)