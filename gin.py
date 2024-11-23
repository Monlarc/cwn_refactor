import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Module, BatchNorm1d as BN
from torch_geometric.nn import GINEConv
from sparse_cin import get_nonlinearity, get_pooling_fn
from torch.nn import Embedding
from torch_scatter import scatter_add
from complex import CochainMessagePassingParams, ComplexBatch


class EmbedVE(Module):
    """Embeds vertices and initializes edge features through boundary aggregation."""
    def __init__(
        self,
        v_embed_layer: Embedding,
    ):
        super().__init__()
        self.v_embed_layer = v_embed_layer
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert 1 <= len(cochain_params) <= 2  # Only nodes and edges
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None

        # Convert one-hot encoded features to indices
        assert v_params.x is not None
        node_type = torch.argmax(v_params.x, dim=1).to(dtype=torch.long)
        
        # Embed vertices
        vx = self.v_embed_layer(node_type)
        out = [vx]
        
        # Process edges if they exist
        if e_params is not None and e_params.boundary_index is not None:
            # Verify indices are within bounds
            assert e_params.boundary_index[0].max() < v_params.x.size(0), \
                f"Node index {e_params.boundary_index[0].max()} is out of bounds (max should be {v_params.x.size(0)-1})"
            
            # Initialize edge features
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
        
        return out


class EmbedGIN(torch.nn.Module):
    """
    GIN with cell complex inputs to test our pipeline.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, atom_types, bond_types, out_size, num_layers, hidden,
                 dropout_rate: float = 0.5, nonlinearity='relu',
                 readout='sum', train_eps=False, apply_dropout_before='lin2',
                 embed_edge=False, embed_dim=None):
        super(EmbedGIN, self).__init__()

        self.max_dim = 1  # Only nodes and edges

        if embed_dim is None:
            embed_dim = hidden
        
        # Initial embeddings
        self.v_embed = Embedding(atom_types, embed_dim)
        self.ve_embed = EmbedVE(self.v_embed)

        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = get_pooling_fn(readout)
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(layer_dim, hidden),
                            BN(hidden),
                            act_module(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            act_module(),
                        ), train_eps=train_eps))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, out_size)

    def reset_parameters(self):
        self.v_embed.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data: ComplexBatch):
        act = get_nonlinearity(self.nonlinearity, return_module=False)


        # Check input node/edge features are scalars
        # assert data.cochains[0].x.size(-1) == 1
        # if 1 in data.cochains and data.cochains[1].x is not None:
        #     assert data.cochains[1].x.size(-1) == 1

        # Extract node and edge params
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
        # Embed the node and edge features
        xs = list(self.ve_embed(*params))
        # Apply dropout on the input node features
        xs[0] = F.dropout(xs[0], p=self.dropout_rate, training=self.training)
        data.set_xs(xs)

        # We fetch input parameters only at dimension 0 (nodes)
        params = data.get_all_cochain_params(max_dim=0, include_down_features=False)[0]
        x = params.x
        edge_index = params.up_index
        edge_attr = params.kwargs['up_attr']

        # For the edge case when no edges are present
        if edge_index is None:
            edge_index = torch.LongTensor([[], []])
            edge_attr = torch.FloatTensor([[0]*x.size(-1)])

        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Pool only over nodes
        batch_size = data.cochains[0].batch.max() + 1
        x = self.pooling_fn(x, data.nodes.batch, size=batch_size)

        if self.apply_dropout_before == 'lin1':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = act(self.lin1(x))

        if self.apply_dropout_before in ['final_readout', 'lin2']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__