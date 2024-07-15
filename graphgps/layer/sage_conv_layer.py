import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
import torch.nn as nn
from torch_geometric.nn import SAGEConv


@register_layer("sage")
class SAGEConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        self.conv = SAGEConv(dim_in, dim_out)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        x = self.conv(x, edge_index)
        batch.x = x
        return batch


"""
1.possible reason for bad performance
    1. reason? lr, batch size, regularization, etc.
"""
