import numpy as np
import torch
import networkx as nx

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from torch_sparse import SparseTensor

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from mamba_ssm import Mamba

from torch_geometric.utils import degree, sort_edge_index, k_hop_subgraph, index_sort
from torch_geometric.graphgym import cfg
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GATConv
from typing import List
import copy

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from copy import deepcopy

from torch_cluster import random_walk
from torch_ppr import personalized_page_rank, page_rank
from ..utils import TopKRouter, NoisyTopkRouter
import pdb


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer."""

    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        pna_degrees=None,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        bigbird_cfg=None,
        layer=0,
    ):
        super().__init__()

        self.layer = layer
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3

        # Local message-passing model.
        if local_gnn_type == "None":
            self.local_model = None
        elif local_gnn_type == "GENConv":
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == "GINE":
            gin_nn = nn.Sequential(
                Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h)
            )
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == "GAT":
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == "PNA":
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ["mean", "max", "sum"]
            scalers = ["identity"]
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=16,  # dim_h,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == "CustomGatedGCN":
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == "None":
            self.self_attn = None
        elif global_model_type == "Transformer":
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True
            )
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == "Performer":
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False
            )
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif "Mamba" in global_model_type:
            if global_model_type.split("_")[-1] == "NodeMulti":

                num_experts = cfg.model.num_experts
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )

                self.self_attn_node = nn.ParameterList(
                    [
                        Mamba(
                            d_model=dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(num_experts)
                    ]
                )

                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = nn.Linear(dim_h, num_experts)

            elif global_model_type.split("_")[-1] == "NodeLevel":
                self.self_attn_node = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=2,  # Local convolution width
                    expand=1,  # Block expansion factor
                )
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )

            else:
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )
                self.rank_linear = GatedGCNLayer(
                    dim_h, 1, dropout=dropout, residual=False
                )
        else:
            raise ValueError(
                f"Unsupported global x-former model: " f"{global_model_type}"
            )
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            # self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(attn_dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.tanh = F.tanh
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == "CustomGatedGCN":
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                )
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(
                        h, batch.edge_index, batch.edge_attr, batch.pe_EquivStableLapPE
                    )
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type in [
                "Transformer",
                "Performer",
                "BigBird",
                "Mamba",
            ]:
                h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == "Transformer":
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == "Performer":
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == "BigBird":
                h_attn = self.self_attn(h_dense, attention_mask=mask)

            elif self.global_model_type == "Mamba":
                h_attn = self.self_attn(h_dense)[mask]

            elif self.global_model_type == "Mamba_Permute":
                h_ind_perm = permute_within_batch(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == "Mamba_Noise":
                deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                deg_noise = torch.rand_like(deg).to(deg.device)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == "Mamba_RandomGraph_NodeMulti":
                """
                1.Input:batch.edge_index, batch.batch, batch.x
                2.Get random walk for subgraph
                    1. subgraph combine need to pad
                3.Mamba on subgraph
                4.Final token for final layer mamba
                """
                # m random walk compose a subgraph
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                h_attn_gating = self.gating_linear(h)
                h_attn_list = []
                for mod in self.self_attn_node:
                    walks = []
                    for i in range(10):
                        walk = random_walk(row, col, start, walk_length=10)
                        walk = torch.flip(walk, [-1])
                        walks.append(walk)
                    subgraph = torch.cat(walks, dim=1)
                    # getting subgraph
                    subgraph = subgraph.sort(dim=1)[0]
                    unique_x, indices = torch.unique_consecutive(
                        subgraph, return_inverse=True
                    )
                    indices -= indices.min(dim=1, keepdims=True)[0]
                    result = torch.zeros_like(subgraph)
                    result = result.scatter_(1, indices, subgraph)
                    walk = torch.flip(result, [1])

                    h_walk = h[walk]
                    h_attn_list.append(mod(h_walk)[:, -1, :])
                    # mean pooling
                    # h_attn = torch.mean(h_walk, dim=1)
                h_attn_gating = self.softmax(h_attn_gating)
                h_attn = torch.sum(  # weighted sum
                    torch.stack(
                        [
                            h_attn_list[i]
                            * h_attn_gating[..., [i]].expand_as(h_attn_list[i])
                            for i in range(len(h_attn_list))
                        ]
                    ),
                    dim=0,
                )

            elif self.global_model_type == "Mamba_RandomWalk_NodeLevel":
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                walk = random_walk(row, col, start, walk_length=20)
                walk = torch.flip(walk, [-1])

                h_walk = h[walk]
                h = self.self_attn_node(h_walk)
                h_attn = h_walk[:, -1, :]

            elif self.global_model_type == "Mamba_NAG_NodeLevel":
                walk = [batch.x]
                for i in range(1, cfg.prep.neighbor_hops + 1):
                    walk.append(batch[f"neighbors_{i}"])
                h_walk = torch.stack(walk, dim=1)
                # flip h_walk
                h_walk = torch.flip(h_walk, [1])
                h_walk = self.self_attn_node(h_walk)
                h_walk = torch.flip(h_walk, [1])
                for i in range(1, cfg.prep.neighbor_hops + 1):
                    batch[f"neighbors_{i}"] = h_walk[:, i, :]

                h = h_walk[:, 0, :] + h

                if batch.split == "train":
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # Potentially use torch.rand_like?
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )  # why still need to batch?
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):  # permutation
                        # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        # deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

                if torch.isnan(h_attn).any():
                    print("nan")
                    pdb.set_trace()

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)
        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block."""
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = (
            f"summary: dim_h={self.dim_h}, "
            f"local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, "
            f"heads={self.num_heads}"
        )
        return s
