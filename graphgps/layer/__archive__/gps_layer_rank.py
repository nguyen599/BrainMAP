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
from utils import (
    permute_nodes_within_identity,
    sort_rand_gpu,
    augment_seq,
    lexsort,
    lexsort_bi,
    permute_within_batch,
    permute_within_batch_bi,
)
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
        self.bigbird_cfg = bigbird_cfg
        self.dropout = dropout
        self.pna_degrees = pna_degrees
        self.global_model_type = global_model_type
        self.local_gnn_type = local_gnn_type
        self.attn_dropout = attn_dropout

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

        self.choose_global_model(global_model_type)

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

    def choose_global_model(self, global_model_type):
        # Global attention transformer-style model.
        if global_model_type == "None":
            self.self_attn = None
        elif global_model_type == "Transformer":
            self.self_attn = torch.nn.MultiheadAttention(
                self.dim_h, self.num_heads, dropout=self.attn_dropout, batch_first=True
            )
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == "Performer":
            self.self_attn = SelfAttention(
                dim=self.dim_h,
                heads=self.num_heads,
                dropout=self.attn_dropout,
                causal=False,
            )
        elif global_model_type == "BigBird":
            self.bigbird_cfg.dim_hidden = self.dim_h
            self.bigbird_cfg.n_heads = self.num_heads
            self.bigbird_cfg.dropout = self.dropout
            self.self_attn = SingleBigBirdLayer(self.bigbird_cfg)
        elif "Mamba" in global_model_type:
            if global_model_type.split("_")[-1] == "2":
                self.self_attn = Mamba(
                    d_model=self.dim_h,  # Model dimension d_model
                    d_state=8,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                )
            elif global_model_type.split("_")[-1] == "4":
                self.self_attn = Mamba(
                    d_model=self.dim_h,  # Model dimension d_model
                    d_state=4,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=4,  # Block expansion factor
                )
            elif global_model_type.split("_")[-1] == "Multi":
                num_experts = cfg.model.num_experts

                self.self_attn = nn.ParameterList(
                    [
                        Mamba(
                            d_model=self.dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(num_experts)
                    ]
                )
                self.attn = Mamba(
                    d_model=self.dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )

                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = MLP(
                    channel_list=[self.dim_h, self.dim_h // 2, num_experts],
                    dropout=self.dropout,
                )
                self.rank_linear = nn.Linear(self.dim_h, 1)
                self.ranker = nn.Linear(self.dim_h, num_experts)

            elif global_model_type.split("_")[-1] == "GCNMulti":
                num_experts = cfg.model.num_experts
                self.num_experts = num_experts

                self.self_attn = nn.ParameterList(
                    [
                        Mamba(
                            d_model=self.dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(num_experts)
                    ]
                )

                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = GatedGCNLayer(
                    self.dim_h, num_experts, dropout=self.dropout, residual=False
                )
                self.gating_linear_for_gcn = GatedGCNLayer(
                    self.dim_h, num_experts, dropout=self.dropout, residual=False
                )
                self.rank_linear = nn.ParameterList(
                    [
                        GatedGCNLayer(
                            self.dim_h, 1, dropout=self.dropout, residual=False
                        )
                        for i in range(num_experts)
                    ]
                )

                self.ranker = GatedGCNLayer(
                    self.dim_h, num_experts, dropout=self.dropout, residual=False
                )

                self.local_agg = nn.ParameterList(
                    [
                        GatedGCNLayer(
                            self.dim_h, self.dim_h, dropout=self.dropout, residual=True
                        )
                        for i in range(num_experts)
                    ]
                )
                self.local_agg_gat = nn.ParameterList(
                    [
                        GATConv(
                            in_channels=self.dim_h,
                            out_channels=self.dim_h // self.num_heads,
                            heads=self.num_heads,
                            edge_dim=self.dim_h,
                        )
                        for i in range(num_experts)
                    ]
                )

                self.score_linear = nn.Linear(self.dim_h, self.dim_h)

            elif global_model_type.split("_")[-1] == "MLPMulti":
                num_experts = cfg.model.num_experts

                self.self_attn = nn.ParameterList(
                    [
                        Mamba(
                            d_model=self.dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(num_experts)
                    ]
                )

                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = MLP(
                    channel_list=[self.dim_h, num_experts], dropout=self.dropout
                )

                self.gating_linear_for_gcn = GatedGCNLayer(
                    self.dim_h, num_experts, dropout=self.dropout, residual=False
                )
                self.rank_linear = nn.ParameterList(
                    [
                        GatedGCNLayer(
                            self.dim_h, 1, dropout=self.dropout, residual=False
                        )
                        for i in range(num_experts)
                    ]
                )

                self.ranker = GatedGCNLayer(
                    self.dim_h, num_experts, dropout=self.dropout, residual=False
                )

                self.local_agg = nn.ParameterList(
                    [
                        GatedGCNLayer(
                            self.dim_h, self.dim_h, dropout=self.dropout, residual=True
                        )
                        for i in range(num_experts)
                    ]
                )
                self.local_agg_gat = nn.ParameterList(
                    [
                        GATConv(
                            in_channels=self.dim_h,
                            out_channels=self.dim_h // self.num_heads,
                            heads=self.num_heads,
                            edge_dim=self.dim_h,
                        )
                        for i in range(num_experts)
                    ]
                )

            elif global_model_type.split("_")[-1] == "SparseMoE":
                num_experts = cfg.model.num_experts

                self.self_attn = nn.ParameterList(
                    [
                        Mamba(
                            d_model=self.dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(num_experts)
                    ]
                )

                self.top_k_gate = NoisyTopkRouter(self.dim_h, num_experts, 2)

            elif global_model_type.split("_")[-1] == "SmallConv":
                self.self_attn = Mamba(
                    d_model=self.dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=2,  # Local convolution width
                    expand=1,  # Block expansion factor
                )
            elif global_model_type.split("_")[-1] == "SmallState":
                self.self_attn = Mamba(
                    d_model=self.dim_h,  # Model dimension d_model
                    d_state=8,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )
            else:
                self.self_attn = Mamba(
                    d_model=self.dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )
                self.rank_linear = GatedGCNLayer(
                    self.dim_h, 1, dropout=self.dropout, residual=False
                )
        else:
            raise ValueError(
                f"Unsupported global x-former model: " f"{global_model_type}"
            )
        self.global_model_type = global_model_type

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

            elif self.global_model_type == "Mamba_Noise":
                deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                deg_noise = torch.rand_like(deg).to(deg.device)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == "Mamba_Degree":
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                # indcies that sort by batch and then deg, by ascending order
                h_ind_perm = lexsort([deg, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == "Mamba_Hybrid":
                if batch.split == "train":
                    h_ind_perm = permute_within_batch(batch.batch)
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_Hybrid_Degree_Noise" == self.global_model_type:
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

            elif "Mamba_Degree_Noise" == self.global_model_type:
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

            elif "Mamba_PPR" == self.global_model_type:
                if batch.split == "train":
                    deg = page_rank(edge_index=batch.edge_index.to("cpu")) * 1e7
                    deg = deg.to(torch.float).to(h.device)

                    if deg.shape[0] != h.shape[0]:
                        # padding
                        deg = torch.cat(
                            [deg, torch.zeros(h.shape[0] - deg.shape[0]).to(deg.device)]
                        )
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # Potentially use torch.rand_like?
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # deg_noise = torch.randn(deg.shape).to(deg.device)
                    h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = page_rank(edge_index=batch.edge_index.to("cpu")) * 1e7
                    deg = deg.to(torch.float).to(h.device)
                    if deg.shape[0] != h.shape[0]:
                        # padding
                        deg = torch.cat(
                            [deg, torch.zeros(h.shape[0] - deg.shape[0]).to(deg.device)]
                        )

                    for i in range(5):
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

            elif "Mamba_Noise_SameIn_WeightedSum_Multi" == self.global_model_type:
                if batch.split == "train":
                    deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )  # why still need to batch?
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_attn_list = []
                    h_attn_gating = self.gating_linear(h)
                    for mod in self.self_attn:
                        mod = mod.to(h.device)
                        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                        h_attn_list.append(h_attn)
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
                else:
                    mamba_arr = []
                    deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                    for i in range(5):  # permutation
                        # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        # deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn_list = []
                        h_attn_gating = self.gating_linear(h)
                        for mod in self.self_attn:
                            mod = mod.to(h.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn)
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

                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_Noise_SameIn_SparseMoE" == self.global_model_type:
                if batch.split == "train":
                    deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )  # why still need to batch?
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_attn_list = []
                    h_attn_gating, indices = self.top_k_gate(h)
                    for mod in self.self_attn:
                        mod = mod.to(h.device)
                        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                        h_attn_list.append(h_attn)

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
                else:
                    mamba_arr = []
                    deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                    for i in range(5):  # permutation
                        # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        # deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn_list = []
                        h_attn_gating, indices = self.top_k_gate(h)
                        for mod in self.self_attn:
                            mod = mod.to(h.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn)
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

                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_Random_WeightedSum_Multi" == self.global_model_type:
                if batch.split == "train":
                    if self.global_model_type.split("_")[-1] == "Multi":
                        h_attn_list = []
                        h_attn_gating = self.gating_linear(h)
                        for mod in self.self_attn:
                            mod = mod.to(h.device)

                            deg = (
                                torch.ones(batch.x.shape[0])
                                .to(torch.float)
                                .to(h.device)
                            )

                            deg_noise = torch.rand_like(deg).to(deg.device)
                            h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                            h_dense, mask = to_dense_batch(
                                h[h_ind_perm], batch.batch[h_ind_perm]
                            )
                            h_ind_perm_reverse = torch.argsort(h_ind_perm)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            # calculate expert head
                            h_attn_list.append(h_attn)
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
                else:
                    deg = (
                        degree(batch.edge_index[0], batch.x.shape[0])
                        .to(torch.float)
                        .to(h.device)
                    )
                    h_attn_list = []
                    h_attn_gating = self.gating_linear(h)
                    for mod in self.self_attn:
                        mod = mod.to(h.device)
                        # deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                        #     torch.float
                        # )
                        deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
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

                        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                        # calculate expert head
                        h_attn_list.append(h_attn)
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

            elif "Mamba_MultiRank_Multi" == self.global_model_type:
                h_attn_list = []
                h_attn_gating = self.gating_linear(h)
                for mod, ranker in zip(self.self_attn, self.rank_linear):
                    rank = ranker(h).squeeze()
                    h_ind_perm = lexsort([rank, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)
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

            elif "Mamba_DegRank_noise" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                rank_deg = deg + torch.rand_like(deg).to(deg.device)

                h_ind_perm = lexsort([rank_deg, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_DegRank" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                rank_deg = deg

                h_ind_perm = lexsort([rank_deg, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_DegRank_SameModel_Multi" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)

                h_attn_list = []
                h_attn_gating = []
                for i in range(len(self.self_attn)):
                    rank_deg = deg + torch.rand_like(deg).to(deg.device)
                    rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([rank_deg, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.attn(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_res = h_attn - torch.cat(
                        [h_attn[:-1], h_attn[-2].unsqueeze(0)]
                    )
                    h_attn_gating.append(self.rank_linear(h_attn_res))
                    h_attn_list.append(h_attn)
                h_attn_gating = torch.cat(h_attn_gating, dim=1)
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

            elif "Mamba_DegRank_SameModel_1_Multi" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)

                h_attn_list = []
                h_attn_gating = []
                for i in range(len(self.self_attn)):
                    rank_deg = deg + torch.rand_like(deg).to(deg.device)
                    rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([rank_deg, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.attn(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_res = h_attn - torch.cat(
                        [h_attn[:-1], h_attn[-2].unsqueeze(0)]
                    )
                    h_attn_gating.append(self.rank_linear(h_attn_res).mean(dim=0))
                    h_attn_list.append(h_attn)
                h_attn_gating = torch.cat(h_attn_gating)
                h_attn_gating = self.softmax(h_attn_gating)
                h_attn = torch.sum(  # weighted sum
                    torch.stack(
                        [
                            h_attn_list[i] * h_attn_gating[i]
                            for i in range(len(h_attn_list))
                        ]
                    ),
                    dim=0,
                )

            elif "Mamba_DegRank_SameModel_2_Multi" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                h_attn_list = []
                h_attn_gating = []

                if self.training:
                    for i in range(len(self.self_attn)):
                        rank_deg = deg + torch.rand_like(deg).to(deg.device)
                        rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([rank_deg, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.attn(h_dense)[mask][h_ind_perm_reverse]
                        # calculate expert head
                        h_attn_res = h_attn - torch.cat(
                            [h_attn[:-1], h_attn[-2].unsqueeze(0)]
                        )
                        h_attn_gating.append(self.rank_linear(h_attn_res).mean(dim=0))
                        h_attn_list.append(h_attn)
                else:
                    for i in range(len(self.self_attn) * 3):
                        rank_deg = deg + torch.rand_like(deg).to(deg.device)
                        rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([rank_deg, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.attn(h_dense)[mask][h_ind_perm_reverse]
                        # calculate expert head
                        h_attn_res = h_attn - torch.cat(
                            [h_attn[:-1], h_attn[-2].unsqueeze(0)]
                        )
                        h_attn_gating.append(self.rank_linear(h_attn_res).mean(dim=0))
                        h_attn_list.append(h_attn)
                h_attn_gating = torch.cat(h_attn_gating)
                h_attn_gating = self.softmax(h_attn_gating)
                h_attn = torch.sum(  # weighted sum
                    torch.stack(
                        [
                            h_attn_list[i] * h_attn_gating[i]
                            for i in range(len(h_attn_list))
                        ]
                    ),
                    dim=0,
                )

            elif "Mamba_DegRank_GCNMulti" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                rank_deg = deg + torch.rand_like(deg).to(deg.device)
                rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)

                h_attn_list = []
                h_attn_gating = self.gating_linear(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                ).x
                for mod in self.self_attn:
                    h_ind_perm = lexsort([rank_deg, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)
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
            elif "Mamba_PCARank" == self.global_model_type:
                pca = PCA(n_components=1)
                pca.fit(h.cpu().detach().numpy())
                h_pca = pca.transform(h.cpu().detach().numpy())
                rank_pca = torch.tensor(h_pca).to(h.device).squeeze()

                h_ind_perm = lexsort([rank_pca, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_PCARank_GCNMulti" == self.global_model_type:
                # get pca for each feature
                pca = PCA(n_components=1)
                pca.fit(h.cpu().detach().numpy())
                h_pca = pca.transform(h.cpu().detach().numpy())
                rank_pca = torch.tensor(h_pca).to(h.device).squeeze()

                h_attn_list = []
                h_attn_gating = self.gating_linear(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                ).x
                for mod in self.self_attn:
                    h_ind_perm = lexsort([rank_pca, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)
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

            elif "Mamba_HeteroRank" == self.global_model_type:
                h_ind_perm = lexsort([hetero_score(h, batch.edge_index), batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_EigenCentralityRank" == self.global_model_type:
                # transform edge_index to networkx graph, node number should be the same as h
                eigen_centrality = batch.eigenvector
                h_ind_perm = lexsort([eigen_centrality, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif (
                "Mamba_Hybrid_Noise_EigenCentralityRank_Bucket"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.eigenvector
                    eigen_centrality = eigen_centrality + torch.rand_like(
                        eigen_centrality
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.eigenvector
                        eigen_centrality = eigen_centrality + torch.rand_like(
                            eigen_centrality
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_KatzCentralityRank" == self.global_model_type:
                eigen_centrality = batch.katz
                h_ind_perm = lexsort([eigen_centrality, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif (
                "Mamba_Hybrid_Noise_KatzCentralityRank_Bucket3"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.katz
                    eigen_centrality = eigen_centrality.float() + torch.rand_like(
                        eigen_centrality.float()
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.katz
                        eigen_centrality = eigen_centrality.float() + torch.rand_like(
                            eigen_centrality.float()
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif (
                "Mamba_Hybrid_Noise_KatzCentralityRank_Bucket2"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.katz
                    eigen_centrality = eigen_centrality.float() + torch.rand_like(
                        eigen_centrality.float()
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.katz
                        eigen_centrality = eigen_centrality.float() + torch.rand_like(
                            eigen_centrality.float()
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif (
                "Mamba_Hybrid_Noise_KatzCentralityRank_Bucket4"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.katz
                    eigen_centrality = eigen_centrality.float() + torch.rand_like(
                        eigen_centrality.float()
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.katz
                        eigen_centrality = eigen_centrality.float() + torch.rand_like(
                            eigen_centrality.float()
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_PagerankCentralityRank" == self.global_model_type:
                eigen_centrality = batch.pagerank
                h_ind_perm = lexsort([eigen_centrality, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif (
                "Mamba_Hybrid_Noise_PagerankCentralityRank_Bucket"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.pagerank
                    eigen_centrality = eigen_centrality.float() + torch.rand_like(
                        eigen_centrality
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.pagerank
                        eigen_centrality = eigen_centrality + torch.rand_like(
                            eigen_centrality
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_LaplacianCentralityRank" == self.global_model_type:
                eigen_centrality = batch.laplacian
                h_ind_perm = lexsort([eigen_centrality, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif (
                "Mamba_Hybrid_Noise_LaplacianCentralityRank_Bucket"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.laplacian
                    eigen_centrality = eigen_centrality + torch.rand_like(
                        eigen_centrality
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.laplacian
                        eigen_centrality = eigen_centrality + torch.rand_like(
                            eigen_centrality
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_SubgraphCentralityRank" == self.global_model_type:
                eigen_centrality = batch.subgraphcentrality
                h_ind_perm = lexsort([eigen_centrality, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif (
                "Mamba_Hybrid_Noise_SubgraphCentralityRank_Bucket"
                == self.global_model_type
            ):
                if self.training:
                    eigen_centrality = batch.subgraph
                    eigen_centrality = eigen_centrality + torch.rand_like(
                        eigen_centrality
                    ).to(eigen_centrality.device)
                    h_ind_perm = lexsort([eigen_centrality, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        eigen_centrality = batch.subgraph
                        eigen_centrality = eigen_centrality + torch.rand_like(
                            eigen_centrality
                        ).to(eigen_centrality.device)
                        h_ind_perm = lexsort([eigen_centrality, batch.batch])

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_LoadCentralityRank" == self.global_model_type:
                # transform edge_index to networkx graph, node number should be the same as h
                G = nx.Graph()
                G.add_nodes_from(range(h.shape[0]))
                G.add_edges_from(batch.edge_index.T.cpu().numpy())
                eigen_centrality = nx.load_centrality(G)
                eigen_centrality = torch.tensor(
                    [eigen_centrality[i] for i in range(h.shape[0])]
                ).to(h.device)
                h_ind_perm = lexsort([eigen_centrality, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_ClusteringRank" == self.global_model_type:
                # transform edge_index to networkx graph, node number should be the same as h
                clustering_rank = batch.clustering

                h_ind_perm = lexsort([clustering_rank, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_RandomWalkRank" == self.global_model_type:
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = batch.ptr[:-1]
                start = torch.randperm(h.shape[0])[:1000].to(h.device)
                walk = random_walk(row, col, start, walk_length=20).to(h.device)
                rank = torch.zeros(h.shape[0], device=h.device)
                for i in range(20):
                    rank[walk[:, i]] = (i + 1) * torch.arange(1, 1001).float().to(
                        h.device
                    )
                rank = rank + torch.rand_like(rank).to(rank.device)

                h_ind_perm = lexsort([rank, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_HeteroRank_GCNMulti" == self.global_model_type:
                h_attn_list = []
                h_attn_gating = self.gating_linear(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                ).x
                for mod in self.self_attn:
                    h_ind_perm = lexsort(
                        [hetero_score(h, batch.edge_index), batch.batch]
                    )

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)
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

            elif "Mamba_Hybrid_Noise_Bucket" == self.global_model_type:
                if batch.split == "train":
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(
                        batch.batch.device
                    )
                    h_ind_perm = lexsort([deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(
                            batch.batch.device
                        )
                        h_ind_perm = lexsort([deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == "Mamba_Eigen":
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                centrality = batch.EigCentrality
                if batch.split == "train":
                    # Shuffle within 1 STD
                    centrality_noise = torch.std(centrality) * torch.rand(
                        centrality.shape
                    ).to(centrality.device)
                    # Order by batch, degree, and centrality
                    h_ind_perm = lexsort([centrality + centrality_noise, batch.batch])
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality) * torch.rand(
                            centrality.shape
                        ).to(centrality.device)
                        h_ind_perm = lexsort(
                            [centrality + centrality_noise, batch.batch]
                        )
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_Eigen_Bucket" == self.global_model_type:
                centrality = batch.EigCentrality
                if batch.split == "train":
                    centrality_noise = torch.std(centrality) * torch.rand(
                        centrality.shape
                    ).to(centrality.device)
                    indices_arr, emb_arr = [], []
                    bucket_assign = torch.randint_like(
                        centrality, 0, self.NUM_BUCKETS
                    ).to(centrality.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign == i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort(
                            [(centrality + centrality_noise)[ind_i], batch.batch[ind_i]]
                        )
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm_i], batch.batch[h_ind_perm_i]
                        )
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality) * torch.rand(
                            centrality.shape
                        ).to(centrality.device)
                        indices_arr, emb_arr = [], []
                        bucket_assign = torch.randint_like(
                            centrality, 0, self.NUM_BUCKETS
                        ).to(centrality.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign == i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort(
                                [
                                    (centrality + centrality_noise)[ind_i],
                                    batch.batch[ind_i],
                                ]
                            )
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(
                                h[h_ind_perm_i], batch.batch[h_ind_perm_i]
                            )
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == "Mamba_RWSE":
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                RWSE_sum = torch.sum(batch.pestat_RWSE, dim=1)
                if batch.split == "train":
                    # Shuffle within 1 STD
                    RWSE_noise = torch.std(RWSE_sum) * torch.randn(RWSE_sum.shape).to(
                        RWSE_sum.device
                    )
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    h_ind_perm = lexsort([-RWSE_sum + RWSE_noise, batch.batch])
                    # h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, deg, batch.batch])
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    # h_ind_perm = lexsort([-RWSE_sum, deg, batch.batch])
                    mamba_arr = []
                    for i in range(5):
                        RWSE_noise = torch.std(RWSE_sum) * torch.randn(
                            RWSE_sum.shape
                        ).to(RWSE_sum.device)
                        h_ind_perm = lexsort([-RWSE_sum + RWSE_noise, batch.batch])
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == "Mamba_Cluster":
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == "train":
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = (
                        torch.zeros(batch.LouvainCluster.shape)
                        .long()
                        .to(batch.LouvainCluster.device)
                    )
                    random_permute = (
                        torch.randperm(unique_cluster_n + 1)
                        .long()
                        .to(batch.LouvainCluster.device)
                    )
                    for i in range(unique_cluster_n):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    # h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    # h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm_1 = lexsort(
                        [permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]]
                    )
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    # h_ind_perm = lexsort([batch.LouvainCluster, deg, batch.batch])
                    # h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    # h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    # h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                    mamba_arr = []
                    for i in range(5):
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = (
                            torch.zeros(batch.LouvainCluster.shape)
                            .long()
                            .to(batch.LouvainCluster.device)
                        )
                        random_permute = (
                            torch.randperm(unique_cluster_n + 1)
                            .long()
                            .to(batch.LouvainCluster.device)
                        )
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        # potentially permute it 5 times and average
                        # on the cluster level
                        # h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        # h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm_1 = lexsort(
                            [permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]]
                        )
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == "Mamba_Hybrid_Degree_Bucket":
                if batch.split == "train":
                    h_ind_perm = permute_within_batch(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    indices_arr, emb_arr = [], []
                    for i in range(self.NUM_BUCKETS):
                        ind_i = h_ind_perm[h_ind_perm % self.NUM_BUCKETS == i]
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm_i], batch.batch[h_ind_perm_i]
                        )
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                            torch.long
                        )
                        indices_arr, emb_arr = [], []
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm % self.NUM_BUCKETS == i]
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(
                                h[h_ind_perm_i], batch.batch[h_ind_perm_i]
                            )
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == "Mamba_Cluster_Bucket":
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == "train":
                    indices_arr, emb_arr = [], []
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = (
                        torch.zeros(batch.LouvainCluster.shape)
                        .long()
                        .to(batch.LouvainCluster.device)
                    )
                    random_permute = (
                        torch.randperm(unique_cluster_n + 1)
                        .long()
                        .to(batch.LouvainCluster.device)
                    )
                    for i in range(len(torch.unique(batch.LouvainCluster))):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    for i in range(self.NUM_BUCKETS):
                        ind_i = h_ind_perm[h_ind_perm % self.NUM_BUCKETS == i]
                        h_ind_perm_sort = lexsort(
                            [permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]]
                        )
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm_i], batch.batch[h_ind_perm_i]
                        )
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        indices_arr, emb_arr = [], []
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = (
                            torch.zeros(batch.LouvainCluster.shape)
                            .long()
                            .to(batch.LouvainCluster.device)
                        )
                        random_permute = (
                            torch.randperm(unique_cluster_n + 1)
                            .long()
                            .to(batch.LouvainCluster.device)
                        )
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm % self.NUM_BUCKETS == i]
                            h_ind_perm_sort = lexsort(
                                [
                                    permuted_louvain[ind_i],
                                    deg[ind_i],
                                    batch.batch[ind_i],
                                ]
                            )
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(
                                h[h_ind_perm_i], batch.batch[h_ind_perm_i]
                            )
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

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

            if not batch.if_rank_loss:
                h_attn = self.dropout_attn(h_attn)
                h_attn = h_in1 + h_attn  # Residual connection.
                if self.layer_norm:
                    h_attn = self.norm1_attn(h_attn, batch.batch)
                if self.batch_norm:
                    h_attn = self.norm1_attn(h_attn)
                h_out_list.append(h_attn)
            else:
                h_out_lists = []
                for i in range(cfg.model.num_experts):
                    h_attn = self.dropout_attn(h_attn_list[i])
                    h_attn = h_in1 + h_attn
                    if self.layer_norm:
                        h_attn = self.norm1_attn(h_attn, batch.batch)
                    if self.batch_norm:
                        h_attn = self.norm1_attn(h_attn)
                    h_out_lists.append([h_out_list[0], h_attn])

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        if not batch.if_rank_loss:
            h = sum(h_out_list)
            # Feed Forward block.
            h = h + self._ff_block(h)
            if self.layer_norm:
                h = self.norm2(h, batch.batch)
            if self.batch_norm:
                h = self.norm2(h)
            batch.x = h
            return batch
        else:
            batchs = []
            for i in range(cfg.model.num_experts):
                h = sum(h_out_lists[i])
                h = h + self._ff_block(h)
                if self.layer_norm:
                    h = self.norm2(h, batch.batch)
                if self.batch_norm:
                    h = self.norm2(h)
                batch.x = h
                batchs.append(batch)
            return batchs

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
