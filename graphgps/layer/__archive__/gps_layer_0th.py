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


def hetero_score(x, edge_index):
    src_feat = x
    src_feat_norm = src_feat.norm(dim=1, keepdim=True)
    src_feat_ = src_feat[edge_index[0]]

    _, indices = index_sort(edge_index[1])

    result, _ = to_dense_batch(src_feat_[indices], edge_index[1][indices])
    result = torch.cat(
        [
            result,
            torch.zeros(
                src_feat.shape[0] - result.shape[0], result.shape[1], result.shape[2]
            ).to(result.device),
        ]
    )

    src_feat = src_feat.unsqueeze(1).expand(-1, result.size(1), -1)
    result = torch.einsum("ijk,ijk->ij", src_feat, result)
    result_norm = result.norm(dim=1)

    norm = src_feat_norm.squeeze(1) * result_norm

    result = result / (norm.unsqueeze(1) + 1e-5)

    a = (~torch.eq(result, 0)).sum(dim=1)

    # normalize by src_feat and result
    result = result.sum(dim=1)
    result = result / (a + 1e-5)
    return result


def permute_nodes_within_identity(identities):
    unique_identities, inverse_indices = torch.unique(identities, return_inverse=True)
    node_indices = torch.arange(len(identities), device=identities.device)

    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)

    # Generate random indices within each identity group using torch.randint
    permuted_indices = torch.cat(
        [
            node_indices[mask][torch.randperm(mask.sum(), device=identities.device)]
            for mask in masks
        ]
    )
    return permuted_indices


def sort_rand_gpu(pop_size, num_samples, neighbours):
    # Randomly generate indices and select num_samples in neighbours
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[
        :num_samples
    ]
    neighbours = neighbours[idx_select]
    return neighbours


def augment_seq(edge_index, batch, num_k=-1):
    unique_batches = torch.unique(batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0] == k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(
                torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device)
            )
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))
    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)


def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """

    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out


def lexsort_bi(keys: List[Tensor], dim: int = -1, descending: bool = False) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """

    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    out_bi = torch.flip(out, [0])
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)

        index_bi = k.gather(dim, out_bi)
        index_bi = index_bi.argsort(dim=dim, descending=descending, stable=True)
        out_bi = out_bi.gather(dim, index_bi)
    return out, out_bi


def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[
            torch.randperm(len(indices_in_batch))
        ]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


def permute_within_batch_bi(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []
    permuted_indices_bi = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[
            torch.randperm(len(indices_in_batch))
        ]
        permuted_indices_in_batch_bi = torch.flip(permuted_indices_in_batch, [0])

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)
        permuted_indices_bi.append(permuted_indices_in_batch_bi)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)
    permuted_indices_bi = torch.cat(permuted_indices_bi)

    return permuted_indices, permuted_indices_bi


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
            if global_model_type.split("_")[-1] == "2":
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=8,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                )
            elif global_model_type.split("_")[-1] == "4":
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=4,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=4,  # Block expansion factor
                )
            elif global_model_type.split("_")[-1] == "Multi":
                num_experts = cfg.model.num_experts

                self.self_attn = nn.ParameterList(
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
                self.attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )

                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = MLP(
                    channel_list=[dim_h, dim_h // 2, num_experts], dropout=dropout
                )
                self.rank_linear = nn.Linear(dim_h, 1)
                self.ranker = nn.Linear(dim_h, num_experts)

            elif global_model_type.split("_")[-1] == "Edge":
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )

                self.edge_linear = nn.Linear(dim_h, dim_h)

            elif global_model_type.split("_")[-1] == "GCNMulti":
                num_experts = cfg.model.num_experts
                self.num_experts = num_experts

                self.self_attn = nn.ParameterList(
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
                self.gating_linear = GatedGCNLayer(
                    dim_h, num_experts, dropout=dropout, residual=False
                )
                self.gating_linear_for_gcn = GatedGCNLayer(
                    dim_h, num_experts, dropout=dropout, residual=False
                )
                self.rank_linear = nn.ParameterList(
                    [
                        GatedGCNLayer(dim_h, 1, dropout=dropout, residual=False)
                        for i in range(num_experts)
                    ]
                )

                self.ranker = GatedGCNLayer(
                    dim_h, num_experts, dropout=dropout, residual=False
                )

                self.local_agg = nn.ParameterList(
                    [
                        GatedGCNLayer(dim_h, dim_h, dropout=dropout, residual=True)
                        for i in range(num_experts)
                    ]
                )
                self.local_agg_gat = nn.ParameterList(
                    [
                        GATConv(
                            in_channels=dim_h,
                            out_channels=dim_h // num_heads,
                            heads=num_heads,
                            edge_dim=dim_h,
                        )
                        for i in range(num_experts)
                    ]
                )

                self.score_linear = nn.Linear(dim_h, dim_h)

            elif global_model_type.split("_")[-1] == "Learn":
                self.score_linear = nn.Linear(dim_h, 1)

                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=1,  # Block expansion factor
                )

            elif global_model_type.split("_")[-1] == "MLPMulti":
                num_experts = cfg.model.num_experts

                self.self_attn = nn.ParameterList(
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
                self.gating_linear = MLP(
                    channel_list=[dim_h, num_experts], dropout=dropout
                )

                self.gating_linear_for_gcn = GatedGCNLayer(
                    dim_h, num_experts, dropout=dropout, residual=False
                )
                self.rank_linear = nn.ParameterList(
                    [
                        GatedGCNLayer(dim_h, 1, dropout=dropout, residual=False)
                        for i in range(num_experts)
                    ]
                )

                self.ranker = GatedGCNLayer(
                    dim_h, num_experts, dropout=dropout, residual=False
                )

                self.local_agg = nn.ParameterList(
                    [
                        GatedGCNLayer(dim_h, dim_h, dropout=dropout, residual=True)
                        for i in range(num_experts)
                    ]
                )
                self.local_agg_gat = nn.ParameterList(
                    [
                        GATConv(
                            in_channels=dim_h,
                            out_channels=dim_h // num_heads,
                            heads=num_heads,
                            edge_dim=dim_h,
                        )
                        for i in range(num_experts)
                    ]
                )

            elif global_model_type.split("_")[-1] == "SparseMoE":
                num_experts = cfg.model.num_experts

                self.self_attn = nn.ParameterList(
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

                self.top_k_gate = NoisyTopkRouter(dim_h, num_experts, 2)

            elif global_model_type.split("_")[-1] == "NodeMulti":

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

            elif global_model_type.split("_")[-1] == "Bi":
                self.self_attn = nn.ParameterList(
                    [
                        Mamba(
                            d_model=dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(2)
                    ]
                )
                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = nn.Linear(dim_h, 2)

            elif global_model_type.split("_")[-1] == "MultiBi":
                self.num_experts = cfg.model.num_experts
                self.self_attn = nn.ParameterList(
                    [
                        Mamba(
                            d_model=dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(2 * self.num_experts)
                    ]
                )
                self.softmax = nn.Softmax(dim=-1)
                self.gating_linear = nn.Linear(dim_h, 2 * self.num_experts)

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

            elif global_model_type.split("_")[-1] == "SmallConv":
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=2,  # Local convolution width
                    expand=1,  # Block expansion factor
                )
            elif global_model_type.split("_")[-1] == "SmallState":
                self.self_attn = Mamba(
                    d_model=dim_h,  # Model dimension d_model
                    d_state=8,  # SSM state expansion factor
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

            elif self.global_model_type == "Mamba_Double_Noise":

                deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                deg_double = deg + 100
                h_ = torch.cat([h, h], dim=0)
                deg = torch.cat([deg, deg_double], dim=0)
                deg_noise = torch.rand_like(deg).to(deg.device)
                new_batch = torch.cat([batch.batch, batch.batch], dim=0)
                new_batch = copy.deepcopy(new_batch)
                new_batch.sort()

                h_ind_perm = lexsort([deg + deg_noise, new_batch])
                h_dense, mask = to_dense_batch(h_[h_ind_perm], new_batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)
                h_attn = h_attn[mask][h_ind_perm_reverse][-h.shape[0] :]

            elif self.global_model_type == "Mamba_Double_Katz":

                deg = batch.katz
                deg_double = deg + 100
                h_ = torch.cat([h, h], dim=0)
                deg = torch.cat([deg, deg_double], dim=0)
                new_batch = torch.cat([batch.batch, batch.batch], dim=0)
                new_batch = copy.deepcopy(new_batch)
                new_batch.sort()

                h_ind_perm = lexsort([deg, new_batch])
                h_dense, mask = to_dense_batch(h_[h_ind_perm], new_batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)
                h_attn = h_attn[mask][h_ind_perm_reverse][-h.shape[0] :]

            elif self.global_model_type == "Mamba_Permute_Bi":
                h_ind_perm, h_ind_perm_bi = permute_within_batch_bi(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_dense_bi, mask_bi = to_dense_batch(
                    h[h_ind_perm_bi], batch.batch[h_ind_perm_bi]
                )
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_ind_perm_bi_reverse = torch.argsort(h_ind_perm_bi)

                h_attn_list = []
                h_attn_gating = self.gating_linear(h)
                h_attn_list.append(self.self_attn[0](h_dense)[mask][h_ind_perm_reverse])
                h_attn_list.append(
                    self.self_attn[1](h_dense_bi)[mask_bi][h_ind_perm_bi_reverse]
                )

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

            elif self.global_model_type == "Mamba_Permute_MultiBi":
                h_ind_perm, h_ind_perm_bi = permute_within_batch_bi(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_dense_bi, mask_bi = to_dense_batch(
                    h[h_ind_perm_bi], batch.batch[h_ind_perm_bi]
                )
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_ind_perm_bi_reverse = torch.argsort(h_ind_perm_bi)

                h_attn_list = []
                h_attn_gating = self.gating_linear(h)
                for id, mod in enumerate(self.self_attn):
                    if id % 2 == 0:
                        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    else:
                        h_attn = mod(h_dense_bi)[mask_bi][h_ind_perm_bi_reverse]
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

            elif "Mamba_Hybrid_Degree" == self.global_model_type:
                if batch.split == "train":
                    h_ind_perm = permute_within_batch(batch.batch)
                    # h_ind_perm = permute_nodes_within_identity(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    h_ind_perm_1 = lexsort([deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    if self.global_model_type.split("_")[-1] == "Multi":
                        h_attn_list = []
                        for mod in self.self_attn:
                            mod = mod.to(h_dense.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn)
                        h_attn = sum(h_attn_list) / len(h_attn_list)
                    else:
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        # h_ind_perm = permute_nodes_within_identity(batch.batch)
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                            torch.long
                        )
                        h_ind_perm_1 = lexsort(
                            [deg[h_ind_perm], batch.batch[h_ind_perm]]
                        )
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        if self.global_model_type.split("_")[-1] == "Multi":
                            h_attn_list = []
                            for mod in self.self_attn:
                                mod = mod.to(h_dense.device)
                                h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                                h_attn_list.append(h_attn)
                            h_attn = sum(h_attn_list) / len(h_attn_list)
                        else:
                            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        # h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_Hybrid_Degree_Multi" == self.global_model_type:
                import torch.multiprocessing as mp
                import threading

                if batch.split == "train":
                    h_ind_perm = permute_within_batch(batch.batch)
                    # h_ind_perm = permute_nodes_within_identity(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    h_ind_perm_1 = lexsort([deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    if self.global_model_type.split("_")[-1] == "Multi":
                        h_attn_list = []

                        for mod in self.self_attn:
                            mod = mod.to(h_dense.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn)
                        h_attn = sum(h_attn_list) / len(h_attn_list)
                    else:
                        h_attn = self.self_attn_(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        # h_ind_perm = permute_nodes_within_identity(batch.batch)
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                            torch.long
                        )
                        h_ind_perm_1 = lexsort(
                            [deg[h_ind_perm], batch.batch[h_ind_perm]]
                        )
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        if self.global_model_type.split("_")[-1] == "Multi":
                            h_attn_list = []
                            for mod in self.self_attn:
                                mod = mod.to(h_dense.device)
                                h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                                h_attn_list.append(h_attn)
                            h_attn = sum(h_attn_list) / len(h_attn_list)
                        else:
                            h_attn = self.self_attn_(h_dense)[mask][h_ind_perm_reverse]
                        # h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
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

            elif "Mamba_Degree_Noise_Edge" == self.global_model_type:
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                # Potentially use torch.rand_like?
                # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                # deg_noise = torch.randn(deg.shape).to(deg.device)
                deg_noise = torch.rand_like(deg).to(deg.device)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                # calculate all the edge features
                edge_features = h[batch.edge_index[0]] + h[batch.edge_index[1]]
                edge_features = self.activation(self.edge_linear(edge_features))
                # split the edge features to each graph in a batch
                edge_features, _ = to_dense_batch(
                    edge_features, batch.batch[batch.edge_index[0]]
                )
                h_dense_node, mask = to_dense_batch(
                    h[h_ind_perm], batch.batch[h_ind_perm]
                )  # why still need to batch?

                # concat the edge features to the node features
                h_dense = torch.cat([h_dense_node, edge_features], dim=1)

                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)

                # get node features only
                h_attn = h_attn[:, : h_dense_node.shape[1]][mask][h_ind_perm_reverse]

            elif "Mamba_Degree_Noise_MultiHead_Multi" == self.global_model_type:
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

                    h_attn_list = []
                    for mod in self.self_attn:
                        mod = mod.to(h.device)
                        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                        h_attn_list.append(h_attn)
                    h_attn = sum(h_attn_list) / len(h_attn_list)
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
                        h_attn_list = []
                        for mod in self.self_attn:
                            mod = mod.to(h.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn)
                        h_attn = sum(h_attn_list) / len(h_attn_list)
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif "Mamba_Noise_Bi" == self.global_model_type:
                if batch.split == "train":
                    deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # Potentially use torch.rand_like?
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm, h_ind_perm_bi = lexsort_bi(
                        [deg + deg_noise, batch.batch]
                    )

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )  # why still need to batch?
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_dense_bi, mask_bi = to_dense_batch(
                        h[h_ind_perm_bi], batch.batch[h_ind_perm_bi]
                    )
                    h_ind_perm_bi_reverse = torch.argsort(h_ind_perm_bi)

                    h_attn_list = []
                    h_attn_gating = self.gating_linear(h)
                    h_attn_list.append(
                        self.self_attn[0](h_dense)[mask][h_ind_perm_reverse]
                    )
                    h_attn_list.append(
                        self.self_attn[1](h_dense_bi)[mask_bi][h_ind_perm_bi_reverse]
                    )

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
                        h_ind_perm, h_ind_perm_bi = lexsort_bi(
                            [deg + deg_noise, batch.batch]
                        )
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)

                        h_dense_bi, mask_bi = to_dense_batch(
                            h[h_ind_perm_bi], batch.batch[h_ind_perm_bi]
                        )
                        h_ind_perm_bi_reverse = torch.argsort(h_ind_perm_bi)

                        h_attn_list = []
                        h_attn_gating = self.gating_linear(h)
                        h_attn_list.append(
                            self.self_attn[0](h_dense)[mask][h_ind_perm_reverse]
                        )
                        h_attn_list.append(
                            self.self_attn[1](h_dense_bi)[mask_bi][
                                h_ind_perm_bi_reverse
                            ]
                        )

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

            elif "Mamba_Noise_MultiBi" == self.global_model_type:
                if batch.split == "train":
                    deg = torch.ones(batch.x.shape[0]).to(torch.float).to(h.device)
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # Potentially use torch.rand_like?
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm, h_ind_perm_bi = lexsort_bi(
                        [deg + deg_noise, batch.batch]
                    )

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )  # why still need to batch?
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_dense_bi, mask_bi = to_dense_batch(
                        h[h_ind_perm_bi], batch.batch[h_ind_perm_bi]
                    )
                    h_ind_perm_bi_reverse = torch.argsort(h_ind_perm_bi)

                    h_attn_list = []
                    h_attn_gating = self.gating_linear(h)
                    for id, mod in enumerate(self.self_attn):
                        if id % 2 == 0:
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                        else:
                            h_attn = mod(h_dense_bi)[mask_bi][h_ind_perm_bi_reverse]
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
                        h_ind_perm, h_ind_perm_bi = lexsort_bi(
                            [deg + deg_noise, batch.batch]
                        )
                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)

                        h_dense_bi, mask_bi = to_dense_batch(
                            h[h_ind_perm_bi], batch.batch[h_ind_perm_bi]
                        )
                        h_ind_perm_bi_reverse = torch.argsort(h_ind_perm_bi)

                        h_attn_list = []
                        h_attn_gating = self.gating_linear(h)
                        for id, mod in enumerate(self.self_attn):
                            if id % 2 == 0:
                                h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            else:
                                h_attn = mod(h_dense_bi)[mask_bi][h_ind_perm_bi_reverse]
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

            elif "Mamba_RandomRank_GCNMulti" == self.global_model_type:
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
                    h_ind_perm = lexsort([torch.rand_like(h), batch.batch])

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

            elif "Mamba_Deg2Rank_GCNMulti" == self.global_model_type:
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
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                deg_2 = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                for mod in self.self_attn:
                    h_ind_perm = lexsort([deg + deg_2, batch.batch])

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

            elif "Mamba_12MoE_GCNMulti" == self.global_model_type:
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
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                deg = deg + torch.rand_like(deg).to(deg.device)
                for mod in self.self_attn:
                    h_ind_perm = lexsort([deg, batch.batch])

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

            elif "Mamba_LocalAgg_GCNMulti" == self.global_model_type:
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
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                deg = deg + torch.rand_like(deg).to(deg.device)
                h_ = batch.x
                for mod in self.self_attn:
                    h_ind_perm = lexsort([deg, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)

                    # local aggregation
                    h_agg = self.local_agg[0](
                        Batch(
                            batch=batch,
                            x=h_,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                        )
                    ).x
                    # h_agg = self.local_agg_gat[0](h_, batch.edge_index, batch.edge_attr)
                    h_ = h_ + h_agg

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

            elif "Mamba_LocalAgg_GCNRank_GCNMulti" == self.global_model_type:
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
                h_ = batch.x
                for mod, ranker in zip(self.self_attn, self.rank_linear):

                    rank = ranker(
                        Batch(
                            batch=batch,
                            x=h,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            pe_EquivStableLapPE=es_data,
                        )
                    ).x.squeeze()

                    h_ind_perm = lexsort([rank, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)

                    # local aggregation
                    h_agg = self.local_agg[0](
                        Batch(
                            batch=batch,
                            x=h_,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                        )
                    ).x
                    # h_agg = self.local_agg_gat[0](h_, batch.edge_index, batch.edge_attr)
                    h_ = h_ + h_agg

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

            elif "Mamba_LearnRank_Learn" == self.global_model_type:
                if not batch.if_rank_loss:
                    rank_score = self.score_linear(h).squeeze()
                    h_ind_perm = lexsort([rank_score, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    # random rank of bucket is shared by all model in the same layer
                    rank_score = self.score_linear(h).squeeze()
                    random_rank = torch.randperm(h.shape[0]).to(h.device)
                    bucket_num = 20
                    bucket_size = h.shape[0] // bucket_num
                    bucket_list = {}
                    for i in range(bucket_num):
                        bucket_list[i] = random_rank[
                            i * bucket_size : (i + 1) * bucket_size
                        ]
                    bucket_list[bucket_num] = random_rank[bucket_num * bucket_size :]
                    bucket_score = []
                    for i in range(bucket_num):
                        bucket_score.append(
                            torch.mean(rank_score[bucket_list[i]], dim=0)
                        )
                    bucket_score = torch.stack(bucket_score).to(h.device)
                    random_candidate_number = cfg.model.num_experts

                    indices = (
                        torch.argsort(
                            torch.rand(random_candidate_number, bucket_num), dim=-1
                        )
                        .to(h.device)
                        .long()
                    )

                    bucket_scores = bucket_score.unsqueeze(0).expand(
                        random_candidate_number, -1
                    )

                    candidate_scores = torch.gather(bucket_scores, 1, indices)

                    # rank_loss 算每一个
                    rank_loss = torch.norm(
                        candidate_scores
                        - (torch.arange(bucket_num).to(h.device) / bucket_num)
                        .unsqueeze(0)
                        .expand(random_candidate_number, -1),
                        dim=-1,
                    )
                    batch.rank_loss = rank_loss

                    # according to indices to calculate 5 rank score not related to rank_loss
                    rank_list = []
                    for i in range(random_candidate_number):
                        rank_score_i = torch.zeros(h.shape[0]).to(h.device)
                        bucket_indices = indices[i].tolist()
                        for j, idx in enumerate(bucket_indices):
                            rank_score_i[bucket_list[idx]] = -j
                        rank_i = lexsort([rank_score_i, batch.batch])
                        rank_list.append(rank_i)

                    h_attn_list = []
                    for i in range(cfg.model.num_experts):
                        h_ind_perm = rank_list[i]

                        h_dense, mask = to_dense_batch(
                            h[h_ind_perm], batch.batch[h_ind_perm]
                        )
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        h_attn_list.append(h_attn)

            elif "Mamba_LearnRank_GCNMulti" == self.global_model_type:
                # random shuffle the features
                random_rank = torch.randperm(h.shape[0]).to(h.device)

                # bucket the features
                ## hyper: bucket number
                bucket_num = 20
                bucket_size = h.shape[0] // bucket_num
                bucket_list = {}
                for i in range(bucket_num):
                    bucket_list[i] = random_rank[
                        i * bucket_size : (i + 1) * bucket_size
                    ]
                bucket_list[bucket_num] = random_rank[bucket_num * bucket_size :]

                # calculate bucket score,
                feature_per_node = self.score_linear(h).squeeze()
                bucket_score = []
                for i in range(bucket_num):
                    bucket_score.append(
                        torch.mean(feature_per_node[bucket_list[i]], dim=0)
                    )
                bucket_score = torch.stack(bucket_score).to(h.device)
                total_score = self.tanh(bucket_score[:-1] - bucket_score[1:])
                total_score = torch.sum(total_score)

                # select highest scores
                ## hyper: random candidate number, candidate number
                candidate_num = self.num_experts
                random_candidate_number = 1000

                indices = (
                    torch.argsort(
                        torch.rand(random_candidate_number, bucket_num), dim=-1
                    )
                    .to(h.device)
                    .long()
                )
                indices_expand = indices.unsqueeze(-1).expand(-1, -1, h.shape[-1])
                bucket_score = bucket_score.unsqueeze(0).expand(
                    random_candidate_number, -1, -1
                )
                candidate_scores = torch.gather(bucket_score, 1, indices_expand)
                candidate_scores = self.tanh(
                    candidate_scores[:, :-1] - candidate_scores[:, 1:]
                ).sum(dim=-1)

                candidate_scores = torch.sum(candidate_scores, dim=-1)
                candidate_indices = torch.argsort(candidate_scores, descending=True)[
                    :candidate_num
                ]

                # store the rank in a list usingg indices
                rank_list = []
                for i in candidate_indices:
                    rank_score_i = torch.zeros(h.shape[0]).to(h.device)
                    bucket_indices = indices[i].tolist()
                    for j, idx in enumerate(bucket_indices):
                        rank_score_i[bucket_list[idx]] = -j
                    rank_i = lexsort([rank_score_i, batch.batch])
                    rank_list.append(rank_i)

                # MoE initialize
                h_attn_list = []
                # h_attn_gating = self.gating_linear(
                #     Batch(
                #         batch=batch,
                #         x=h,
                #         edge_index=batch.edge_index,
                #         edge_attr=batch.edge_attr,
                #         pe_EquivStableLapPE=es_data,
                #     )
                # ).x
                h_attn_gating = bucket_score.sum(1)[candidate_indices]

                for id, mod in enumerate(self.self_attn):

                    h_ind_perm = rank_list[id]

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)
                h_attn_gating = nn.Softmax(dim=-2)(h_attn_gating)

                h_attn = torch.sum(  # weighted sum
                    torch.stack(
                        [
                            h_attn_list[i] * h_attn_gating[i].expand_as(h_attn_list[i])
                            for i in range(len(h_attn_list))
                        ]
                    ),
                    dim=0,
                )

            elif "Mamba_LocalAggMoE_GCNMulti" == self.global_model_type:
                h_attn_list = []
                h_gcn_gating = self.gating_linear_for_gcn(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                ).x

                h_list = [h]
                for i in range(len(self.self_attn) - 1):
                    h = self.local_agg[0](
                        Batch(
                            batch=batch,
                            x=h,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            pe_EquivStableLapPE=es_data,
                        )
                    ).x
                    h_list.append(h)

                h_gcn_gating = self.softmax(h_gcn_gating)
                h_gcn = torch.sum(  # weighted sum
                    torch.stack(
                        [
                            h_list[i] * h_gcn_gating[..., [i]].expand_as(h_list[i])
                            for i in range(len(h_list))
                        ]
                    ),
                    dim=0,
                )

                h_attn_gating = self.gating_linear(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                ).x
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                deg = deg + torch.rand_like(deg).to(deg.device)
                h = h_gcn
                for mod in self.self_attn:
                    h_ind_perm = lexsort([deg, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                    # calculate expert head
                    h_attn_list.append(h_attn)

                    # local aggregation
                    h_agg = self.local_agg[0](
                        Batch(
                            batch=batch,
                            x=h,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                        )
                    ).x
                    h = h + h_agg

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

            elif "Mamba_DiverseRank_MLPMulti" == self.global_model_type:
                h_attn_list = []

                h_attn_gating = self.gating_linear(h)

                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                rank_deg = deg + torch.rand_like(deg).to(deg.device)
                rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)

                rank_gcn = []
                for mod in self.rank_linear[:2]:
                    rank_gcn.append(
                        mod(
                            Batch(
                                batch=batch,
                                x=h,
                                edge_index=batch.edge_index,
                                edge_attr=batch.edge_attr,
                                pe_EquivStableLapPE=es_data,
                            )
                        ).x.squeeze()
                    )

                # get pca for each feature
                if torch.isnan(h).any():
                    print("nan in h")
                    nan_mask = torch.isnan(h)
                    h = h.masked_fill(nan_mask, 0)
                pca = PCA(n_components=1)
                pca.fit(h.cpu().detach().numpy())
                h_pca = pca.transform(h.cpu().detach().numpy())
                rank_pca = torch.tensor(h_pca).to(h.device).squeeze()
                rank_pca_neg = -rank_pca

                # hetero rank
                rank_hetero = hetero_score(h, batch.edge_index)

                # random rank
                rank_random = torch.rand_like(deg).to(deg.device)

                rank_list = [
                    rank_deg,
                    # rank_deg_neg,
                    # rank_deg_neg,
                    # rank_deg_2,
                    # rank_deg_2_neg,
                    # rank_ppr,
                    # rank_ppr_neg,
                    rank_pca,
                    # rank_pca_neg,
                    # rank_gcn[0],
                    rank_gcn[1],
                    # rank_hetero,
                    rank_random,
                ]

                for id, (mod, rank) in enumerate(zip(self.self_attn, rank_list)):

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

            elif "Mamba_DiverseRank_Default_MLPMulti" == self.global_model_type:
                h_attn_list = []

                h_attn_gating = self.gating_linear(h)

                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                rank_deg = deg + torch.rand_like(deg).to(deg.device)
                rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)

                # adj_1 = torch.sparse_coo_tensor(
                #     batch.edge_index,
                #     torch.ones(batch.edge_index.shape[1]).to(batch.edge_index.device),
                #     (batch.x.shape[0], batch.x.shape[0]),
                # )
                # adj_2 = adj_1 @ adj_1
                # edge_index_2 = adj_2.coalesce().indices()

                # deg_2 = degree(edge_index_2[0], batch.x.shape[0]).to(torch.float)
                # rank_deg_2 = deg_2 + torch.rand_like(deg_2).to(deg_2.device)
                # rank_deg_2_neg = -deg_2 + torch.rand_like(deg_2).to(deg_2.device)

                # ppr = page_rank(edge_index=batch.edge_index) * 1e7
                # ppr = ppr.to(torch.float).to(h.device)
                # rank_ppr = ppr + torch.rand_like(ppr).to(ppr.device)
                # rank_ppr_neg = -ppr + torch.rand_like(ppr).to(ppr.device)

                rank_gcn = []
                for mod in self.rank_linear[:2]:
                    rank_gcn.append(
                        mod(
                            Batch(
                                batch=batch,
                                x=h,
                                edge_index=batch.edge_index,
                                edge_attr=batch.edge_attr,
                                pe_EquivStableLapPE=es_data,
                            )
                        ).x.squeeze()
                    )

                # get pca for each feature
                if torch.isnan(h).any():
                    print("nan in h")
                    nan_mask = torch.isnan(h)
                    h = h.masked_fill(nan_mask, 0)
                pca = PCA(n_components=1)
                pca.fit(h.cpu().detach().numpy())
                h_pca = pca.transform(h.cpu().detach().numpy())
                rank_pca = torch.tensor(h_pca).to(h.device).squeeze()
                rank_pca_neg = -rank_pca

                # hetero rank
                rank_hetero = hetero_score(h, batch.edge_index)

                # random rank
                rank_random = torch.rand_like(deg).to(deg.device)

                rank_default = torch.zeros_like(deg).to(deg.device)

                rank_list = [
                    rank_deg,
                    # rank_deg_neg,
                    # rank_deg_neg,
                    # rank_deg_2,
                    # rank_deg_2_neg,
                    # rank_ppr,
                    # rank_ppr_neg,
                    # rank_pca,
                    rank_default,
                    # rank_pca_neg,
                    # rank_gcn[0],
                    rank_gcn[1],
                    # rank_hetero,
                    rank_random,
                ]

                for id, (mod, rank) in enumerate(zip(self.self_attn, rank_list)):

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

            elif "Mamba_DiverseRank_GCNMulti" == self.global_model_type:
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

                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                rank_deg = deg + torch.rand_like(deg).to(deg.device)
                rank_deg_neg = -deg + torch.rand_like(deg).to(deg.device)

                rank_gcn = []
                for mod in self.rank_linear[:2]:
                    rank_gcn.append(
                        mod(
                            Batch(
                                batch=batch,
                                x=h,
                                edge_index=batch.edge_index,
                                edge_attr=batch.edge_attr,
                                pe_EquivStableLapPE=es_data,
                            )
                        ).x.squeeze()
                    )

                # hetero rank
                rank_hetero = hetero_score(h, batch.edge_index)

                # random rank
                rank_random = torch.rand_like(deg).to(deg.device)

                rank_list = [
                    rank_deg,
                    rank_deg_neg,
                    rank_gcn[0],
                    rank_random,
                ]

                for id, (mod, rank) in enumerate(zip(self.self_attn, rank_list)):

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

            elif "Mamba_DiverseRank_Katz_PageRank_GCNMulti" == self.global_model_type:
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

                # katz
                katz = batch.katz
                # pagerank
                pagerank = batch.pagerank
                # random rank
                rank_random = torch.rand_like(katz.float()).to(katz.float().device)

                rank_list = [
                    pagerank,
                    katz,
                    # rank_deg_neg,
                    # rank_deg_2,
                    # rank_deg_2_neg,
                    # rank_ppr,
                    # rank_ppr_neg,
                    # rank_pca,
                    # rank_pca_neg,
                    # rank_gcn[0],
                    # rank_gcn[1],
                    # rank_hetero,
                    # rank_random,
                ]

                for id, (mod, rank) in enumerate(zip(self.self_attn, rank_list)):

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

            elif "Mamba_DiverseRank_Katz_Deg_Noise_GCNMulti" == self.global_model_type:
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

                # katz
                katz = batch.katz
                # pagerank
                # pagerank = batch.pagerank
                # random rank
                rank_random = torch.rand_like(katz.float()).to(katz.device)
                katz = katz + torch.rand_like(katz.float()).to(katz.device)
                # pagerank = pagerank + torch.rand_like(pagerank.float()).to(
                #     pagerank.device
                # )

                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                deg = deg + torch.rand_like(deg).to(deg.device)

                rank_list = [
                    # pagerank,
                    katz,
                    deg,
                    # rank_deg_neg,
                    # rank_deg_2,
                    # rank_deg_2_neg,
                    # rank_ppr,
                    # rank_ppr_neg,
                    # rank_pca,
                    # rank_pca_neg,
                    # rank_gcn[0],
                    # rank_gcn[1],
                    # rank_hetero,
                    # rank_random,
                ]

                for id, (mod, rank) in enumerate(zip(self.self_attn, rank_list)):

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

            elif "Mamba_GCNRank" == self.global_model_type:
                rank = self.rank_linear(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                ).x.squeeze()

                h_ind_perm = lexsort([rank, batch.batch])

                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            # elif "Mamba_GCNRank_noise" == self.global_model_type:
            #     rank = self.rank_linear(
            #         Batch(
            #             batch=batch,
            #             x=h,
            #             edge_index=batch.edge_index,
            #             edge_attr=batch.edge_attr,
            #             pe_EquivStableLapPE=es_data,
            #         )
            #     ).x.squeeze()

            #     # randomly permute the rank
            #     rank = rank + torch.rand_like(rank).to(rank.device)

            #     h_ind_perm = lexsort([rank, batch.batch])

            #     h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
            #     h_ind_perm_reverse = torch.argsort(h_ind_perm)
            #     h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif "Mamba_GCNRank_GCNMulti" == self.global_model_type:
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
                for mod, ranker in zip(self.self_attn, self.rank_linear):
                    rank = ranker(
                        (
                            Batch(
                                batch=batch,
                                x=h,
                                edge_index=batch.edge_index,
                                edge_attr=batch.edge_attr,
                                pe_EquivStableLapPE=es_data,
                            )
                        )
                    ).x.squeeze()

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

            elif "Mamba_Random_Average_Multi" == self.global_model_type:
                if batch.split == "train":
                    if self.global_model_type.split("_")[-1] == "Multi":
                        h_attn_list = []
                        gating_list = []
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
                            h_attn_gating = self.gating_linear(h_attn)
                            h_attn_list.append(h_attn)
                        h_attn = sum(h_attn_list) / len(h_attn_list)

                else:
                    deg = (
                        degree(batch.edge_index[0], batch.x.shape[0])
                        .to(torch.float)
                        .to(h.device)
                    )
                    h_attn_list = []
                    gating_list = []
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
                        h_attn_gating = self.gating_linear(h_attn)
                        h_attn_list.append(h_attn)
                    h_attn = sum(h_attn_list) / len(h_attn_list)

            elif "Mamba_Degree_Noise_Average_Multi" == self.global_model_type:
                if batch.split == "train":
                    if self.global_model_type.split("_")[-1] == "Multi":
                        h_attn_list = []
                        h_attn_gating = self.gating_linear(h)

                        for mod in self.self_attn:
                            mod = mod.to(h.device)

                            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                                torch.float
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

                        h_attn = sum(h_attn_list) / len(h_attn_list)
                else:
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    h_attn_list = []
                    h_attn_gating = self.gating_linear(h)

                    for mod in self.self_attn:
                        mod = mod.to(h.device)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                            torch.float
                        )
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
                    h_attn = sum(h_attn_list) / len(h_attn_list)

            elif "Mamba_Degree_Noise_WeightedSum_Multi" == self.global_model_type:
                if batch.split == "train":
                    if self.global_model_type.split("_")[-1] == "Multi":
                        h_attn_list = []
                        h_attn_gating = self.gating_linear(h)

                        for mod in self.self_attn:
                            mod = mod.to(h.device)

                            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                                torch.float
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
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    h_attn_list = []
                    h_attn_gating = self.gating_linear(h)
                    for mod in self.self_attn:
                        mod = mod.to(h.device)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(
                            torch.float
                        )
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

            elif "Mamba_Hybrid_Degree_Noise_Bucket" == self.global_model_type:
                if batch.split == "train":
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    # deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg = deg + deg_noise
                    indices_arr, emb_arr = [], []
                    bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(
                        deg.device
                    )
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign == i).nonzero().squeeze()
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
                    deg_ = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg_).to(deg_.device)
                        # deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg = deg_ + deg_noise
                        indices_arr, emb_arr = [], []
                        bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(
                            deg.device
                        )
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign == i).nonzero().squeeze()
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

            elif "Mamba_Hybrid_Noise" == self.global_model_type:
                if batch.split == "train":
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(
                        batch.batch.device
                    )
                    indices_arr, emb_arr = [], []
                    bucket_assign = torch.randint_like(
                        deg_noise, 0, self.NUM_BUCKETS
                    ).to(deg_noise.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign == i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort(
                            [deg_noise[ind_i], batch.batch[ind_i]]
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
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(
                            batch.batch.device
                        )
                        indices_arr, emb_arr = [], []
                        bucket_assign = torch.randint_like(
                            deg_noise, 0, self.NUM_BUCKETS
                        ).to(deg_noise.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign == i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort(
                                [deg_noise[ind_i], batch.batch[ind_i]]
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

            elif self.global_model_type == "Mamba_RandomWalk_NodeMulti":
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                h_attn_gating = self.gating_linear(h)
                h_attn_list = []
                for mod in self.self_attn_node:
                    walk = random_walk(row, col, start, walk_length=20)

                    h_walk = h[walk]
                    h_walk = mod(h_walk)
                    h_attn_list.append(h_walk[:, -1, :])
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

            elif self.global_model_type == "Mamba_RandomWalk_MeanPooling_NodeMulti":
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                h_attn_gating = self.gating_linear(h)
                h_attn_list = []
                for mod in self.self_attn_node:
                    walk = random_walk(row, col, start, walk_length=20)

                    h_walk = h[walk]
                    h_walk = mod(h_walk)
                    h_attn_list.append(torch.mean(h_walk, dim=1))
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

            elif self.global_model_type == "Mamba_RandomWalk_Staircase_NodeMulti":
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                h_attn_gating = self.gating_linear(h)
                h_attn_list = []
                for i, mod in enumerate(self.self_attn_node):
                    walk = random_walk(row, col, start, walk_length=i * 4 + 1)

                    h_walk = h[walk]
                    h_walk = mod(h_walk)
                    h_attn_list.append(h_walk[:, -1, :])
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

            elif self.global_model_type == "Mamba_Several_RandomWalk_NodeMulti":
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
                    walk_attr_list = []
                    for i in range(10):
                        walk = random_walk(row, col, start, walk_length=10)

                        h_walk = h[walk]
                        h_walk = mod(h_walk)
                        walk_attr_list.append(h_walk[:, -1, :])
                    h_attn_list.append(sum(walk_attr_list) / len(walk_attr_list))
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

            elif (
                self.global_model_type == "Mamba_Several_RandomWalk_Staircase_NodeMulti"
            ):
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
                for i, mod in enumerate(self.self_attn_node):
                    walk_attr_list = []
                    for i in range(10):
                        walk = random_walk(row, col, start, walk_length=i * 4 + 1)

                        h_walk = h[walk]
                        h_walk = mod(h_walk)
                        walk_attr_list.append(h_walk[:, -1, :])
                    h_attn_list.append(sum(walk_attr_list) / len(walk_attr_list))
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

            elif self.global_model_type == "Mamba_Several_Same_RandomWalk_NodeMulti":
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

                for i in range(10):
                    walk = random_walk(row, col, start, walk_length=10)

                    h_walk = h[walk]

                h_attn_gating = self.gating_linear(h)
                h_attn_list = []
                for mod in self.self_attn_node:
                    walk_attr_list = []
                    h_walk = mod(h_walk)
                    h_attn_list.append(h_walk[:, -1, :])
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

            elif self.global_model_type == "Mamba_SubgraphToken_NodeLevel":
                """
                three hyper-param: m (total length of random walk), M (number of each random walk), s (number of each subgraph)
                """
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                m = 10
                S = 5
                M = 3
                nodelevel_repre = []
                for m_tilde in range(m):
                    m_tilde_repr = []
                    for s in range(S):
                        random_walks = []
                        for i in range(M):
                            random_walks.append(
                                random_walk(row, col, start, walk_length=m_tilde)
                            )
                        subgraph = torch.cat(random_walks, dim=1)
                        subgraph_repr = local_encode(subgraph, h)
                        m_tilde_repr.append(subgraph_repr)
                    m_tilde_repr = torch.cat(m_tilde_repr, dim=1)
                    nodelevel_repre.append(m_tilde_repr)
                nodelevel_repre = torch.cat(nodelevel_repre, dim=1)
                h = self.self_attn_node(nodelevel_repre)

            elif self.global_model_type == "Mamba_NodeGCN":
                """
                three hyper-param: m (total length of random walk), M (number of each random walk), s (number of each subgraph)
                """
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE

                nodelevel_repre = []
                nodelevel_repre.append(h)

                h_token = h
                edge_attr_token = batch.edge_attr
                for i, neighbor_encoder in enumerate(self.neighbor_encoder_list):
                    if i < len(self.neighbor_encoder_list) // 2:
                        h_token_out = neighbor_encoder(
                            Batch(
                                batch=batch,
                                x=h_token,
                                edge_index=batch.edge_index,
                                edge_attr=edge_attr_token,
                                pe_EquivStableLapPE=es_data,
                            )
                        )
                    else:
                        h_token_out = neighbor_encoder(
                            Batch(
                                batch=batch,
                                x=h_token,
                                edge_index=batch.edge_index,
                                edge_attr=edge_attr_token,
                                pe_EquivStableLapPE=es_data,
                            )
                        )
                        h_token_out = neighbor_encoder(
                            Batch(
                                batch=batch,
                                x=h_token_out.x,
                                edge_index=batch.edge_index,
                                edge_attr=edge_attr_token,
                                pe_EquivStableLapPE=es_data,
                            )
                        )

                    # GatedGCN does residual connection and dropout internally.
                    h_token = h_token_out.x
                    edge_attr_token = h_token_out.edge_attr
                    nodelevel_repre.append(h_token)
                nodelevel_repre = torch.stack(nodelevel_repre, dim=1)
                h_attn = self.self_attn_node(nodelevel_repre)
                h_attn = h_attn[:, -1, :]

            elif (
                self.global_model_type
                == "Mamba_Hybrid_Degree_Noise_RandomGraph_NodeLevel"
            ):
                # m random walk compose a subgraph
                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)
                walks = []
                for i in range(5):
                    walk = random_walk(row, col, start, walk_length=20)
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
                h_walk = self.self_attn_node(h_walk)
                h = h_walk[:, -1, :] + h

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

            elif (
                self.global_model_type
                == "Mamba_Hybrid_Degree_Noise_RandomWalk_NodeMulti"
            ):

                row = batch.edge_index[0]
                col = batch.edge_index[1]
                start = torch.arange(len(batch.x)).to(row.device)

                h_attn_gating = self.gating_linear(h)
                h_attn_list = []
                adjs = []
                for mod in self.self_attn_node:
                    walk = random_walk(row, col, start, walk_length=20)

                    h_walk = h[walk]
                    h_walk = mod(h_walk)
                    h_attn_list.append(h_walk[:, -1, :])
                h_attn_gating = self.softmax(h_attn_gating)
                h_walk = torch.sum(  # weighted sum
                    torch.stack(
                        [
                            h_attn_list[i]
                            * h_attn_gating[..., [i]].expand_as(h_attn_list[i])
                            for i in range(len(h_attn_list))
                        ]
                    ),
                    dim=0,
                )
                h = h_walk + h

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

            elif self.global_model_type == "Mamba_Augment":
                aug_idx, aug_mask = augment_seq(batch.edge_index, batch.batch, 3)
                h_dense, mask = to_dense_batch(h[aug_idx], batch.batch[aug_idx])
                aug_idx_reverse = torch.nonzero(aug_mask).squeeze()
                h_attn = self.self_attn(h_dense)[mask][aug_idx_reverse]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

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
