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
from ..utils import TopKRouter, NoisyTopkRouter, Top1Router
from .utils import (
    permute_nodes_within_identity,
    sort_rand_gpu,
    augment_seq,
    lexsort,
    lexsort_bi,
    permute_within_batch,
    permute_within_batch_bi,
    hetero_score,
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

        # if cfg.model.learn_rank:
        #     cfg.model["opt_node_rank_dict"] = {}
        #     cfg.model["node_rank_dict"] = {}

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

            elif global_model_type.split("_")[-1] == "MoE":
                self.num_experts = cfg.model.num_experts
                self.self_attn = None

                self.experts_mamba = nn.ParameterList(
                    [
                        Mamba(
                            d_model=self.dim_h,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=1,  # Block expansion factor
                        )
                        for i in range(self.num_experts)
                    ]
                )
               
                self.router = Top1Router(self.dim_h, self.num_experts)

                self.gcn_for_router = GatedGCNLayer(
                    self.dim_h, self.dim_h, dropout=self.dropout, residual=False
                )


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
                    use_fast_path=False,
                )
                self.calculate_rank_score = nn.Linear(self.dim_h, 1)

        else:
            raise ValueError(
                f"Unsupported global x-former model: " f"{global_model_type}"
            )
        self.global_model_type = global_model_type

    def forward(
        self,
        batch,
        opt_rank=True,
    ):
        self.opt_rank = opt_rank
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
        if self.self_attn or self.experts_mamba is not None:
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

            elif self.global_model_type == "Mamba_Degree_Noise":
                """
                TODO:
                1. two random seqs, the same graph as the positive pair, different graphs as the negative pair, 
                2. input: h: [N, D], batch: [000,111,..,128128], 
                3. input: h: [2 * N, D], batch: [000,111,..,256], 
                """
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                deg_noise = torch.rand_like(deg).to(deg.device)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])

                h_dense, mask = to_dense_batch(
                    h[h_ind_perm], batch.batch[h_ind_perm]
                )  # why still need to batch?
                h_ind_perm_reverse = torch.argsort(h_ind_perm)

                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

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
                    if cfg.model.get("mamba_interpret", False):
                        h_attn, interpret_score = self.self_attn(
                            h_dense, interpret=True
                        )
                        h_attn = h_attn[mask][h_ind_perm_reverse]
                        interpret_score = interpret_score.unsqueeze(-1)[mask][
                            h_ind_perm_reverse
                        ]
                        torch.save(
                            [interpret_score, h_attn, batch.y],
                            f"explanations/mamba/interpret_score_{batch.graph_id[0]}.pt",
                        )

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

            elif self.global_model_type == "Mamba_LearnRank":
                if self.opt_rank:
                    rank_score = torch.sigmoid(self.calculate_rank_score(h).squeeze())
                    h_ind_perm = lexsort([rank_score, batch.batch])

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    random_score = torch.rand(h.shape[0]).to(h.device)
                    h_ind_perm = lexsort([random_score, batch.batch])
                    h_shuffle = h[h_ind_perm]
                    rank_score = torch.sigmoid(
                        self.calculate_rank_score(h_shuffle).squeeze()
                    )

                    h_dense, mask = to_dense_batch(
                        h[h_ind_perm], batch.batch[h_ind_perm]
                    )
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)

                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.global_model_type == "Mamba_MoE":
                """
                input:h, batch -> random permutation -> h_dense, mask -> h_attn
                """
                if cfg.model.get('same_order_all_layers', False):
                    h_attn = self.same_order_moe(h, batch)
                else:
                    h_attn = self.diff_order_moe(h, batch)


            h_attn = self.dropout_attn(h_attn)
            if cfg.model.get("rtr_type", None) == "mamba":
                rtr_repr = h_attn

            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        h = sum(h_out_list)
        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)
        batch.x = h
        # if cfg.model.get("learn_rank", False) and self.training:
        #     return batch, rank_score
        if cfg.model.get("rtr_now", False):
            return batch, rtr_repr
        return batch

    def same_order_moe(self, h, batch):
        h_ind_perm = lexsort([batch.order_score, batch.batch])

        h_dense, mask = to_dense_batch(
            h[h_ind_perm], batch.batch[h_ind_perm]
        )
        h_ind_perm_reverse = torch.argsort(h_ind_perm)

        router_mask, router_probs, router_logits = self.router(h_dense)
        
        h_clone = h_dense.clone()

        router_mask = router_mask.bool()
        idx_mask = router_mask.sum(dim=0)
        idx_mask = torch.nonzero(idx_mask, as_tuple=True)[0].tolist()
        
        for i in idx_mask:
            h_clone[router_mask[:, i]] = self.experts_mamba[i](h_dense[router_mask[:, i]])
        
        h_attn = router_probs.unsqueeze(-1) * h_clone # B * [B, L, D] -> [B, L, D]
        h_attn = h_clone[mask][h_ind_perm_reverse]

        batch.router_logits = router_logits
        return h_attn

    def diff_order_moe(self, h, batch):
        router_tmp = self.gcn_for_router(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                    )
                ).x
        
        h_dense, mask = to_dense_batch(
            router_tmp, batch.batch
        )

        router_mask, router_probs, router_logits = self.router(h_dense)

        router_mask = router_mask.bool()
        idx_mask = router_mask.sum(dim=0)
        idx_mask = torch.nonzero(idx_mask, as_tuple=True)[0].tolist()

        h_clone = h_dense.clone()
        
        for i in idx_mask:
            # 1. learn order
            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            deg_noise = torch.rand_like(deg).to(deg.device)
            h_ind_perm = lexsort([deg + deg_noise, batch.batch])
            h_ind_perm_reverse = torch.argsort(h_ind_perm)

            h_dense, mask = to_dense_batch(
                h, batch.batch
            )

            h_clone[router_mask[:, i]] = self.experts_mamba[i](h_dense[router_mask[:, i]])
        
        h_attn = router_probs.unsqueeze(-1) * h_clone # B * [B, L, D] -> [B, L, D]
        h_attn = h_clone[mask][h_ind_perm_reverse]

        batch.router_logits = router_logits
        return h_attn
        

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
