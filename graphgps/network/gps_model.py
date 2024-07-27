import torch
import torch_geometric
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder
from torch_geometric.data import Batch
from graphgps.layer.gatedgcn_layer import GatedGCNLayer

from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gps_layer import GPSLayer
import torch.nn as nn

import os

import pdb


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_inner,
                        -1,
                        -1,
                        has_act=False,
                        has_bias=False,
                        cfg=cfg,
                    )
                )
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if "PNA" in cfg.gt.layer_type else cfg.gnn.dim_inner
            if cfg.dataset.edge_encoder_name == "ER":
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith("+ER"):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]
                ]
                self.edge_encoder = EdgeEncoder(
                    cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe
                )
                self.edge_encoder_er = EREdgeEncoder(
                    cfg.posenc_ERE.dim_pe, use_edge_attr=True
                )
            else:
                EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        cfg.gnn.dim_edge, -1, -1, has_act=False, has_bias=False, cfg=cfg
                    )
                )

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)

        return batch


@register_network("GPSModel")
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        assert (
            cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in
        ), "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split("+")
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                GPSLayer(
                    dim_h=cfg.gt.dim_hidden,
                    local_gnn_type=local_gnn_type,
                    global_model_type=global_model_type,
                    num_heads=cfg.gt.n_heads,
                    pna_degrees=cfg.gt.pna_degrees,
                    equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    bigbird_cfg=cfg.gt.bigbird,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        if cfg.model.get("learn_rank", False):
            self.rank_learner_gcn = nn.ParameterList(
                    [
                        GatedGCNLayer(
                            cfg.gt.dim_hidden,
                            cfg.gt.dim_hidden,
                            dropout=cfg.gt.dropout,
                            residual=False,
                        )
                        for i in range(cfg.model.get('num_orders', 1))
                    ]
                )

            self.rank_learner = torch.nn.Linear(cfg.gt.dim_hidden, 1)

    def forward(self, batch):
        """
        1. we have several models GCNs as rank learners
        2. each GCN has a positive ranking sampled, how to store it? since we have four rankings, we can store them 
        in a list. n_orders * (positive + negative)
        3. every time the sample is the same for different order learners. if instantiated differently, will the order be the same?
        3. we need two para: num_orders, num_samples
        4. it doesn't have to be the best order, the probability of sampling is corresponding to the performance. 
        but we don't have a standard. we could use our experience to set the standard. : a curated loss
        """

        if cfg.model.get("rtr_now", False):
            batch = self.encoder(batch)
            rtr_reprs = []
            for i, layer in enumerate(self.layers):
                batch, rtr_repr = layer(batch)
                rtr_reprs.append(rtr_repr)
            rtr_repr = torch.stack(rtr_reprs, dim=0)
            if os.path.exists("rtr_repr") is False:
                os.mkdir("rtr_repr")
            torch.save(
                [rtr_repr, batch.graph_id, batch.ptr],
                f"rtr_repr/rtr_rpre_{batch.graph_id[0]}.pt",
            )
            return self.post_mp(batch)

        if cfg.model.get("same_order_all_layers", False) and cfg.model.get(
            "learn_rank", False
        ):
            batch = self.encoder(batch)

            scores = []
            for i in range(cfg.model.get('num_orders', 1)):
                # GNN for order learner
                score_tmp = self.rank_learner_gcn[i](
                    Batch(
                        batch=batch,
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                    )
                ).x
                score_tmp, mask = to_dense_batch(score_tmp, batch.batch)
                score = self.rank_learner(score_tmp).squeeze()
                s_min = torch.min(score, dim=-1).values
                s_max = torch.max(score, dim=-1).values

                S = score.unsqueeze(2) - score.unsqueeze(1)
                S = torch.sigmoid(S)
                S_min = torch.sigmoid(
                    s_min.unsqueeze(-1).unsqueeze(-1) - score.unsqueeze(1)
                )
                S_max = torch.sigmoid(
                    s_max.unsqueeze(-1).unsqueeze(-1) - score.unsqueeze(1)
                )

                numerator = S.sum(dim=2) - S_min.sum(dim=2)
                denominator = S_max.sum(dim=2) - S_min.sum(dim=2)
                score = (numerator / denominator) * (score.size(1) - 1) + 1

                scores.append(score)


            # order_score = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            if self.training:
                ranks = []
                batches = []
                router_logits = []
                for i in range(cfg.model.get("num_orders", 1)*cfg.model.get("num_samples", 2)):
                    batch_clone = batch.clone()
                    deg = degree(batch_clone.edge_index[0], batch_clone.x.shape[0]).to(
                        torch.float
                    )
                    order_score = torch.rand_like(deg).to(deg.device)
                    batch_clone.order_score = order_score + deg

                    order_score, mask = to_dense_batch(order_score, batch.batch)
                    rank = torch.argsort(torch.argsort(order_score, descending=True))
                    ranks.append(rank)

                    # edge_index = [batch.edge_index]
                    # x = [batch.x]
                    # edge_attr = [batch.edge_attr]
                    # y = [batch.y]
                    # ptr = [batch.ptr]
                    # batch_batch = [batch.batch]
                    # if cfg.model.get("duplicate_batch", False):
                    #     duplicate = cfg.model.get("num_experts", 2) - 1
                    # else:
                    #     duplicate = 1
                    # for i in range(duplicate):
                    #     edge_index.append(batch.edge_index + batch.x.shape[0] * (i + 1))
                    #     x.append(batch.x)
                    #     edge_attr.append(batch.edge_attr)
                    #     y.append(batch.y)
                    #     ptr.append(ptr[-1][1:] + ptr[0][-1])
                    #     batch_batch.append(batch_batch[-1] + batch_batch[0][-1] + 1)
                    # batch.edge_index = torch.cat(edge_index, dim=1)
                    # batch.x = torch.cat(x, dim=0)
                    # if batch.edge_attr is not None:
                    #     batch.edge_attr = torch.cat(edge_attr, dim=0)
                    # batch.y = torch.cat(y, dim=0)
                    # batch.ptr = torch.cat(ptr, dim=0)
                    # batch.batch = torch.cat(batch_batch, dim=0)
                    for i, layer in enumerate(self.layers):
                        batch_clone = layer(batch_clone)
                    router_logit = batch_clone.router_logits
                    batch_clone = self.post_mp(batch_clone)
                    batches.append(batch_clone)
                    router_logits.append(router_logit)

                return batches, ranks, scores, router_logits

            else:
                preds = []
                for i in range(cfg.model.get("num_orders", 1)):
                    batch_clone = batch.clone()
                    batch_clone.order_score = scores[i][mask]
                    for i, layer in enumerate(self.layers):
                        batch_clone = layer(batch_clone)
                    batch_clone = self.post_mp(batch_clone)
                    preds.append(batch_clone[0])
                # mean
                return torch.stack(preds, dim=0).mean(dim=0), batch_clone[1]

        for module in self.children():
            batch = module(batch)
            # print(batch)
            # pdb.set_trace()
        return batch
