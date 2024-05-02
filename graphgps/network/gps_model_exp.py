import torch
import torch_geometric
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder

from graphgps.layer.gps_layer import GPSLayer

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


@register_network("GPSModelEXP")
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

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        y,
        batch,
        ptr,
    ):
        # input: tensors in the databatch
        # output: a data batch
        # x = batch.x
        # edge_index = batch.edge_index
        # edge_attr = batch.edge_attr
        # y = batch.y
        # batch_ = batch.batch
        # ptr = batch.ptr
        # split = self.training

        # rebuild the batch
        batch = torch_geometric.data.Batch(
            x=x[0],
            edge_index=edge_index[0],
            edge_attr=edge_attr[0],
            y=y[0],
            batch=batch[0],
            ptr=ptr[0],
        )

        if cfg.model.learn_rank and self.training:
            batch = self.encoder(batch)
            batch_clone = batch.clone()
            rank_scores = []
            rank_scores_opt = []
            for i, layer in enumerate(self.layers):
                batch, rank_score_opt = layer(batch, True)
                batch_clone, rank_score = layer(batch_clone, False)
                rank_scores.append(rank_score)
                rank_scores_opt.append(rank_score_opt)
            rank_scores = torch.stack(rank_scores, dim=0)
            rank_scores_opt = torch.stack(rank_scores_opt, dim=0)
            return (
                self.post_mp(batch),
                self.post_mp(batch_clone),
                rank_scores_opt,
                rank_scores,
            )

        if cfg.prep.get("neighbors", False):
            batch = self.encoder(batch)
            # batch = self.pre_mp(batch)
            batch = self.layers(batch)
            return self.post_mp(batch)

        for module in self.children():
            batch = module(batch)
        return batch
