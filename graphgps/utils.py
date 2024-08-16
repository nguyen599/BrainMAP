import logging

import torch
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pdb
from torch_geometric.graphgym.config import cfg
import torch_geometric

from yacs.config import CfgNode


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short, device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce="mul")

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ["train", "val", "test"]
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(
                f"Key {'.'.join(key_list)} with "
                f"value {type(cfg_node)} is not "
                f"a valid type; valid types: {_VALID_TYPES}"
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith("OGB"):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith("PyG-"):
        dataset_name = dataset_name[4:]
    if dataset_name in ["GNNBenchmarkDataset", "TUDataset"]:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != "none":
        dataset_name += "-" if dataset_name != "" else ""
        if cfg.dataset.name == "LocalDegreeProfile":
            dataset_name += "LDP"
        else:
            dataset_name += cfg.dataset.name
    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ["gnn", "custom_gnn"]:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == "GPSModel":
        model_name = f"GPS.{cfg.gt.layer_type}"
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    # Compose wandb run name.
    name = f"{dataset_name}.{model_name}.r{cfg.run_id}"
    return name


class TopKRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopKRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.linear(mh_output)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class Top1Router(nn.Module):
    def __init__(self, n_embed, num_experts, if_jitter=False):
        super(Top1Router, self).__init__()
        self.num_experts = num_experts
        self.jitter_noise = if_jitter
        self.W_Q = nn.Linear(n_embed, n_embed)
        self.W_K = nn.Linear(n_embed, n_embed)
        self.W_V = nn.Linear(n_embed, n_embed)

        self.experts_linear = nn.Linear(n_embed, num_experts)

        self.pe = PositionalEncoding1D(n_embed)

    def compute_router_probabilities(
        self, h_dense: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r""" """
        
        # normalize layer_norm
        # h_dense = torch.nn.functional.layer_norm(h_dense, h_dense.shape[1:])

        if cfg.model.get(
            "moe_pos_enc",
            False,
        ):
            h_dense = self.pe(h_dense)  # B * N * D

        self.attention_token = torch.ones_like(h_dense[:, [0], :])
        Q = self.W_Q(self.attention_token)
        K = self.W_K(h_dense)
        V = self.W_V(h_dense)
        # scaled dot product attention
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        # router
        router_logits = attention_output.squeeze(1)

        # router_logits = torch.mean(h_dense, dim=1)

        if self.jitter_noise:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise

        router_logits = self.experts_linear(router_logits)
        
        router_probs = F.softmax(router_logits, dim=-1)

        return router_probs, router_logits

    def forward(
        self, h_dense: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        h_dense: B * N * D -> 1D positional encoding -> B * N * (D + H) -> attention -> B * D -> MLP -> B * 1
        """
        router_probs, router_logits = self.compute_router_probabilities(h_dense)
        expert_index = router_logits.argmax(dim=-1)
        expert_index = F.one_hot(expert_index, self.num_experts)

        router_probs = router_probs.max(dim=-1).values.unsqueeze(-1)

        return expert_index, router_probs, router_logits


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1000, pe_type="cat"):
        super(PositionalEncoding1D, self).__init__()
        pe = torch.zeros(max_len, d_model)
        self.pe_type = pe_type

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        self.cat_linear = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        if self.pe_type == "cat":
            out = torch.cat(
                (x, self.pe[:, : x.size(1), :].to(x.device).expand_as(x)), dim=-1
            )
            return self.cat_linear(out)
        return x + self.pe[:, : x.size(1), :].to(x.device)


class ExpModel(torch.nn.Module):
    def __init__(self, model):
        super(ExpModel, self).__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr1, y1, batch1, ptr1, batch_):
        batch_model = torch_geometric.data.Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr1,
            y=y1,
            batch=batch1,
            ptr=ptr1,
            split=batch_.split,
        )
        output = self.model(batch_model)
        # pdb.set_trace()
        return output[0]
