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
