import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    to_dense_adj,
    to_networkx,
)
from scipy import sparse
import pdb


def compute_neighbors(data, neighbor_hops, is_undirected=True, cfg=None):
    """Compute neighbors for nodes in the graph.
        data (torch_geometric.data.Data): Graph data object
        neighbor_hops (int): Number of hops to consider
        is_undirected (bool): Whether the graph is undirected
    Returns: Dictionary of computed neighbors
    """
    # idea is to use sparse matrix multiplication to get different hop adjacency matrices
    # for each node in the graph
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocoo()

    # normalize adj
    # adj = adj + np.eye(adj.shape[0])
    row_sum = adj.sum(axis=1)
    row_sum[row_sum == 0] = 1
    adj = adj / row_sum
    adj = sparse.csr_matrix(adj)
    for i in range(1, neighbor_hops + 1):
        adj = adj.dot(adj)
        # calculate embeddings for each node by matrix multiplication
        # with the adjacency matrix
        if "Atom" in cfg.dataset.node_encoder_name:
            # adj to edge_index
            edge_index = adj.nonzero()
            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            data[f"edge_index_{i}"] = edge_index
            data[f"edge_attr_{i}"] = torch.tensor(adj.data, dtype=torch.float32)
        else:
            x = adj.dot(data.x)
            data[f"neighbors_{i}"] = torch.tensor(x)
    return data
