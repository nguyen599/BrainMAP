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
import pdb


def compute_centrality(data, centrality_types, is_undirected=True, cfg=None):
    """Compute centrality scores for nodes in the graph.

    Args:
        data (torch_geometric.data.Data): Graph data object
        centrality_types (list): List of centrality types to compute
        is_undirected (bool): Whether the graph is undirected

    Returns: Dictionary of computed centrality scores

    """
    if is_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.T.numpy())
    centrality_scores = {}
    for centrality_type in centrality_types:
        if centrality_type == "degree":
            centrality_scores["degree"] = dict(G.degree())
        elif centrality_type == "closeness":
            centrality_scores["closeness"] = nx.closeness_centrality(G)
        elif centrality_type == "betweenness":
            centrality_scores["betweenness"] = nx.betweenness_centrality(G)
        elif centrality_type == "eigenvector":
            centrality_scores["eigenvector"] = nx.eigenvector_centrality_numpy(G)  #
        elif centrality_type == "pagerank":
            centrality_scores["pagerank"] = nx.pagerank(G)  #
        elif centrality_type == "katz":
            centrality_scores["katz"] = nx.katz_centrality(G)  #
        elif centrality_type == "harmonic":
            centrality_scores["harmonic"] = nx.harmonic_centrality(G)
        elif centrality_type == "subgraphcentrality":
            centrality_scores["subgraphcentrality"] = nx.subgraph_centrality(G)  #
        elif centrality_type == "load":
            centrality_scores["load"] = nx.load_centrality(G)
        elif centrality_type == "clustering":
            centrality_scores["clustering"] = nx.clustering(G)  #
        elif centrality_type == "eccentricity":
            centrality_scores["eccentricity"] = nx.eccentricity(G)
        elif centrality_type == "communicability":
            centrality_scores["communicability"] = nx.communicability_centrality(G)
        elif centrality_type == "current_flow":
            centrality_scores["current_flow"] = nx.current_flow_closeness_centrality(G)
        elif centrality_type == "laplacian":
            centrality_scores["laplacian"] = nx.laplacian_centrality(G)  #
        elif centrality_type == "trophic":
            centrality_scores["trophic"] = nx.trophic_levels(G)
        elif centrality_type == "percolation":
            centrality_scores["percolation"] = nx.percolation_centrality(G)
        elif centrality_type == "second_order":
            centrality_scores["second_order"] = nx.second_order_centrality(G)
        else:
            raise ValueError(f"Centrality type '{centrality_type}' not supported")

        centrality_scores[centrality_type] = torch.tensor(
            list(centrality_scores[centrality_type].values())
        ).float()

        if cfg.prep.get("centrality_buckets", 0):
            bucket = torch.quantile(
                centrality_scores[centrality_type],
                torch.tensor(
                    [
                        i / cfg.prep.centrality_buckets
                        for i in range(1, cfg.prep.centrality_buckets + 1)
                    ]
                ),
            )
            centrality_scores[centrality_type] = torch.bucketize(
                centrality_scores[centrality_type],
                bucket,
            )

        data[centrality_type] = centrality_scores[centrality_type]

    return data
