import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, index_sort, to_dense_batch
from torch.nn import Linear, Parameter


# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr="add")  # "Add" aggregation (Step 5).
#         self.lin = Linear(in_channels, out_channels, bias=False)
#         self.bias = Parameter(torch.empty(out_channels))

#     #     self.reset_parameters()

#     # def reset_parameters(self):
#     #     self.lin.reset_parameters()
#     #     self.bias.data.zero_()

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # calculate similarity between nodes
#         _, indices = index_sort(x, edge_index[0])
#         result, _ = to_dense_batch(x, edge_index[0], fill_value=0)
#         return out

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]

#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j


src_feat = torch.tensor(
    [
        [1, 1, 1.0],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
    ]
)

src_feat_norm = src_feat.norm(dim=1, keepdim=True)
edge_index = torch.tensor(
    [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
)
add_self_loops(edge_index, num_nodes=src_feat.size(0))

src_feat_ = src_feat[edge_index[0]]

_, indices = index_sort(edge_index[1])

result, _ = to_dense_batch(src_feat_[indices], edge_index[1][indices])
result = torch.cat(
    [
        result,
        torch.zeros(
            src_feat.shape[0] - result.shape[0], result.shape[1], result.shape[2]
        ),
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
