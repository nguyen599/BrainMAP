edge_index = {}
edge_index_1 = batch.edge_index
sparse_adj = SparseTensor(
    col=batch.edge_index[0],
    row=batch.edge_index[1],
    sparse_sizes=(batch.x.shape[0], batch.x.shape[0]),
)
sparse_adj_2 = sparse_adj @ sparse_adj
edge_index_2 = torch.stack([sparse_adj_2.coo()[0], sparse_adj_2.coo()[1]])

edge_index[0] = edge_index_1[0] * batch.x.shape[0] + edge_index_1[1]
edge_index[1] = edge_index_2[0] * batch.x.shape[0] + edge_index_2[1]
test_is_in = torch.isin(edge_index[1], edge_index[0])

edge_index_2 = edge_index_2.T[~test_is_in].T


def if_in_1(src_idx, dst_idx):
    for hop in range(0, 2):
        if torch.isin(src_idx * batch.x.shape[0] + dst_idx, edge_index[hop]):
            return hop
    return hop


t1 = torch.arange(10000).expand(10000, 10000)
t2 = torch.arange(10000).expand(10000, 10000).T
re = np.vectorize(if_in_1)(t1, t2)
re = torch.tensor(re)
res = torch.zeros_like(re)
res = res.scatter_(1, re, t1)
