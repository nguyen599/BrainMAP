import torch.nn as nn
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch.nn.functional as F
import pdb


# def get_rank_target(batch):
#     """Get rank target."""
#     for i in range(len(batch.ptr) - 1):
#         if i == 0:
#             rank_target = torch.arange(
#                 batch.ptr[i + 1] - batch.ptr[i], dtype=torch.float32
#             ).to(batch.x.device) / (batch.ptr[i + 1] - batch.ptr[i])
#         else:
#             rank_target = torch.cat(
#                 (
#                     rank_target,
#                     torch.arange(
#                         batch.ptr[i + 1] - batch.ptr[i], dtype=torch.float32
#                     ).to(batch.x.device)
#                     / (batch.ptr[i + 1] - batch.ptr[i]),
#                 )
#             )
#     return rank_target


# @register_loss("learnrank_cross_entropy")
def learnrank_cross_entropy(batches, ranks, scores, learn_rank):
    losses = []
    for i in range(cfg.model.get("num_orders", 2)):
        pred, true = batches[i]
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true

        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(pred, true)
        # binary or multilabel
        else:
            true = true.float()
            loss = torch.nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)(
                pred, true
            )
            true = true.long()
        losses.append(loss)

    best_index = torch.argmin(torch.stack(losses))
    numerators = F.mse_loss(ranks[best_index].float(), scores)
    denominators = [
        F.mse_loss(ranks[index].float(), torch.sort(scores)[0])
        for index in range(len(ranks))
        if index != best_index
    ]
    denominators = torch.stack(denominators).sum()
    rank_loss = numerators / denominators
    loss = losses[best_index] + learn_rank * rank_loss

    return (
        loss,
        batches[best_index][1].cpu().squeeze(-1),
        batches[best_index][0].cpu().squeeze(-1),
    )
