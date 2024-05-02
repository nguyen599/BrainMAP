import torch.nn as nn
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

import pdb


def get_rank_target(batch):
    """Get rank target."""
    for i in range(len(batch.ptr) - 1):
        if i == 0:
            rank_target = torch.arange(
                batch.ptr[i + 1] - batch.ptr[i], dtype=torch.float32
            ).to(batch.x.device) / (batch.ptr[i + 1] - batch.ptr[i])
        else:
            rank_target = torch.cat(
                (
                    rank_target,
                    torch.arange(
                        batch.ptr[i + 1] - batch.ptr[i], dtype=torch.float32
                    ).to(batch.x.device)
                    / (batch.ptr[i + 1] - batch.ptr[i]),
                )
            )
    return rank_target


# @register_loss("learnrank_cross_entropy")
def learnrank_cross_entropy(
    batch_1, batch_2, batch, rank_scores_opt, rank_scores, learn_rank=False
):
    """LearnRank cross-entropy loss."""
    if cfg.dataset.task_type == "classification_multilabel":
        bce_loss_func = nn.BCEWithLogitsLoss()
        loss_1 = bce_loss_func(batch_1[0], batch_1[1])
        loss_2 = bce_loss_func(batch_2[0], batch_2[1])
        if learn_rank:
            if loss_1 < loss_2 + 0.01:
                rank_target = get_rank_target(batch).to(rank_scores[0].device)
                loss_rank = 0
                for i, rank_score_opt in enumerate(rank_scores):
                    loss_rank += nn.CrossEntropyLoss()(rank_score_opt, rank_target)
                loss_1 += 0.01 * loss_rank / len(rank_scores[0])
                return loss_1, batch_1[1], batch_1[0]
            else:
                rank_target = get_rank_target(batch).to(rank_scores_opt[0].device)
                loss_rank = 0
                for i, rank_score in enumerate(rank_scores_opt):
                    loss_rank += nn.CrossEntropyLoss()(rank_score, rank_target)
                loss_2 += 0.01 * loss_rank / len(rank_scores_opt[0])
                print("switch")
                return loss_2, batch_2[1], batch_2[0]
        else:
            return loss_2, batch_2[1], batch_2[0]
