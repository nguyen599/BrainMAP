import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss("multilabel_cross_entropy")
def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss."""
    if cfg.dataset.task_type == "classification_multilabel":
        if cfg.model.loss_fun != "cross_entropy":
            raise ValueError(
                "Only 'cross_entropy' loss_fun supported with "
                "'classification_multilabel' task_type."
            )
        bce_loss = nn.BCEWithLogitsLoss()
        is_labeled = true == true  # Filter our nans.
        return bce_loss(pred[is_labeled], true[is_labeled].float()), pred


# @register_loss("learnrank_cross_entropy")
# def learnrank_cross_entropy(pred, true):
#     """LearnRank cross-entropy loss."""
#     if cfg.dataset.task_type == "classification_multilabel":
#         if cfg.model.loss_fun != "learnrank_cross_entropy":
#             raise ValueError(
#                 "Only 'cross_entropy' loss_fun supported with "
#                 "'classification_multilabel' task_type."
#             )
#         bce_loss_func = nn.BCEWithLogitsLoss()
#         is_labeled = true == true
#         bce_loss = bce_loss_func(pred[is_labeled], true[is_labeled].float())
