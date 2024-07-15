from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch
import pdb


# @register_loss("contrastive_entropy")
def contrastive_entropy(pred, true):
    """Contrastive entropy loss.
    1. pred: 128 * 2,
    2. positive: 1
    """
    if cfg.dataset.task_type == "contrastive":
        if cfg.model.loss_fun != "contrastive_entropy":
            raise ValueError(
                "Only 'contrastive_entropy' loss_fun supported with "
                "'contrastive' task_type."
            )
    # cosine similarity of the first 128 samples and the last 128 samples, 128 * 128
    cosine_similarity = torch.mm(pred, pred.t())
    cosine_similarity = cosine_similarity[: pred.shape[0] // 2, pred.shape[0] // 2 :]
    # positive in the diagonal
    positive_sample = torch.eye(pred.shape[0] // 2).to(cfg.device)
    # negative in the off-diagonal
    negative_sample = 1 - positive_sample
    # info_nce_loss

    positive_values = torch.mean(positive_sample * cosine_similarity)
    negative_values = torch.mean(negative_sample * cosine_similarity)
    loss = -torch.log(
        torch.exp(positive_values)
        / (torch.exp(positive_values) + torch.exp(negative_values))
    )
    return loss


if __name__ == "__main__":
    # train
    data_raw = torch.randn(256, 2)
