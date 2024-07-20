import torch.nn as nn
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch.nn.functional as F


import pdb


def load_balancing_loss(router_probs):
    """
    input: router_probs: B * N * E -> expert_index -> need to be similar (entropy)
    output: loss
    """
    num_experts = router_probs.shape[-1]
    expert_indices = torch.argmax(router_probs, dim=-1)

    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.long()

    expert_mask = F.one_hot(expert_indices, num_classes=num_experts)

    expert_mask = expert_mask.float()
    graphs_per_expert = torch.mean(expert_mask, dim=0)
    router_prob_per_expert = torch.mean(router_probs, dim=0)

    return torch.mean(graphs_per_expert * router_prob_per_expert) * (num_experts**2)


def compute_loss_moe(pred, true, batch):
    """
    Input: pred: , true:, batch for router_probs and expert_index
    Output: loss, true, pred

    add:load balancing loss:
    """
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if cfg.model.loss_fun == 'cross_entropy':
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(pred, true)
        # binary or multilabel
        else:
            true = true.float()
            loss = torch.nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)(pred, true)
            true = true.long()
    elif cfg.model.loss_fun =='mse':
        loss = torch.nn.MSELoss(reduction=cfg.model.size_average)(pred, true)

    for i in range(len(batch.router_logits)):
        router_probs = torch.softmax(batch.router_logits[i], dim=-1)
        loss += load_balancing_loss(router_probs) * 0.01

    return loss, true, pred
