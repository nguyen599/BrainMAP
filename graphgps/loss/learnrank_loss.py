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


def learnrank_cross_entropy(batches, ranks, scores, learn_rank, router_logits):
    losses = []
    for i in range(cfg.model.get("num_orders", 1) * 2):
        pred, true = batches[i]
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true

        if cfg.model.loss_fun == 'cross_entropy':
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
        elif cfg.model.loss_fun =='mse':
            # loss = torch.nn.MSELoss(reduction=cfg.model.size_average)(pred, true)
            loss = F.mse_loss(pred, true)

        losses.append(loss)

    # best_indices = torch.argmin(torch.stack(losses))
    if cfg.model.loss_fun == 'cross_entropy':
        best_indices = torch.topk(torch.stack(losses), cfg.model.get('num_orders', 1))[1]
    elif cfg.model.loss_fun =='mse':
        best_indices = torch.topk(torch.stack(losses), cfg.model.get('num_orders', 1), largest=False)[1]   
    
    for i in range(len(best_indices)):
        numerators = F.mse_loss(ranks[best_indices[i]].float(), scores[i])
    denominators = [
        F.mse_loss(ranks[index].float(), scores[0])
        for index in range(len(ranks))
    ]
    denominators = torch.stack(denominators).sum()
    rank_loss = numerators / denominators
    loss = torch.sum(losses[best_indices]) + learn_rank * rank_loss

    if cfg.model.get('load_balancing_loss', True):
        for i in range(len(best_indices)):
            for j in range(len(router_logits[0])):
                router_probs = torch.softmax(router_logits[best_indices[i]][j], dim=-1)
                loss += load_balancing_loss(router_probs) * 0.1

    return (
        loss,
        batches[best_indices[0]][1].cpu().squeeze(-1),
        batches[best_indices[0]][0].cpu().squeeze(-1),
    )
