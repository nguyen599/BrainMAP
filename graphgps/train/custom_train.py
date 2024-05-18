import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.loss.learnrank_loss import learnrank_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

from deepspeed.profiling.flops_profiler import FlopsProfiler

# from torch.autograd import profiler
from torch.profiler import profile, record_function, ProfilerActivity
from captum.attr import (
    IntegratedGradients,
    DeepLiftShap,
    DeepLift,
    ShapleyValueSampling,
    ShapleyValues,
    Lime,
)
from torch_geometric.explain import Explainer, GNNExplainer


from torch_geometric.nn import to_captum_model, to_captum_input
import torch_geometric

import pdb


def subsample_batch_index(batch, min_k=1, ratio=0.1):
    torch.manual_seed(0)
    unique_batches = torch.unique(batch.batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch.batch == batch_index).nonzero().squeeze()
        # See how many nodes in the graphs
        # And how many left after subsetting
        k = int(indices_in_batch.size(0) * ratio)
        # If subsetting gives more than 1, do subsetting
        if k > min_k:
            perm = torch.randperm(indices_in_batch.size(0))
            idx = perm[:k]
            idx = indices_in_batch[idx]
            idx, _ = torch.sort(idx)
        # Otherwise retain entire graph
        else:
            idx = indices_in_batch
        permuted_indices.append(idx)
    idx = torch.cat(permuted_indices)
    return idx


def arxiv_cross_entropy(pred, true, split_idx):
    true = true.squeeze(-1)
    if pred.ndim > 1 and true.ndim == 1:
        pred_score = F.log_softmax(pred[split_idx], dim=-1)
        loss = F.nll_loss(pred_score, true[split_idx])
    else:
        raise ValueError("In ogbn cross_entropy calculation dimensions did not match.")
    return loss, pred_score


# @profiler.profile
# def profile_mem_forward(model, batch):
#     pred, true = model(batch)
#     return pred, true


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, epoch):
    # flop related
    if_mem = False
    if_flop = False
    if_select = False
    if if_flop:
        prof = FlopsProfiler(model, None)
        # profile_step = 0
        total_flop_s = 0.0
        sample_count = 0
        if if_select:
            total_node = 0

    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        if if_select:
            ratio = 1.0
            idx = subsample_batch_index(batch, min_k=1, ratio=ratio)
            batch = batch.subgraph(idx)
        # flop related
        if if_flop:  # and iter == profile_step:
            prof.start_profile()
        batch.split = "train"
        batch.to(torch.device(cfg.device))
        if epoch == 50 and cfg.model.get("gnn_explainer", False):
            batch_ = batch.clone().to(torch.device(cfg.device))

            class ExpModel(torch.nn.Module):
                def __init__(self, model):
                    super(ExpModel, self).__init__()
                    self.model = model

                def forward(self, x, edge_index, edge_attr1, y1, batch1, ptr1):
                    batch_model = torch_geometric.data.Batch(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr1,
                        y=y1,
                        batch=batch1,
                        ptr=ptr1,
                        split=batch_.split,
                    )
                    output = self.model(batch_model)
                    # pdb.set_trace()
                    return output[0]

            custom_forward = ExpModel(model)

            explainer = Explainer(
                model=custom_forward,
                algorithm=GNNExplainer(epochs=200),
                explanation_type="model",
                node_mask_type="object",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="graph",
                    return_type="probs",
                ),
            )
            node_index = 0
            for i in range(5):
                try:
                    if batch_[i].y == 3:
                        explanation = explainer(
                            batch_[i].x,
                            batch_[i].edge_index,
                            # index=node_index,
                            edge_attr1=batch_[i].edge_attr,
                            y1=batch_[i].y,
                            batch1=batch_.batch[:360],
                            ptr1=batch_.ptr[:2],
                        )
                        print("Explanation:", explanation)
                        if (
                            cfg.model.type == "GPSModel"
                            and cfg.gt.layer_type
                            == "CustomGatedGCN+Mamba_Hybrid_Degree_Noise"
                        ):
                            file = f"explanations/explanation_{batch_.graph_id[i]}.pt"
                        else:
                            file = (
                                f"explanations/explanationgcn_{batch_.graph_id[i]}.pt"
                            )
                        torch.save(
                            [
                                explanation,
                                batch_.graph_id[i],
                                batch_[i].y,
                            ],
                            file,
                        )
                except:
                    continue
        else:
            if cfg.model.learn_rank:
                (batch_1, batch_2, rank_scores_opt, rank_scores) = model(batch)

                if epoch < cfg.model.learn_rank_start_epoch:
                    learn_rank = False
                else:
                    learn_rank = True
                loss, true, pred_score = learnrank_cross_entropy(
                    batch_1, batch_2, batch, rank_scores_opt, rank_scores, learn_rank
                )
                _true = true
                _pred = pred_score
            else:
                test = batch.x.clone()
                pred, true = model(batch)
                if cfg.dataset.name == "ogbg-code2":
                    loss, pred_score = subtoken_cross_entropy(pred, true)
                    _true = true
                    _pred = pred_score
                elif cfg.dataset.name == "ogbn-arxiv":
                    split_idx = loader.dataset.split_idx["train"].to(
                        torch.device(cfg.device)
                    )
                    loss, pred_score = arxiv_cross_entropy(pred, true, split_idx)
                    _true = true[split_idx].detach().to("cpu", non_blocking=True)
                    _pred = pred_score.detach().to("cpu", non_blocking=True)
                else:
                    loss, pred_score = compute_loss(pred, true)
                    _true = true.detach().to("cpu", non_blocking=True)
                    _pred = pred_score.detach().to("cpu", non_blocking=True)

            if if_flop:
                prof.stop_profile()
                flops = prof.get_total_flops()
                flops_s = flops / 1000000000.0
                total_flop_s += flops_s
                sample_count += len(torch.unique(batch.batch))
                params = prof.get_total_params()
                prof.end_profile()
                if if_select:
                    total_node += batch.x.size(0)

            loss = loss.clamp(-1e6, 1e6)
            loss.backward()

            # Parameters update after accumulating gradients for given num. batches.
            if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
                if cfg.optim.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            logger.update_stats(
                true=_true,
                pred=_pred,
                loss=loss.detach().cpu().item(),
                lr=scheduler.get_last_lr()[0],
                time_used=time.time() - time_start,
                params=cfg.params,
                dataset_name=cfg.dataset.name,
            )
            time_start = time.time()

    if if_flop:
        print("################ Print flop")
        print(total_flop_s / sample_count, params)
        print("################ End print flop")
    if if_mem:
        print("################ Print mem")
        print(torch.cuda.max_memory_allocated() / (1024**2))
        print(torch.cuda.max_memory_reserved() / (1024**2))
        # print(prof_mem.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        print("################ End print mem")
    if if_select:
        print("################ Print avg nodes")
        print(total_node / sample_count)


@torch.no_grad()
def eval_epoch(logger, loader, model, split="val", epoch=0):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        if cfg.model.get("captum", False):
            batch_ = batch.clone()
            pred, true = model(batch)
            extra_stats = {}

            class CustomModel(torch.nn.Module):
                def __init__(self, model):
                    super(CustomModel, self).__init__()
                    self.model = model

                def forward(self, x, edge_index, edge_attr, y, batch, ptr):
                    batch_model = torch_geometric.data.Batch(
                        x=x[0].reshape(-1, batch_.x.size(-1)),
                        edge_index=edge_index[0],
                        edge_attr=edge_attr[0],
                        y=y[0],
                        batch=batch[0],
                        ptr=ptr[0],
                        split=batch_.split,
                    )
                    output = self.model(batch_model)
                    # pdb.set_trace()
                    return output[0]

            custom_forward = CustomModel(model)

            if epoch == 100 and cfg.model.get("captum", False):
                ig = Lime(
                    custom_forward,
                )
                for i in range(5):
                    if batch_[i].y == 3:
                        attributions = ig.attribute(
                            batch_[i].x.flatten().unsqueeze(0),
                            # baselines=batch_.x.unsqueeze(0) * 0,
                            additional_forward_args=(
                                batch_[i].edge_index.unsqueeze(0),
                                batch_[i].edge_attr.unsqueeze(0),
                                batch_[i].y.unsqueeze(0),
                                batch_.batch[:360].unsqueeze(0),
                                batch_.ptr[:2].unsqueeze(0),
                            ),
                            target=0,
                            # return_convergence_delta=True,
                            # n_samples=200,
                        )
                        print("IG Attributions:", attributions)
                        # print("Convergence Delta:", delta)
                        if not os.path.exists("attributions"):
                            os.makedirs("attributions")
                        torch.save(
                            [
                                attributions,
                                batch_.graph_id[i],
                                batch_[i].y,
                            ],
                            f"attributions/attributions_{batch_.graph_id[i]}.pt",
                        )

        elif epoch == 1000 and cfg.model.get("gnn_explainer", False):
            batch_ = batch.clone()
            pred, true = model(batch)
            extra_stats = {}

            class ExpModel(torch.nn.Module):
                def __init__(self, model):
                    super(ExpModel, self).__init__()
                    self.model = model

                def forward(self, x, edge_index, edge_attr1, y1, batch1, ptr1):
                    x.requires_grad = True
                    batch_model = torch_geometric.data.Batch(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr1,
                        y=y1,
                        batch=batch1,
                        ptr=ptr1,
                        split=batch_.split,
                    )
                    output = self.model(batch_model)
                    # pdb.set_trace()
                    return output[0]

            custom_forward = ExpModel(model)

            explainer = Explainer(
                model=custom_forward,
                algorithm=GNNExplainer(epochs=200),
                explanation_type="model",
                node_mask_type="object",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="graph",
                    return_type="probs",
                ),
            )
            node_index = 0
            i = 0
            explanation = explainer(
                batch_[i].x,
                batch_[i].edge_index,
                # index=node_index,
                edge_attr1=batch_[i].edge_attr,
                y1=batch_[i].y,
                batch1=batch_.batch[:360],
                ptr1=batch_.ptr[:2],
            )
            print("Explanation:", explanation)
        elif cfg.model.get("rtr_type", None) is not None:
            cfg.model["rtr_now"] = False
            if epoch == cfg.model.get("rtr_epoch", 100):
                cfg.model["rtr_now"] = True
            pred, true = model(batch)
            extra_stats = {}

        elif cfg.gnn.head == "inductive_edge":
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == "ogbg-code2":
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == "ogbn-arxiv":
            index_split = loader.dataset.split_idx[split].to(torch.device(cfg.device))
            loss, pred_score = arxiv_cross_entropy(pred, true, index_split)
            _true = true[index_split].detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **extra_stats,
        )
        time_start = time.time()


@register_train("custom")
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info("Checkpoint found, Task already done")
    else:
        logging.info("Start from epoch %s", start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError("WandB is not installed.")
        if cfg.wandb.name == "":
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=wandb_name,
            settings=wandb.Settings(code_dir="."),
        )
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ["val", "test"]
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    with torch.autograd.detect_anomaly():
        attributions_all = []
        graph_id_all = []
        for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
            start_time = time.perf_counter()
            train_epoch(
                loggers[0],
                loaders[0],
                model,
                optimizer,
                scheduler,
                cfg.optim.batch_accumulation,
                cur_epoch,
            )
            perf[0].append(loggers[0].write_epoch(cur_epoch))
            if is_eval_epoch(cur_epoch):
                # Perform evaluation
                for i in range(1, num_splits):
                    eval_epoch(
                        loggers[i],
                        loaders[i],
                        model,
                        split=split_names[i - 1],
                        epoch=cur_epoch,
                    )
                    perf[i].append(loggers[i].write_epoch(cur_epoch))
            else:
                for i in range(1, num_splits):
                    perf[i].append(perf[i][-1])

            val_perf = perf[1]
            if cfg.optim.scheduler == "reduce_on_plateau":
                scheduler.step(val_perf[-1]["loss"])
            else:
                scheduler.step()
            full_epoch_times.append(time.perf_counter() - start_time)
            # Checkpoint with regular frequency (if enabled).
            if (
                cfg.train.enable_ckpt
                and not cfg.train.ckpt_best
                and is_ckpt_epoch(cur_epoch)
            ):
                save_ckpt(model, optimizer, scheduler, cur_epoch)

            if cfg.wandb.use:
                run.log(flatten_dict(perf), step=cur_epoch)

            # Log current best stats on eval epoch.
            if is_eval_epoch(cur_epoch):
                best_epoch = np.array([vp["loss"] for vp in val_perf]).argmin()
                best_train = best_val = best_test = ""
                if cfg.metric_best != "auto":
                    # Select again based on val perf of `cfg.metric_best`.
                    m = cfg.metric_best
                    best_epoch = getattr(
                        np.array([vp[m] for vp in val_perf]), cfg.metric_agg
                    )()
                    if m in perf[0][best_epoch]:
                        best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                    else:
                        # Note: For some datasets it is too expensive to compute
                        # the main metric on the training set.
                        best_train = f"train_{m}: {0:.4f}"
                    best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                    best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                    if cfg.wandb.use:
                        bstats = {"best/epoch": best_epoch}
                        for i, s in enumerate(["train", "val", "test"]):
                            bstats[f"best/{s}_loss"] = perf[i][best_epoch]["loss"]
                            if m in perf[i][best_epoch]:
                                bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                                run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
                            for x in ["hits@1", "hits@3", "hits@10", "mrr"]:
                                if x in perf[i][best_epoch]:
                                    bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                        run.log(bstats, step=cur_epoch)
                        run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                        run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
                # Checkpoint the best epoch params (if enabled).
                if (
                    cfg.train.enable_ckpt
                    and cfg.train.ckpt_best
                    and best_epoch == cur_epoch
                ):
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                    if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                        clean_ckpt()
                logging.info(
                    f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                    f"(avg {np.mean(full_epoch_times):.1f}s) | "
                    f"Best so far: epoch {best_epoch}\t"
                    f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                    f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                    f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
                )
                if hasattr(model, "trf_layers"):
                    # Log SAN's gamma parameter values if they are trainable.
                    for li, gtl in enumerate(model.trf_layers):
                        if (
                            torch.is_tensor(gtl.attention.gamma)
                            and gtl.attention.gamma.requires_grad
                        ):
                            logging.info(
                                f"    {gtl.__class__.__name__} {li}: "
                                f"gamma={gtl.attention.gamma.item()}"
                            )

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info("Task done, results saved in %s", cfg.run_dir)


@register_train("inference-only")
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ["train", "val", "test"]
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model, split=split_names[i], epoch=cur_epoch)
        perf[i].append(loggers[i].write_epoch(cur_epoch))
    val_perf = perf[1]

    best_epoch = np.array([vp["loss"] for vp in val_perf]).argmin()
    best_train = best_val = best_test = ""
    if cfg.metric_best != "auto":
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        best_epoch = getattr(np.array([vp[m] for vp in val_perf]), cfg.metric_agg)()
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f"Done! took: {time.perf_counter() - start_time:.2f}s")
    for logger in loggers:
        logger.close()
