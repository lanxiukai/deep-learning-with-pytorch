import math
import os
from collections.abc import Callable

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..data.vision import TensorDataLoader, vision_loaders
from ..devices.randomness import set_seed
from ..filesystem.directories import reset_dir
from ..plot.images import save_grid
from ..training.metrics import NumericScalar, save_metrics_csv
from ..training.timing import Timer
from ._ebm_types import DBMPretrainMetrics, EpochMetrics, MetricHistory, StepMetrics, _DBMConfig, layer_cfg
from .dbm_model import DBM
from .rbm_model import BinaryRBM
from .rbm_primitives import binarize, rbm_energy_free_energy_fast
from .training_artifacts import save_rbm_training_artifacts


@torch.no_grad()
def _initialize_visible_bias(rbm: BinaryRBM, train_iter: TensorDataLoader, n_visible: int, epsilon: float, device: torch.device) -> None:
    total = torch.zeros(n_visible, device=device)
    count = 0
    for index, (batch, _) in enumerate(train_iter):
        batch = batch.to(device, non_blocking=True).reshape(batch.size(0), -1)
        total += batch.sum(dim=0)
        count += batch.size(0)
        if index + 1 >= 20:
            break
    if count > 0:
        probability = (total / float(count)).clamp(epsilon, 1.0 - epsilon)
        rbm.bv.copy_(torch.log(probability / (1.0 - probability)))
        rbm.bv_vel.zero_()
        print(f"Initialized bv from data mean (logit(p)) using {count} samples (20 batches max).")


@torch.no_grad()
def _initialize_hidden_visible_bias(lower: BinaryRBM, upper: BinaryRBM, train_iter: TensorDataLoader, n_visible: int, epsilon: float, device: torch.device) -> None:
    total = torch.zeros(n_visible, device=device)
    count = 0
    for index, (batch, _) in enumerate(train_iter):
        batch = batch.to(device, non_blocking=True).reshape(batch.size(0), -1)
        probabilities, _ = lower.sample_h_given_v(binarize(batch, mode="stochastic", threshold=0.5))
        total += probabilities.sum(dim=0)
        count += probabilities.size(0)
        if index + 1 >= 20:
            break
    if count > 0:
        probability = (total / float(count)).clamp(epsilon, 1.0 - epsilon)
        upper.bv.copy_(torch.log(probability / (1.0 - probability)))
        upper.bv_vel.zero_()
        print(f"Initialized layer-2 visible bias from layer-1 activations using {count} samples.")


def _pretrain_layer1_dbm(rbm: BinaryRBM, train_iter: TensorDataLoader, *, cfg: _DBMConfig, device: torch.device, epochs: int, lr_base: float, cd_k: int, use_pcd: bool = False) -> DBMPretrainMetrics:
    metrics: DBMPretrainMetrics = {"epoch": [], "lr": [], "avg_mse": [], "avg_pl": []}
    persistent_v: torch.Tensor | None = None
    for epoch in tqdm(range(1, epochs + 1), desc="Pre-train Layer 1 (DBM)"):
        lr = lr_base * (cfg.lr_decay ** (epoch - 1))
        momentum = cfg.momentum_final if cfg.momentum_anneal_epoch > 1 and epoch >= cfg.momentum_anneal_epoch else cfg.momentum
        total_mse = 0.0
        n_batches = 0
        for batch, _ in train_iter:
            v0 = binarize(batch.to(device, non_blocking=True).reshape(batch.size(0), -1), mode=cfg.binarize_mode, threshold=cfg.threshold)
            batch_size = v0.size(0)
            p_h0 = torch.sigmoid(rbm.bh + v0 @ (2.0 * rbm.W))
            h0 = torch.bernoulli(p_h0)
            if use_pcd:
                if persistent_v is None or persistent_v.size(0) != batch_size:
                    persistent_v = torch.bernoulli(0.5 * torch.ones(batch_size, rbm.n_visible, device=device))
                vk = persistent_v
            else:
                vk = v0.clone()
            for _ in range(cd_k):
                hk = torch.bernoulli(torch.sigmoid(rbm.bh + vk @ rbm.W))
                vk = torch.bernoulli(rbm.prob_v_given_h(hk))
            p_hk = torch.sigmoid(rbm.bh + vk @ rbm.W)
            if use_pcd:
                persistent_v = vk
            dW = (v0.t() @ p_h0 - vk.t() @ p_hk) / batch_size - cfg.weight_decay * rbm.W.data
            dbv, dbh = (v0 - vk).mean(0), (p_h0 - p_hk).mean(0)
            rbm.W_vel = momentum * rbm.W_vel + lr * dW
            rbm.bv_vel = momentum * rbm.bv_vel + lr * dbv
            rbm.bh_vel = momentum * rbm.bh_vel + lr * dbh
            rbm.W.data.add_(rbm.W_vel)
            rbm.bv.data.add_(rbm.bv_vel)
            rbm.bh.data.add_(rbm.bh_vel)
            total_mse += ((v0 - rbm.prob_v_given_h(h0)) ** 2).mean().item()
            n_batches += 1
        average_mse = total_mse / max(1, n_batches)
        metrics["epoch"].append(epoch)
        metrics["lr"].append(lr)
        metrics["avg_mse"].append(average_mse)
        metrics["avg_pl"].append(float("nan"))
        tqdm.write(f"  [L1] epoch {epoch:3d}/{epochs} | lr={lr:.5f} | mse={average_mse:.5f}")
    return metrics


def _pretrain_layer2_dbm(rbm: BinaryRBM, train_iter: TensorDataLoader, *, cfg: _DBMConfig, device: torch.device, transform: Callable[[torch.Tensor], tuple[torch.Tensor, None]]) -> tuple[StepMetrics, EpochMetrics]:
    step_metrics: StepMetrics = {} if not cfg.monitor_step_metrics else {"step": [], "epoch": [], "step_in_epoch": [], "recon_mse": [], "recon_bce": [], "energy": [], "free_energy": []}
    epoch_metrics: EpochMetrics = {"epoch": [], "lr": [], "momentum": [], "avg_mse": [], "avg_bce": [], "avg_pl": [], "final_energy": [], "final_free_energy": []}
    global_steps = 0
    timer = Timer()
    for epoch in tqdm(range(1, cfg.pretrain_epochs2 + 1), total=cfg.pretrain_epochs2, desc="Layer2", unit="epoch"):
        timer.start()
        rbm.train()
        lr = cfg.pretrain_lr2 * (cfg.lr_decay ** (epoch - 1))
        momentum = cfg.momentum_final if cfg.momentum_anneal_epoch > 1 and epoch >= cfg.momentum_anneal_epoch else cfg.momentum
        mse_sum = bce_sum = pl_sum = 0.0
        logged_steps = epoch_steps = 0
        last_v0: torch.Tensor | None = None
        for step, (batch, _) in enumerate(train_iter, start=1):
            v0, _ = transform(batch)
            last_v0 = v0.detach()
            global_steps += 1
            log_metrics = cfg.monitor_step_metrics and (global_steps == 1 or global_steps % max(1, cfg.log_every_steps) == 0)
            stats = rbm.cd_k_update(v0=v0, lr=lr, k=cfg.pretrain_cd_k2, momentum=momentum, weight_decay=cfg.weight_decay, use_pcd=False, pcd_init="random", max_w_norm=cfg.max_w_norm, compute_pl=(cfg.monitor_pseudo_likelihood and log_metrics), compute_metrics=log_metrics)
            if log_metrics:
                mse_sum += stats["mse"]
                bce_sum += stats["bce"]
                pl_sum += stats.get("pl", 0.0)
                for name, value in (("step", global_steps), ("epoch", epoch), ("step_in_epoch", step), ("recon_mse", stats["mse"]), ("recon_bce", stats["bce"]), ("energy", stats["energy"]), ("free_energy", stats["free_energy"])):
                    step_metrics.setdefault(name, []).append(value)
                logged_steps += 1
            epoch_steps += 1
        final_energy = final_free_energy = math.nan
        if last_v0 is not None:
            energy, free_energy, _ = rbm_energy_free_energy_fast(rbm, last_v0, h=None)
            final_energy, final_free_energy = energy.mean().item(), free_energy.mean().item()
        if cfg.monitor_step_metrics:
            average_mse = mse_sum / max(1, logged_steps) if logged_steps else math.nan
            average_bce = bce_sum / max(1, logged_steps) if logged_steps else math.nan
            average_pl = pl_sum / max(1, logged_steps) if cfg.monitor_pseudo_likelihood and logged_steps else math.nan
        elif last_v0 is None:
            average_mse = average_bce = average_pl = math.nan
        else:
            p_v = rbm.prob_v_given_h(torch.bernoulli(rbm.prob_h_given_v(last_v0)))
            average_mse = torch.mean((last_v0 - p_v) ** 2).item()
            average_bce = F.binary_cross_entropy(p_v.clamp(1e-6, 1 - 1e-6), last_v0).item()
            average_pl = rbm.pseudo_likelihood(last_v0).item() if cfg.monitor_pseudo_likelihood else math.nan
        for name, value in (("epoch", epoch), ("lr", lr), ("momentum", momentum), ("avg_mse", average_mse), ("avg_bce", average_bce), ("avg_pl", average_pl), ("final_energy", final_energy), ("final_free_energy", final_free_energy)):
            epoch_metrics[name].append(value)
        timer.stop()
    print(f"[Layer 2] Total time: {timer.format_time(timer.sum())}")
    save_rbm_training_artifacts(cfg=cfg, train_iter=train_iter, step_metrics=step_metrics, epoch_metrics=epoch_metrics, rbm=rbm, ckpt_name=cfg.ckpt_pt2_name, layer_title="Layer 2")
    return step_metrics, epoch_metrics


def _plot_metric_curve(xs: list[NumericScalar], ys: list[NumericScalar], path: str, *, xlabel: str, ylabel: str, title: str) -> None:
    from dl_utils.plot._backend import pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(xs, ys)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_recon_grid(dbm: DBM, train_iter: TensorDataLoader, cfg: _DBMConfig, device: torch.device, epoch: int) -> None:
    recon_dir = os.path.join(cfg.out_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    for batch, _ in train_iter:
        v0 = binarize(batch.to(device, non_blocking=True).reshape(batch.size(0), -1), mode=cfg.binarize_mode, threshold=cfg.threshold)
        count = min(25, v0.size(0))
        v0 = v0[:count]
        mu1, _ = dbm.mean_field(v0, n_steps=cfg.mf_steps)
        combined = torch.cat([v0.cpu(), dbm.prob_v_given_h1(torch.bernoulli(mu1)).cpu()], dim=0)
        save_grid(combined, os.path.join(recon_dir, f"epoch{epoch:03d}_recon.png"), nrow=int(math.sqrt(count)), title=f"Top: original | Bottom: DBM recon (epoch {epoch})", hw=(cfg.size_h, cfg.size_w))
        break


def train_dbm(cfg: _DBMConfig, device: torch.device) -> None:
    reset_dir(cfg.out_dir)
    set_seed(cfg.seed)
    train_iter, _ = vision_loaders(dataset=cfg.dataset, data_dir=cfg.data_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, resize=cfg.resize)
    metrics_dir = os.path.join(cfg.out_dir, cfg.metrics_subdir)
    os.makedirs(metrics_dir, exist_ok=True)
    print("\n" + "=" * 60 + "\nPhase 1: Pre-training Layer 1 (DBM doubled-weight trick)\n" + "=" * 60)
    rbm1 = BinaryRBM(cfg.n_visible, cfg.n_hidden1, device=device)
    if cfg.init_bv_from_data:
        _initialize_visible_bias(rbm1, train_iter, cfg.n_visible, cfg.init_bv_eps, device)
    pt1_metrics = _pretrain_layer1_dbm(rbm1, train_iter, cfg=cfg, device=device, epochs=cfg.pretrain_epochs1, lr_base=cfg.pretrain_lr1, cd_k=cfg.pretrain_cd_k1, use_pcd=cfg.pretrain_use_pcd)
    layer1_dir = os.path.join(cfg.out_dir, "layer1")
    layer1_metrics_dir = os.path.join(layer1_dir, cfg.metrics_subdir)
    os.makedirs(layer1_metrics_dir, exist_ok=True)
    torch.save({"W": rbm1.W.data.cpu(), "bv": rbm1.bv.data.cpu(), "bh": rbm1.bh.data.cpu()}, os.path.join(layer1_dir, cfg.ckpt_pt1_name))
    save_metrics_csv(pt1_metrics, os.path.join(layer1_metrics_dir, "pretrain_layer1_epoch.csv"))
    _plot_metric_curve(pt1_metrics["epoch"], pt1_metrics["avg_mse"], os.path.join(layer1_metrics_dir, "pretrain_layer1_mse.png"), xlabel="Epoch", ylabel="Recon MSE", title="Layer 1 Pre-training MSE")
    print("\n" + "=" * 60 + "\nPhase 2: Pre-training Layer 2 (standard RBM on h1 samples)\n" + "=" * 60)
    rbm2 = BinaryRBM(cfg.n_hidden1, cfg.n_hidden2, device=device)
    _initialize_hidden_visible_bias(rbm1, rbm2, train_iter, cfg.n_hidden1, cfg.init_bv_eps, device)
    def transform_l2(batch: torch.Tensor) -> tuple[torch.Tensor, None]:
        visible = binarize(batch.to(device, non_blocking=True).reshape(batch.size(0), -1), mode=cfg.binarize_mode, threshold=cfg.threshold)
        return rbm1.sample_h_given_v(visible)[1], None
    _pretrain_layer2_dbm(rbm2, train_iter, cfg=layer_cfg(cfg, layer_idx=2), device=device, transform=transform_l2)
    print("\n" + "=" * 60 + "\nPhase 3: Initializing DBM from pre-trained RBMs\n" + "=" * 60)
    dbm = DBM(cfg.n_visible, cfg.n_hidden1, cfg.n_hidden2, device)
    for target, source in ((dbm.W1, rbm1.W), (dbm.bv, rbm1.bv), (dbm.bh1, rbm1.bh), (dbm.W2, rbm2.W), (dbm.bh2, rbm2.bh)):
        target.data.copy_(source.data)
    print("DBM initialized from pre-trained weights.")
    print(f"  W1: {dbm.W1.shape}, W2: {dbm.W2.shape}")
    print("\n" + "=" * 60 + "\nPhase 4: Fine-tuning DBM (mean-field + PCD)\n" + "=" * 60)
    ft_metrics: MetricHistory = {"epoch": [], "lr": [], "avg_mse": []}
    timer = Timer()
    for epoch in tqdm(range(1, cfg.finetune_epochs + 1), total=cfg.finetune_epochs, desc="Fine-tune DBM", unit="epoch"):
        timer.start()
        lr = cfg.finetune_lr * (cfg.lr_decay ** (epoch - 1))
        momentum = cfg.momentum_final if cfg.momentum_anneal_epoch > 1 and epoch >= cfg.momentum_anneal_epoch else cfg.momentum
        total_mse = 0.0
        n_batches = 0
        for batch, _ in train_iter:
            v0 = binarize(batch.to(device, non_blocking=True).reshape(batch.size(0), -1), mode=cfg.binarize_mode, threshold=cfg.threshold)
            total_mse += dbm.finetune_step(v0, lr=lr, momentum=momentum, weight_decay=cfg.weight_decay, mf_steps=cfg.mf_steps, pcd_steps=cfg.finetune_pcd_steps, use_adam=cfg.finetune_use_adam, adam_beta1=cfg.adam_beta1, adam_beta2=cfg.adam_beta2, adam_eps=cfg.adam_eps, max_grad_norm=cfg.max_grad_norm, max_w_norm=cfg.max_w_norm, use_pcd=cfg.finetune_use_pcd)["mse"]
            n_batches += 1
        average_mse = total_mse / max(1, n_batches)
        timer.stop()
        ft_metrics["epoch"].append(epoch)
        ft_metrics["lr"].append(lr)
        ft_metrics["avg_mse"].append(average_mse)
        tqdm.write(f"  [FT] epoch {epoch:3d}/{cfg.finetune_epochs} | lr={lr:.6f} | mse={average_mse:.5f} | {timer.format_time(timer.times[-1])}")
        if epoch == 1 or epoch % cfg.save_every_epochs == 0:
            _save_recon_grid(dbm, train_iter, cfg, device, epoch)
    print(f"\nFine-tuning done. Total time: {timer.format_time(timer.sum())}")
    dbm.save(os.path.join(cfg.out_dir, cfg.ckpt_ft_name))
    save_metrics_csv(ft_metrics, os.path.join(metrics_dir, "finetune_epoch.csv"))
    _plot_metric_curve(ft_metrics["epoch"], ft_metrics["avg_mse"], os.path.join(metrics_dir, "finetune_mse.png"), xlabel="Epoch", ylabel="Recon MSE", title="DBM Fine-tuning MSE")
    print(f"\nAll outputs written to: {cfg.out_dir}\n")
