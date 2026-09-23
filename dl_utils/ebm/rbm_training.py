import math
from collections.abc import Callable

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..data.vision import TensorDataLoader, vision_loaders
from ..filesystem.directories import reset_dir
from ..runtime.randomness import set_seed
from ..training.timing import Timer
from ._ebm_types import (
    EpochMetrics,
    StepMetrics,
    _RBMFixedPointConfig,
    _RBMRunConfig,
    _RBMSetupConfig,
    _RBMTrainingConfig,
)
from .rbm_artifacts import save_filters_grids, save_rbm_recon_grids
from .rbm_model import BinaryRBM
from .rbm_primitives import binarize, rbm_energy_free_energy_fast
from .training_artifacts import save_rbm_training_artifacts


@torch.no_grad()
def _init_bias_from_stats(
    rbm: BinaryRBM, sum_vals: torch.Tensor, n: int, init_bv_eps: float,
) -> None:
    if n > 0:
        p = (sum_vals / float(n)).clamp(init_bv_eps, 1.0 - init_bv_eps)
        rbm.bv.copy_(torch.log(p / (1.0 - p)))
        rbm.bv_vel.zero_()


@torch.no_grad()
def init_bv_from_data(
    rbm: BinaryRBM,
    train_iter: TensorDataLoader,
    n_visible: int,
    init_bv_eps: float,
    device: torch.device,
    max_batches: int = 20,
) -> None:
    sum_x = torch.zeros(n_visible, device=device)
    n = 0
    for i, (xb, _) in enumerate(train_iter):
        xb = xb.to(device, non_blocking=True).reshape(xb.size(0), -1)
        sum_x += xb.sum(dim=0)
        n += xb.size(0)
        if i + 1 >= max_batches:
            break
    if n > 0:
        _init_bias_from_stats(rbm, sum_x, n, init_bv_eps)
        print(f"Initialized bv from data mean (logit(p)) using {n} samples ({max_batches} batches max).")


@torch.no_grad()
def init_hidden_visible_bias(
    rbm_lower: BinaryRBM,
    rbm_upper: BinaryRBM,
    train_iter: TensorDataLoader,
    n_visible: int,
    init_bv_eps: float,
    device: torch.device,
    max_batches: int = 20,
) -> None:
    sum_h = torch.zeros(n_visible, device=device)
    n = 0
    for i, (xb, _) in enumerate(train_iter):
        xb = xb.to(device, non_blocking=True).reshape(xb.size(0), -1)
        vb = binarize(xb, mode="stochastic", threshold=0.5)
        p_h, _ = rbm_lower.sample_h_given_v(vb)
        sum_h += p_h.sum(dim=0)
        n += p_h.size(0)
        if i + 1 >= max_batches:
            break
    if n > 0:
        _init_bias_from_stats(rbm_upper, sum_h, n, init_bv_eps)
        print(f"Initialized layer-2 visible bias from layer-1 activations using {n} samples.")


def train_rbm_layer(
    rbm: BinaryRBM,
    train_iter: TensorDataLoader,
    *,
    cfg: _RBMTrainingConfig,
    device: torch.device,
    epochs: int,
    lr_base: float,
    cd_k: int,
    use_pcd: bool,
    pcd_init: str,
    layer_idx: int,
    transform: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]],
    gen_iter: TensorDataLoader | None = None,
    ckpt_name: str | None = None,
) -> tuple[StepMetrics, EpochMetrics]:
    monitor_step_metrics = cfg.monitor_step_metrics
    step_metrics: StepMetrics = {}
    if monitor_step_metrics:
        step_metrics = {
            "step": [], "epoch": [], "step_in_epoch": [], "recon_mse": [],
            "recon_bce": [], "energy": [], "free_energy": [],
        }
    epoch_metrics: EpochMetrics = {
        "epoch": [], "lr": [], "momentum": [], "avg_mse": [], "avg_bce": [],
        "avg_pl": [], "final_energy": [], "final_free_energy": [],
    }
    global_steps = 0
    timer = Timer()
    epoch_iter = tqdm(range(1, epochs + 1), total=epochs, desc=f"Layer{layer_idx}", unit="epoch")
    save_every_epochs = max(1, cfg.save_every_epochs)
    compute_pl = cfg.monitor_pseudo_likelihood
    log_every_steps = max(1, cfg.log_every_steps)
    for epoch in epoch_iter:
        timer.start()
        rbm.train()
        save_this_epoch = (epoch == 1) or (epoch % save_every_epochs == 0)
        running_mse = running_bce = running_pl = 0.0
        metrics_steps = epoch_steps = 0
        last_v0: torch.Tensor | None = None
        lr_epoch = lr_base * (cfg.lr_decay ** (epoch - 1))
        momentum_epoch = cfg.momentum_final if (
            cfg.momentum_anneal_epoch > 1 and epoch >= cfg.momentum_anneal_epoch
        ) else cfg.momentum
        for step, (x, _) in enumerate(train_iter, start=1):
            v0, x_orig = transform(x)
            last_v0 = v0.detach()
            next_global_step = global_steps + 1
            do_step_metrics = monitor_step_metrics and (
                (next_global_step == 1) or (next_global_step % log_every_steps == 0)
            )
            stats = rbm.cd_k_update(
                v0=v0, lr=lr_epoch, k=cd_k, momentum=momentum_epoch,
                weight_decay=cfg.weight_decay, use_pcd=use_pcd, pcd_init=pcd_init,
                max_w_norm=cfg.max_w_norm, compute_pl=(compute_pl and do_step_metrics),
                compute_metrics=do_step_metrics,
            )
            if do_step_metrics:
                running_mse += stats["mse"]
                running_bce += stats["bce"]
                if compute_pl:
                    running_pl += stats.get("pl", 0.0)
                metrics_steps += 1
            epoch_steps += 1
            global_steps += 1
            if do_step_metrics:
                step_metrics.setdefault("step", []).append(global_steps)
                step_metrics.setdefault("epoch", []).append(epoch)
                step_metrics.setdefault("step_in_epoch", []).append(step)
                step_metrics.setdefault("recon_mse", []).append(stats["mse"])
                step_metrics.setdefault("recon_bce", []).append(stats["bce"])
                step_metrics.setdefault("energy", []).append(stats["energy"])
                step_metrics.setdefault("free_energy", []).append(stats["free_energy"])
            if monitor_step_metrics and layer_idx == 1 and save_this_epoch and step <= cfg.recon_vis_batches:
                save_rbm_recon_grids(rbm, v0, epoch, step, cfg, x_orig=x_orig)
        final_energy = math.nan
        final_free_energy = math.nan
        if last_v0 is not None:
            e_last, fe_last, _ = rbm_energy_free_energy_fast(rbm, last_v0, h=None)
            final_energy = e_last.mean().item()
            final_free_energy = fe_last.mean().item()
        if monitor_step_metrics:
            denom = max(1, metrics_steps)
            avg_mse = (running_mse / denom) if metrics_steps > 0 else float("nan")
            avg_bce = (running_bce / denom) if metrics_steps > 0 else float("nan")
            avg_pl = (running_pl / denom) if (compute_pl and metrics_steps > 0) else float("nan")
        else:
            avg_mse = avg_bce = avg_pl = float("nan")
            if last_v0 is not None:
                p_h = rbm.prob_h_given_v(last_v0)
                h0 = torch.bernoulli(p_h)
                p_v1 = rbm.prob_v_given_h(h0)
                avg_mse = torch.mean((last_v0 - p_v1) ** 2).item()
                avg_bce = F.binary_cross_entropy(p_v1.clamp(1e-6, 1 - 1e-6), last_v0).item()
                avg_pl = rbm.pseudo_likelihood(last_v0).item() if compute_pl else float("nan")
        epoch_metrics["epoch"].append(epoch)
        epoch_metrics["lr"].append(lr_epoch)
        epoch_metrics["momentum"].append(momentum_epoch)
        epoch_metrics["avg_mse"].append(avg_mse)
        epoch_metrics["avg_bce"].append(avg_bce)
        epoch_metrics["avg_pl"].append(avg_pl)
        epoch_metrics["final_energy"].append(final_energy)
        epoch_metrics["final_free_energy"].append(final_free_energy)
        timer.stop()
        if save_this_epoch and layer_idx == 1 and cfg.save_filters:
            save_filters_grids(cfg=cfg, W=rbm.W.detach(), epoch=epoch)
    print(f"[Layer {layer_idx}] Total time: {timer.format_time(timer.sum())}")
    if ckpt_name is None:
        ckpt_name = f"layer{layer_idx}_rbm_{cfg.dataset}.pt"
    save_rbm_training_artifacts(
        cfg=cfg, train_iter=train_iter, step_metrics=step_metrics, epoch_metrics=epoch_metrics,
        rbm=rbm, ckpt_name=ckpt_name, layer_title=f"Layer {layer_idx}",
    )
    return step_metrics, epoch_metrics


def _prepare_layer1_rbm(
    cfg: _RBMSetupConfig, device: torch.device, n_hidden: int,
) -> tuple[BinaryRBM, TensorDataLoader, TensorDataLoader, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    reset_dir(cfg.out_dir)
    set_seed(cfg.seed)
    train_iter, test_iter = vision_loaders(
        dataset=cfg.dataset, data_dir=cfg.data_dir, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, resize=cfg.resize,
    )
    rbm = BinaryRBM(cfg.n_visible, n_hidden, device=device)
    if cfg.init_bv_from_data:
        init_bv_from_data(rbm, train_iter, cfg.n_visible, cfg.init_bv_eps, device)
    _configure_rbm_fixed_point(rbm, cfg)

    def transform(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(device, non_blocking=True).reshape(x.size(0), -1)
        v0 = binarize(x, mode=cfg.binarize_mode, threshold=cfg.threshold)
        return v0, x

    return rbm, train_iter, test_iter, transform


def _configure_rbm_fixed_point(rbm: BinaryRBM, cfg: _RBMFixedPointConfig) -> None:
    if cfg.rbm_param_fixed_point:
        rbm.enable_param_fixed_point_(True, q_frac=cfg.rbm_param_q_frac, quantize_now=True)


def train_rbm(cfg: _RBMRunConfig, device: torch.device) -> None:
    rbm, train_iter, test_iter, transform = _prepare_layer1_rbm(cfg, device, cfg.n_hidden)
    train_rbm_layer(
        rbm, train_iter, cfg=cfg, device=device, epochs=cfg.epochs, lr_base=cfg.lr,
        cd_k=cfg.cd_k, use_pcd=cfg.use_pcd, pcd_init=cfg.pcd_init, layer_idx=1,
        transform=transform, gen_iter=test_iter, ckpt_name=f"rbm_{cfg.dataset}.pt",
    )
    print("Done. Outputs written to:", cfg.out_dir, "\n")
