"""
Two-layer Deep Boltzmann Machine (DBM) — optimised training & sampling.

Improvements over baseline:
  - Multi-step PCD (finetune_pcd_steps=5) for better negative-phase mixing.
  - Adam optimiser with conservative LR for stable convergence.
  - PCD pre-training for Layer 1.
  - Chain desynchronisation to reduce lockstep evolution.
  - MCMC convergence diagnostics (R-hat, ESS, autocorrelation).
  - Focused hyperparameter sweep over LR + PCD steps.

Energy:  E(v, h1, h2) = -v^T W1 h1 - h1^T W2 h2 - bv^T v - bh1^T h1 - bh2^T h2

References:
  Salakhutdinov & Hinton (2009): "Deep Boltzmann Machines"
  Salakhutdinov & Larochelle (2010): "Efficient Learning of Deep Boltzmann Machines"
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import itertools
import os

import torch

from dl_utils.runtime.devices import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.ebm._ebm_types import Config as EBMConfig
from dl_utils.ebm.dbm_sampling import sampling_dbm
from dl_utils.ebm.dbm_training import train_dbm

_PROJECT_ROOT = infer_project_root()
_OUTPUT_ROOT = str(_PROJECT_ROOT / "output")


# ============================
# Config
# ============================
@dataclass(kw_only=True)
class Config(EBMConfig):
    seed: int | None = 42

    # System
    data_dir: str = str(_PROJECT_ROOT / "data" / "fashion_mnist")
    num_workers: int = 16
    pin_memory: bool = True

    # Dataset
    dataset: str = "fashion_mnist"
    resize: int | tuple[int, int] | None = None
    size_h: int = 28
    size_w: int = 28

    # Binarization
    binarize_mode: str = "stochastic"
    threshold: float = 0.5

    # Model sizes
    batch_size: int = 256
    n_visible: int = 784
    n_hidden1: int = 500
    n_hidden2: int = 1000

    # ---- Phase 1: Layer-1 pre-training (DBM doubled-weight trick) ----
    pretrain_epochs1: int = 60
    pretrain_lr1: float = 0.001
    pretrain_cd_k1: int = 5
    pretrain_use_pcd: bool = True                     # PCD persistent chains

    # ---- Phase 2: Layer-2 pre-training (standard RBM on h1) ----
    pretrain_epochs2: int = 60
    pretrain_lr2: float = 0.001
    pretrain_cd_k2: int = 5

    # ---- Phase 3: DBM fine-tuning (mean-field + PCD) ----
    finetune_epochs: int = 100                           # fine-tuning epochs
    finetune_lr: float = 0.0001                       # conservative for Adam stability
    finetune_pcd_steps: int = 5                       # multi-step PCD
    finetune_use_adam: bool = True
    finetune_use_pcd: bool = False                     # CD mode → prevents mode collapse
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 5.0                        # tighter clipping (was 10)
    mf_steps: int = 15                                # more MF iterations (was 10)

    # ---- Shared optimization ----
    lr_decay: float = 1.0                             # no decay (Adam handles it)
    momentum: float = 0.5
    momentum_final: float = 0.9
    momentum_anneal_epoch: int = 5
    weight_decay: float = 1e-5                        # lighter (was 1e-4)
    max_w_norm: float = 5.0                             # prevent pre-activation saturation
    init_bv_from_data: bool = True
    init_bv_eps: float = 1e-4
    rbm_param_fixed_point: bool = False
    rbm_param_q_frac: int = 12

    monitor_step_metrics: bool = True
    monitor_pseudo_likelihood: bool = True
    log_every_steps: int = 1
    recon_vis_batches: int = 1
    recon_vis_count: int = 25
    save_filters: bool = False
    max_filters: int = 25

    # ---- Output ----
    out_root: str = (
        f"{_OUTPUT_ROOT}/{dataset}-size={size_h}x{size_w}/"
        f"dbm-h1={n_hidden1}-h2={n_hidden2}/"
        f"pt1_{pretrain_epochs1}e_pcd{pretrain_cd_k1}-"
        f"pt2_{pretrain_epochs2}e_cd{pretrain_cd_k2}-"
        f"ft_{finetune_epochs}e_pcd{finetune_pcd_steps}_adam"
    )

    # ---- Visualization / generation ----
    gen_count: int = 25
    gen_gibbs_steps: int = 3000
    gen_init: str = "random"
    gen_seed: int | None = None
    gen_desync_steps: int = 100
    save_every_epochs: int = 10
    save_steps: tuple[int, ...] | None = (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000)
    save_steps_divisor: int = 1

    # MCMC convergence diagnostics
    mcmc_diagnostics: bool = True

    # Output subdirectories
    sample_root: str = "sample"
    grid_prob_subdir: str = "grids_prob"
    grid_binary_subdir: str = "grids_binary"
    metrics_subdir: str = "metrics"
    gif_subdir: str = "gifs"
    save_gif: bool = True
    gif_fps: int = 2

    # Checkpoint names
    ckpt_pt1_name: str = "pretrain_layer1.pt"
    ckpt_pt2_name: str = "pretrain_layer2.pt"
    ckpt_ft_name: str = "dbm_finetuned.pt"

    # Sweep mode
    sweep_mode: bool = False                          # set True after verification
    sweep_skip_training: bool = False
    sweep_skip_sampling: bool = False


# ============================
# Focused hyperparameter sweep
# ============================
# Set sweep_mode=True and adjust the grid below. Each combination trains+generates.
# Use CD mode (finetune_use_pcd=False) — proven to prevent mode collapse.

@dataclass(frozen=True)
class SweepParameters:
    finetune_lr: float
    finetune_pcd_steps: int


SWEEP_LEARNING_RATES: tuple[float, ...] = (0.00005, 0.0001, 0.0002)
SWEEP_PCD_STEPS: tuple[int, ...] = (3, 5, 10)
# Total: 3 × 3 = 9 runs. Each ~6 min → ~1 hour total.

SWEEP_OVERRIDE: list[SweepParameters] | None = None


def _make_sweep_combinations() -> list[SweepParameters]:
    if SWEEP_OVERRIDE is not None:
        return SWEEP_OVERRIDE
    return [
        SweepParameters(finetune_lr=lr, finetune_pcd_steps=pcd_steps)
        for lr, pcd_steps in itertools.product(SWEEP_LEARNING_RATES, SWEEP_PCD_STEPS)
    ]


def _sweep_name(params: SweepParameters) -> str:
    return f"finetune_lr={params.finetune_lr:.0e}-finetune_pcd_steps={params.finetune_pcd_steps}"


def run_sweep(base_cfg: Config, device: torch.device) -> None:
    combos = _make_sweep_combinations()
    if not combos:
        print("WARNING: SWEEP_GRID is empty. Nothing to sweep.")
        return
    print(f"\n{'='*60}")
    print(f"Hyperparameter Sweep: {len(combos)} combinations")
    print("  varying: finetune_lr, finetune_pcd_steps")
    print(f"{'='*60}\n")

    for idx, params in enumerate(combos, start=1):
        name = _sweep_name(params)
        print(f"\n{'─'*60}")
        print(f"Run {idx}/{len(combos)}: {name}")
        print(f"{'─'*60}")

        sweep_out = os.path.join(base_cfg.out_root, "sweep", name)
        cfg = replace(
            base_cfg,
            finetune_lr=params.finetune_lr,
            finetune_pcd_steps=params.finetune_pcd_steps,
            out_root=sweep_out,
            out_dir=sweep_out,
        )

        try:
            if not cfg.sweep_skip_training:
                train_dbm(cfg, device)
            if not cfg.sweep_skip_sampling:
                sampling_dbm(cfg, device)
            else:
                print(f"  [SKIP] sampling")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Sweep complete → {base_cfg.out_root}/sweep/")
    print(f"{'='*60}")


# ============================
# Entry point
# ============================
def main() -> None:
    device = try_gpu()
    cfg = Config()

    if cfg.sweep_mode:
        run_sweep(cfg, device)
        return

    cfg.out_dir = cfg.out_root
    print(f"Output:          {cfg.out_dir}")
    print(f"Pre-train L1:    {cfg.pretrain_epochs1}e, {'PCD' if cfg.pretrain_use_pcd else 'CD'}-{cfg.pretrain_cd_k1}, lr={cfg.pretrain_lr1}")
    print(f"Pre-train L2:    {cfg.pretrain_epochs2}e, CD-{cfg.pretrain_cd_k2}, lr={cfg.pretrain_lr2}")
    print(f"Fine-tune:       {cfg.finetune_epochs}e, Adam, PCD-{cfg.finetune_pcd_steps}, lr={cfg.finetune_lr}")
    print(f"Grad clip:       {cfg.max_grad_norm}")
    print(f"MF steps:        {cfg.mf_steps}")
    print(f"MCMC diag:       {'ON' if cfg.mcmc_diagnostics else 'OFF'}")

    train_dbm(cfg, device)
    sampling_dbm(cfg, device)


if __name__ == "__main__":
    main()
