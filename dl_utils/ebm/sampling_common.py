import os
from collections.abc import Sized

import torch

from ..data.vision import TensorDataLoader, vision_loaders
from ..filesystem.directories import reset_dir
from dl_utils.ebm._ebm_types import (
    SamplingOutputDirs,
    _DBMConfig,
    _DBNSamplingConfig,
    _InitialVisibleConfig,
    _RBMFixedPointConfig,
    _RBMLoadConfig,
    _RBMSamplingConfig,
    _SamplingOutputConfig,
)
from dl_utils.ebm.rbm_model import BinaryRBM
from dl_utils.ebm.rbm_primitives import binarize
from dl_utils.ebm.sampling_artifacts import ensure_dir


def _infer_gen_desync_steps(cfg: _RBMSamplingConfig | _DBNSamplingConfig) -> int:
    """
    Per-sample random burn-in offset (a.k.a. chain "desync") helps reduce the
    "synchronized evolution" look across a batch of parallel chains.

    Priority:
      1) cfg.gen_desync_steps if provided (0 disables)
      2) cfg.save_steps_divisor as a sensible default for visualization cadence
    """
    raw = cfg.gen_desync_steps
    if raw is None:
        raw = cfg.save_steps_divisor
    return max(0, raw)


def _parse_save_steps(
    cfg: _RBMSamplingConfig | _DBNSamplingConfig | _DBMConfig,
    *,
    divisor_default: bool,
) -> list[int]:
    if cfg.save_steps is None:
        if divisor_default:
            divisor = max(1, cfg.save_steps_divisor)
            raw_steps = [1] + [
                step
                for step in range(1, cfg.gen_gibbs_steps + 1)
                if step % divisor == 0
            ]
        else:
            raw_steps = range(1, cfg.gen_gibbs_steps + 1)
    else:
        raw_steps = list(cfg.save_steps)

    return sorted(
        {int(step) for step in raw_steps if 0 < int(step) <= cfg.gen_gibbs_steps}
    )


@torch.no_grad()
def _desync_prepare(v: torch.Tensor, max_offset_steps: int) -> tuple[int, torch.Tensor | None]:
    """
    Compute random per-sample chain offsets for Gibbs desynchronisation.

    Returns (max_steps, offsets) where offsets.shape = (B,) and max_steps = max(offsets).
    Returns (0, None) if desync should be skipped (disabled, single sample, etc.).
    """
    if max_offset_steps <= 0:
        return 0, None
    if v.dim() != 2 or v.size(0) <= 1:
        return 0, None
    B = v.size(0)
    offsets = torch.randint(0, max_offset_steps + 1, (B,), device=v.device)
    max_steps = int(offsets.max().item())
    if max_steps <= 0:
        return 0, None
    return max_steps, offsets


@torch.no_grad()
def _desync_gibbs_visible_chains(rbm: BinaryRBM, v: torch.Tensor, max_offset_steps: int) -> torch.Tensor:
    """
    Run a different number of burn-in Gibbs steps for each sample in the batch.
    This decorrelates ("de-coheres") samples that otherwise start at the same time.
    """
    max_steps, offsets = _desync_prepare(v, max_offset_steps)
    if offsets is None:
        return v

    for t in range(1, max_steps + 1):
        _, _, _, v_new = rbm.gibbs_vhv(v)
        mask = (offsets >= t).unsqueeze(1)
        v = torch.where(mask, v_new, v)
    return v


def _sampling_output_dirs(cfg: _SamplingOutputConfig) -> SamplingOutputDirs:
    """Set up and reset all sampling output directories for a single call site."""
    sample_root = os.path.join(cfg.out_dir, cfg.sample_root)
    reset_dir(sample_root)
    data_save_enabled = cfg.data_save
    return {
        "sample_root": sample_root,
        "data_dir": ensure_dir(os.path.join(sample_root, cfg.data_subdir)) if data_save_enabled else None,
        "data_save_enabled": data_save_enabled,
        "grid_prob_dir": ensure_dir(os.path.join(sample_root, cfg.grid_prob_subdir)),
        "grid_binary_dir": ensure_dir(os.path.join(sample_root, cfg.grid_binary_subdir)),
        "timeline_prob_dir": ensure_dir(os.path.join(sample_root, cfg.timeline_prob_subdir)),
        "timeline_binary_dir": ensure_dir(os.path.join(sample_root, cfg.timeline_binary_subdir)),
        "gif_dir": ensure_dir(os.path.join(sample_root, cfg.gif_subdir)),
        "metrics_dir": ensure_dir(os.path.join(sample_root, cfg.metrics_subdir)),
    }


def checkpoint_path(cfg: _RBMLoadConfig) -> str:
    name = cfg.ckpt_name or f"rbm_{cfg.dataset}.pt"
    return os.path.join(cfg.out_dir, name)


def validate_out_dir_and_ckpt(cfg: _RBMLoadConfig) -> str:
    """
    Ensure output directory and checkpoint file exist.
    Returns the checkpoint path if both checks pass.
    """
    out_dir = os.path.abspath(cfg.out_dir)
    if not out_dir or not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    ckpt = checkpoint_path(cfg)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _configure_rbm_fixed_point(rbm: BinaryRBM, cfg: _RBMFixedPointConfig) -> None:
    if cfg.rbm_param_fixed_point:
        rbm.enable_param_fixed_point_(
            True,
            q_frac=cfg.rbm_param_q_frac,
            quantize_now=True,
        )


def load_rbm_model_from_ckpt(cfg: _RBMLoadConfig, device: torch.device) -> BinaryRBM:
    ckpt_path = checkpoint_path(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    rbm = BinaryRBM(cfg.n_visible, cfg.n_hidden, device=device)
    if "W" not in ckpt or "bv" not in ckpt or "bh" not in ckpt:
        raise KeyError(f"Checkpoint missing required keys: {ckpt_path}")
    rbm.W.data.copy_(ckpt["W"].to(device))
    rbm.bv.data.copy_(ckpt["bv"].to(device))
    rbm.bh.data.copy_(ckpt["bh"].to(device))
    rbm.W_vel.zero_()
    rbm.bv_vel.zero_()
    rbm.bh_vel.zero_()

    # Optional: enforce FPGA-aligned int16 fixed-point parameter grid at load time
    # (useful when sampling a float-trained checkpoint with fixed-point parameters).
    _configure_rbm_fixed_point(rbm, cfg)
    return rbm


def prepare_init_visible(
    cfg: _InitialVisibleConfig,
    device: torch.device,
) -> tuple[torch.Tensor, TensorDataLoader | None]:
    if cfg.gen_init == "data":
        _, test_iter = vision_loaders(
            dataset=cfg.dataset,
            data_dir=cfg.data_dir,
            batch_size=max(cfg.gen_count, 1),
            resize=cfg.resize,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        # Randomly pick cfg.gen_count samples from the dataset instead of always
        # taking the first batch (and repeating if too few).
        test_ds = test_iter.dataset
        if not isinstance(test_ds, Sized):
            raise TypeError("prepare_init_visible requires a sized dataset.")
        total = len(test_ds)
        if total <= 0:
            raise ValueError("prepare_init_visible: empty dataset for gen_init='data'.")

        n = int(cfg.gen_count)
        if total >= n:
            idx = torch.randperm(total)[:n]
        else:
            # sample with replacement if dataset is smaller than requested count
            idx = torch.randint(0, total, (n,))

        xs = []
        for i in idx.tolist():
            x, _ = test_ds[int(i)]
            xs.append(x)
        xb = torch.stack(xs, dim=0)
        xb = xb.to(device).reshape(xb.size(0), -1)
        v0 = binarize(xb, mode=cfg.binarize_mode, threshold=cfg.threshold)
        return v0, test_iter

    v0 = torch.bernoulli(0.5 * torch.ones(cfg.gen_count, cfg.n_visible, device=device))
    return v0, None
