import os

import torch

from ..data.vision import TensorDataLoader
from ..training.metrics import (
    align_and_drop_all_nan_rows,
    align_metrics_for_csv,
    save_metrics_csv,
)
from dl_utils.ebm._ebm_types import EpochMetrics, StepMetrics, _RBMArtifactConfig
from dl_utils.ebm.rbm_model import BinaryRBM
from .training_curves import save_metric_curves


def save_rbm_training_artifacts(
    cfg: _RBMArtifactConfig,
    train_iter: TensorDataLoader | None,
    step_metrics: StepMetrics,
    epoch_metrics: EpochMetrics,
    rbm: BinaryRBM | None = None,
    *,
    metrics_subdir: str = "metrics",
    ckpt_name: str | None = None,
    verbose: bool = False,
    layer_title: str | None = None,
) -> str | None:
    """
    Save RBM training artifacts for reuse:
      - metrics CSV (step/epoch)
      - metric curves (only generated when metrics contain corresponding data)
      - optional RBM checkpoint (W/bv/bh + config)

    Returns:
        ckpt_path if checkpoint is saved, otherwise None.
    """
    metrics_dir = os.path.join(cfg.out_dir, metrics_subdir)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save raw metrics to CSV (robustly aligned)
    if step_metrics:
        # Step metrics often contain NaN placeholders for un-logged steps; drop those rows when saving.
        step_metrics = align_and_drop_all_nan_rows(step_metrics)
        save_metrics_csv(step_metrics, os.path.join(metrics_dir, "step_metrics.csv"))
    if epoch_metrics:
        save_metrics_csv(align_metrics_for_csv(epoch_metrics), os.path.join(metrics_dir, "epoch_metrics.csv"))

    # Save metric curves (x-axes built inside the helper)
    save_metric_curves(
        train_iter, step_metrics, epoch_metrics, metrics_dir,
        layer_title=layer_title,
        verbose=verbose,
    )

    # Optional checkpoint
    if rbm is None:
        return None

    if ckpt_name is None:
        dataset = cfg.dataset
        ckpt_name = f"rbm_{dataset}.pt"

    ckpt_path = os.path.join(cfg.out_dir, ckpt_name)
    torch.save(
        {
            "config": vars(cfg),
            "W": rbm.W.detach().cpu(),
            "bv": rbm.bv.detach().cpu(),
            "bh": rbm.bh.detach().cpu(),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to: {ckpt_path}")
    return ckpt_path
