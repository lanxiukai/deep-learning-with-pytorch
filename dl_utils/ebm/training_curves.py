"""Metric curve plotting for RBM training artifacts.

Extracted from ``training_artifacts.py`` to keep each module under 250 pure LOC.
"""

import os

from ..plot.figures import maybe_save_curve
from ..training.metrics import as_list
from dl_utils.ebm._ebm_types import EpochMetrics, StepMetrics


def _build_step_x(
    step_metrics: StepMetrics,
    steps_per_epoch: int | None,
) -> tuple[list[float], str]:
    """Build x-axis values for step-level metrics.

    Returns:
        (x_values, xlabel_string)
    """
    step_xlabel = "Epoch"
    step_epochs = as_list(step_metrics.get("epoch"))
    step_in_epoch = as_list(step_metrics.get("step_in_epoch"))
    if step_epochs and step_in_epoch and steps_per_epoch is not None:
        denom = max(1.0, float(steps_per_epoch))
        n = min(len(step_epochs), len(step_in_epoch))
        step_epoch_x = [
            (float(step_epochs[i]) - 1.0) + float(step_in_epoch[i]) / denom
            for i in range(n)
        ]
    else:
        step_xlabel = "Step"
        step_x = as_list(step_metrics.get("step"))
        if step_x:
            step_epoch_x = [float(v) for v in step_x]
        else:
            n = max((len(as_list(v)) for v in step_metrics.values()), default=0)
            step_epoch_x = [float(i + 1) for i in range(n)]
    return step_epoch_x, step_xlabel


def _build_epoch_x(epoch_metrics: EpochMetrics) -> list[float]:
    """Build x-axis values for epoch-level metrics."""
    epoch_x_raw = as_list(epoch_metrics.get("epoch"))
    if epoch_x_raw:
        epoch_x = [float(v) for v in epoch_x_raw]
    else:
        n = max((len(as_list(v)) for v in epoch_metrics.values()), default=0)
        epoch_x = [float(i + 1) for i in range(n)]
    return epoch_x


def save_metric_curves(
    train_iter,
    step_metrics: StepMetrics,
    epoch_metrics: EpochMetrics,
    metrics_dir: str,
    *,
    layer_title: str | None = None,
    verbose: bool = False,
) -> None:
    """Build x-axes and save all metric curve plots (step-level and epoch-level).

    Called by :func:`save_rbm_training_artifacts`.
    """
    # --- steps per epoch (for fractional-epoch x-axis) ---
    steps_per_epoch = None
    if train_iter is not None:
        try:
            steps_per_epoch = len(train_iter)
        except Exception:
            steps_per_epoch = None

    step_epoch_x, step_xlabel = _build_step_x(step_metrics, steps_per_epoch)
    epoch_x = _build_epoch_x(epoch_metrics)

    def _with_layer_prefix(title: str) -> str:
        return f"{layer_title} - {title}" if layer_title else title

    # ---- step-level single-metric curves ----
    maybe_save_curve(
        step_epoch_x,
        step_metrics,
        {"recon_mse": "recon_mse"},
        os.path.join(metrics_dir, "step_recon_mse.png"),
        xlabel=step_xlabel,
        ylabel="Loss",
        title=_with_layer_prefix(
            "Reconstruction MSE vs epoch (step metrics)"
            if step_xlabel == "Epoch"
            else "Reconstruction MSE vs step (step metrics)"
        ),
        verbose=verbose,
    )
    maybe_save_curve(
        step_epoch_x,
        step_metrics,
        {"recon_bce": "recon_bce"},
        os.path.join(metrics_dir, "step_recon_bce.png"),
        xlabel=step_xlabel,
        ylabel="Loss",
        title=_with_layer_prefix(
            "Reconstruction BCE vs epoch (step metrics)"
            if step_xlabel == "Epoch"
            else "Reconstruction BCE vs step (step metrics)"
        ),
        verbose=verbose,
    )
    maybe_save_curve(
        step_epoch_x,
        step_metrics,
        {"energy": "energy"},
        os.path.join(metrics_dir, "step_energy.png"),
        xlabel=step_xlabel,
        ylabel="Energy",
        title=_with_layer_prefix(
            "Energy vs epoch (step metrics)"
            if step_xlabel == "Epoch"
            else "Energy vs step (step metrics)"
        ),
        verbose=verbose,
    )
    maybe_save_curve(
        step_epoch_x,
        step_metrics,
        {"free_energy": "free_energy"},
        os.path.join(metrics_dir, "step_free_energy.png"),
        xlabel=step_xlabel,
        ylabel="Free energy",
        title=_with_layer_prefix(
            "Free energy vs epoch (step metrics)"
            if step_xlabel == "Epoch"
            else "Free energy vs step (step metrics)"
        ),
        verbose=verbose,
    )

    # ---- step-level comparison curves ----
    maybe_save_curve(
        step_epoch_x,
        step_metrics,
        {"recon_mse": "recon_mse", "recon_bce": "recon_bce"},
        os.path.join(metrics_dir, "step_recon_losses.png"),
        xlabel=step_xlabel,
        ylabel="Loss",
        title=_with_layer_prefix(
            "Reconstruction losses vs epoch (step metrics)"
            if step_xlabel == "Epoch"
            else "Reconstruction losses vs step (step metrics)"
        ),
        verbose=verbose,
    )
    maybe_save_curve(
        step_epoch_x,
        step_metrics,
        {"energy": "energy", "free_energy": "free_energy"},
        os.path.join(metrics_dir, "step_energy_free_energy.png"),
        xlabel=step_xlabel,
        ylabel="Energy",
        title=_with_layer_prefix(
            "Energy and free energy vs epoch (step metrics)"
            if step_xlabel == "Epoch"
            else "Energy and free energy vs step (step metrics)"
        ),
        verbose=verbose,
    )

    # ---- epoch-level single-metric curves ----
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"lr": "lr"},
        os.path.join(metrics_dir, "epoch_lr.png"),
        xlabel="Epoch",
        ylabel="Learning rate",
        title=_with_layer_prefix("Learning rate vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"momentum": "momentum"},
        os.path.join(metrics_dir, "epoch_momentum.png"),
        xlabel="Epoch",
        ylabel="Momentum",
        title=_with_layer_prefix("Momentum vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"avg_mse": "avg_mse"},
        os.path.join(metrics_dir, "epoch_avg_mse.png"),
        xlabel="Epoch",
        ylabel="Loss",
        title=_with_layer_prefix("Average MSE vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"avg_bce": "avg_bce"},
        os.path.join(metrics_dir, "epoch_avg_bce.png"),
        xlabel="Epoch",
        ylabel="Loss",
        title=_with_layer_prefix("Average BCE vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"avg_pl": "avg_pl"},
        os.path.join(metrics_dir, "epoch_avg_pseudo_likelihood.png"),
        xlabel="Epoch",
        ylabel="Log pseudo-likelihood",
        title=_with_layer_prefix("Average log pseudo-likelihood vs epoch"),
        verbose=verbose,
    )

    # ---- epoch-final energy curves ----
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"final_energy": "final_energy"},
        os.path.join(metrics_dir, "epoch_final_energy.png"),
        xlabel="Epoch",
        ylabel="Energy",
        title=_with_layer_prefix("Final energy vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"final_free_energy": "final_free_energy"},
        os.path.join(metrics_dir, "epoch_final_free_energy.png"),
        xlabel="Epoch",
        ylabel="Free energy",
        title=_with_layer_prefix("Final free energy vs epoch"),
        verbose=verbose,
    )

    # ---- epoch-level comparison curves ----
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"lr": "lr", "momentum": "momentum"},
        os.path.join(metrics_dir, "epoch_lr_momentum.png"),
        xlabel="Epoch",
        ylabel="Value",
        title=_with_layer_prefix("Learning rate and momentum vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"avg_mse": "avg_mse", "avg_bce": "avg_bce"},
        os.path.join(metrics_dir, "epoch_recon_losses.png"),
        xlabel="Epoch",
        ylabel="Loss",
        title=_with_layer_prefix("Reconstruction losses vs epoch"),
        verbose=verbose,
    )
    maybe_save_curve(
        epoch_x,
        epoch_metrics,
        {"final_energy": "final_energy", "final_free_energy": "final_free_energy"},
        os.path.join(metrics_dir, "epoch_final_energy_free_energy.png"),
        xlabel="Epoch",
        ylabel="Value",
        title=_with_layer_prefix("Final energy and free energy vs epoch"),
        verbose=verbose,
    )
