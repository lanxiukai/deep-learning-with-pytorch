"""Checkpoint and kimg metric-history contracts for GAN lessons."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from dl_utils.training.checkpoints import TrainingCheckpoint


def start_gan_checkpoint(
    path: str | PathLike[str],
    *,
    resume_from: str | PathLike[str] | None,
    unit: str,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    metric_names: tuple[str, ...],
    fixed_z: torch.Tensor,
    run_config: Mapping[str, Any],
    extra_state: Mapping[str, Any] | None = None,
) -> tuple[TrainingCheckpoint, int, dict[str, Any]]:
    """Start one compatible latest-checkpoint stream for a GAN lesson."""
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("metric_names must be non-empty and unique.")
    reserved = {"loss_history", "fixed_z", "run_config"}
    extras = dict(extra_state or {})
    overlap = sorted(reserved & set(extras))
    if overlap:
        raise ValueError(f"extra_state uses reserved keys: {overlap}.")

    initial_state = {
        "loss_history": {
            "kimg": [],
            **{name: [] for name in metric_names},
        },
        "fixed_z": fixed_z.detach().cpu(),
        "run_config": dict(run_config),
        **extras,
    }
    checkpoint = TrainingCheckpoint(
        path,
        unit=unit,
        models=models,
        optimizers=optimizers,
    )
    completed_units, state = checkpoint.resume(
        resume_from,
        initial_state=initial_state,
    )
    if set(state) != set(initial_state):
        raise ValueError(
            "Checkpoint training-state keys differ from this lesson; "
            f"saved={sorted(state)}, expected={sorted(initial_state)}."
        )
    if state["run_config"] != initial_state["run_config"]:
        raise ValueError(
            "Checkpoint runtime options differ from this run; reuse the "
            "original batch, budget, regularization, and data-pipeline "
            "options."
        )
    history = state["loss_history"]
    expected_history_keys = {"kimg", *metric_names}
    if not isinstance(history, dict) or set(history) != expected_history_keys:
        raise ValueError("Checkpoint loss history does not match this lesson.")
    if not isinstance(state["fixed_z"], torch.Tensor):
        raise TypeError("Checkpoint fixed_z must be a tensor.")
    return checkpoint, completed_units, state


def append_gan_metrics(
    history: dict[str, list[float]],
    seen_kimg: float,
    metrics: Mapping[str, float],
) -> None:
    """Append one boundary's ordered metrics to a GAN loss history."""
    expected_names = set(history) - {"kimg"}
    if set(metrics) != expected_names:
        raise ValueError("metrics do not match the configured loss history.")
    history["kimg"].append(float(seen_kimg))
    for name, value in metrics.items():
        history[name].append(float(value))


__all__ = ["append_gan_metrics", "start_gan_checkpoint"]
