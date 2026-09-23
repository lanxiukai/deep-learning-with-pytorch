"""CelebA GAN runtime, model setup, checkpoints, and metric histories."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from dl_utils.data.datasets.celeba import CelebATrainingStream
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import configure_device, try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import TrainingCheckpoint
from dl_utils.training.precision import BF16Precision, resolve_bf16_precision


@dataclass
class GANRun:
    """Own common paths, BF16 runtime state, and the CelebA stream."""

    output_dir: Path
    training_dir: Path
    checkpoint_dir: Path
    device: torch.device
    precision: BF16Precision
    data: CelebATrainingStream

    @property
    def pipeline(self) -> str:
        return self.data.pipeline

    @property
    def dataset_size(self) -> int:
        return self.data.dataset_size

    def prepare_output(self, resume_from: str | PathLike[str] | None) -> None:
        """Reset transient artifacts for a fresh run or retain them on resume."""
        if resume_from is None:
            reset_dir(str(self.training_dir))
            reset_dir(str(self.checkpoint_dir))
            return
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def prepare_gan_run(
    model_name: str,
    *,
    seed: int,
    data_pipeline: str,
    num_workers: int,
    prefetch_factor: int,
    project_root: str | PathLike[str] | None = None,
) -> GANRun:
    """Validate data and construct the single-GPU BF16 lesson runtime."""
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be one safe path component.")
    root = (
        infer_project_root() if project_root is None else Path(project_root).resolve()
    )
    data_dir = root / "data" / "celeba"
    if not (data_dir / "list_eval_partition.csv").is_file():
        raise FileNotFoundError(
            f"CelebA data not found: {data_dir}. "
            "Run tool_scripts/download_dataset.py first."
        )

    output_root = Path(os.environ.get("DL_OUTPUT_ROOT", str(root / "output" / "gan")))
    output_dir = output_root / model_name
    set_seed(seed)
    device = try_gpu()
    configure_device(device)
    precision = resolve_bf16_precision(device)
    data = CelebATrainingStream(
        data_dir,
        device,
        pipeline=data_pipeline,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return GANRun(
        output_dir=output_dir,
        training_dir=output_dir / "training",
        checkpoint_dir=output_dir / "checkpoints",
        device=device,
        precision=precision,
        data=data,
    )


def initialize_gan_models[
    GeneratorT: nn.Module,
    DiscriminatorT: nn.Module,
](
    generator: GeneratorT,
    discriminator: DiscriminatorT,
    device: torch.device,
) -> tuple[GeneratorT, DiscriminatorT, GeneratorT]:
    """Move both models to the device, optimize 4D state, and create the EMA."""

    def move(module: nn.Module) -> None:
        module.to(device=device)
        module._apply(
            lambda tensor: (
                tensor.contiguous(memory_format=torch.channels_last)
                if tensor.ndim == 4
                else tensor
            )
        )

    move(generator)
    move(discriminator)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    return generator, discriminator, averaged_generator


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


__all__ = [
    "GANRun",
    "append_gan_metrics",
    "initialize_gan_models",
    "prepare_gan_run",
    "start_gan_checkpoint",
]
