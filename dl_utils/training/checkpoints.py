"""Reusable PyTorch checkpoint saving and resumption helpers."""

from __future__ import annotations

import random
import tempfile
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_FORMAT_VERSION = 1

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "TrainingCheckpoint",
    "atomic_torch_save",
    "capture_rng_state",
    "load_model_weights",
    "load_training_checkpoint",
    "make_training_checkpoint",
    "restore_rng_state",
    "save_model_weights",
    "save_periodic_checkpoint",
]


class TrainingCheckpoint:
    """Save and resume one latest full-state training checkpoint."""

    def __init__(self, path, *, unit, models, optimizers):
        self.path = Path(path)
        self.metadata = {"unit": unit}
        self.models = dict(models)
        self.optimizers = dict(optimizers)

    def resume(self, resume_from=None, *, initial_state=None):
        """Return completed units and state, restoring objects when requested."""
        if resume_from is None:
            return 0, dict(initial_state or {})
        checkpoint = load_training_checkpoint(
            resume_from,
            models=self.models,
            optimizers=self.optimizers,
            expected_metadata=self.metadata,
        )
        return checkpoint["epoch"], dict(checkpoint["training_state"])

    def save(self, completed_units, state=None):
        """Replace the latest checkpoint after a phase or data epoch."""
        payload = make_training_checkpoint(
            epoch=completed_units,
            models=self.models,
            optimizers=self.optimizers,
            training_state=state,
            metadata=self.metadata,
        )
        return atomic_torch_save(payload, self.path)


def atomic_torch_save(payload: Any, path: str | PathLike[str]) -> Path:
    """Save through a same-directory temporary file, then replace ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG states."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore random-number generators captured by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"RNG state is missing keys: {missing}.")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())

    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        for device_index, device_state in enumerate(
            cuda_states[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(device_state.cpu(), device_index)


def _validate_named_objects(
    objects: Mapping[str, Any],
    expected_type: type,
    description: str,
) -> None:
    if not objects:
        raise ValueError(f"{description} must not be empty.")
    for name, value in objects.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{description} keys must be non-empty strings.")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{description}[{name!r}] must be a "
                f"{expected_type.__name__}."
            )


def make_training_checkpoint(
    *,
    epoch: int,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a portable epoch-boundary training checkpoint."""
    if epoch < 1:
        raise ValueError("epoch must be positive.")
    _validate_named_objects(models, nn.Module, "models")
    _validate_named_objects(optimizers, Optimizer, "optimizers")
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "models": {
            name: model.state_dict() for name, model in models.items()
        },
        "optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in optimizers.items()
        },
        "training_state": dict(training_state or {}),
        "metadata": dict(metadata or {}),
        "rng_state": capture_rng_state(),
    }


def save_periodic_checkpoint(
    checkpoint_dir: str | PathLike[str],
    *,
    epoch: int,
    total_epochs: int,
    every_epochs: int,
    archive_every_epochs: int | None,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    training_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path | None] | None:
    """Update ``latest.pth`` and optionally archive an epoch checkpoint.

    Set ``archive_every_epochs=None`` to disable archives.  Otherwise the
    final epoch is always archived in addition to the requested interval.
    """
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive.")
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be within [1, total_epochs].")
    if every_epochs < 1:
        raise ValueError("every_epochs must be positive.")
    if archive_every_epochs is not None and archive_every_epochs < 1:
        raise ValueError("archive_every_epochs must be positive or None.")

    should_save = (
        epoch == 1 or epoch % every_epochs == 0 or epoch == total_epochs
    )
    if not should_save:
        return None

    payload = make_training_checkpoint(
        epoch=epoch,
        models=models,
        optimizers=optimizers,
        training_state=training_state,
        metadata=metadata,
    )
    directory = Path(checkpoint_dir)
    latest_path = atomic_torch_save(payload, directory / "latest.pth")

    archive_path = None
    should_archive = archive_every_epochs is not None and (
        epoch == total_epochs or epoch % archive_every_epochs == 0
    )
    if should_archive:
        archive_path = atomic_torch_save(
            payload,
            directory / f"epoch_{epoch:03d}.pth",
        )
    return latest_path, archive_path


def _require_matching_names(
    saved: Mapping[str, Any],
    supplied: Mapping[str, Any],
    description: str,
) -> None:
    saved_names = set(saved)
    supplied_names = set(supplied)
    if saved_names != supplied_names:
        raise ValueError(
            f"Checkpoint {description} do not match supplied {description}; "
            f"saved={sorted(saved_names)}, supplied={sorted(supplied_names)}."
        )


def load_training_checkpoint(
    path: str | PathLike[str],
    *,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    strict: bool = True,
    restore_random_state: bool = True,
) -> dict[str, Any]:
    """Load model/optimizer state and optionally restore all RNG streams."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.")
    _validate_named_objects(models, nn.Module, "models")
    if optimizers is not None:
        _validate_named_objects(optimizers, Optimizer, "optimizers")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Training checkpoint must contain a mapping.")
    if checkpoint.get("checkpoint_format_version") != (
        CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported training checkpoint format: "
            f"{checkpoint.get('checkpoint_format_version')!r}."
        )

    saved_models = checkpoint.get("models")
    saved_optimizers = checkpoint.get("optimizers")
    training_state = checkpoint.get("training_state")
    metadata = checkpoint.get("metadata")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Checkpoint epoch must be a positive integer.")
    if not isinstance(saved_models, Mapping):
        raise ValueError("Checkpoint models must be a mapping.")
    if not isinstance(saved_optimizers, Mapping):
        raise ValueError("Checkpoint optimizers must be a mapping.")
    if not isinstance(training_state, Mapping):
        raise ValueError("Checkpoint training state must be a mapping.")
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    _require_matching_names(saved_models, models, "models")
    if optimizers is not None:
        _require_matching_names(saved_optimizers, optimizers, "optimizers")

    for name, expected in (expected_metadata or {}).items():
        if name not in metadata or metadata[name] != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch for {name!r}: "
                f"expected {expected!r}, found {metadata.get(name)!r}."
            )

    for name, model in models.items():
        model.load_state_dict(saved_models[name], strict=strict)
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(saved_optimizers[name])

    if restore_random_state:
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("Checkpoint RNG state must be a mapping.")
        restore_rng_state(rng_state)
    return checkpoint


def save_model_weights(
    model: nn.Module,
    path: str | PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save one model state dict together with metadata."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module.")
    payload = dict(metadata or {})
    if "state_dict" in payload:
        raise ValueError("metadata must not define the reserved 'state_dict'.")
    payload["state_dict"] = model.state_dict()
    return atomic_torch_save(payload, path)


def load_model_weights(
    path: str | PathLike[str],
    model_class: type[nn.Module],
    *,
    device: str | torch.device,
    expected_metadata: Mapping[str, Any] | None = None,
    config_transform: (
        Callable[[dict[str, Any]], Mapping[str, Any]] | None
    ) = None,
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load and validate one metadata-rich model weight checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}.")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Model checkpoint must contain a mapping.")

    for name, expected in (expected_metadata or {}).items():
        if checkpoint.get(name) != expected:
            raise ValueError(
                f"Model checkpoint metadata mismatch for {name!r}: "
                f"expected {expected!r}, found {checkpoint.get(name)!r}."
            )

    saved_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Model checkpoint model_config must be a mapping.")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Model checkpoint state_dict must be a mapping.")

    model_config = dict(saved_config)
    constructor_config = (
        config_transform(dict(model_config))
        if config_transform is not None
        else model_config
    )
    if not isinstance(constructor_config, Mapping):
        raise TypeError("config_transform must return a mapping.")
    model = model_class(**dict(constructor_config)).to(device)
    model.load_state_dict(state_dict, strict=strict)
    return model.eval(), model_config
