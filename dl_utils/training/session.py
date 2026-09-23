"""Training-run lifecycle helpers that keep optimization loops explicit."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import Optimizer

from dl_utils.filesystem.directories import reset_dir
from dl_utils.training.checkpoints import (
    load_training_checkpoint,
    save_model_weights,
    save_periodic_checkpoint,
)

__all__ = ["TrainingSession"]


class TrainingSession:
    """Manage output, recovery, and artifacts without owning training steps.

    Call :meth:`checkpoint` once after every completed epoch.  The configured
    interval controls whether that call writes a file; keeping the call in the
    lesson's epoch loop makes the training algorithm remain visible.
    """

    def __init__(
        self,
        output_dir: str | PathLike[str],
        *,
        total_epochs: int,
        models: Mapping[str, nn.Module],
        optimizers: Mapping[str, Optimizer],
        checkpoint_every_epochs: int,
        archive_every_epochs: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        model_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive.")
        if checkpoint_every_epochs < 1:
            raise ValueError("checkpoint_every_epochs must be positive.")
        if archive_every_epochs is not None and archive_every_epochs < 1:
            raise ValueError("archive_every_epochs must be positive or None.")
        if not models:
            raise ValueError("models must not be empty.")
        if not optimizers:
            raise ValueError("optimizers must not be empty.")

        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.total_epochs = total_epochs
        self.models = dict(models)
        self.optimizers = dict(optimizers)
        self.checkpoint_every_epochs = checkpoint_every_epochs
        self.archive_every_epochs = archive_every_epochs
        self.metadata = dict(metadata or {})
        self.model_metadata = {
            name: dict(values) for name, values in (model_metadata or {}).items()
        }
        unknown_models = sorted(set(self.model_metadata) - set(self.models))
        if unknown_models:
            raise ValueError(
                f"model_metadata contains unknown models: {unknown_models}."
            )

        self._checkpoint_metadata = {
            "total_epochs": total_epochs,
            "run": self.metadata,
            "models": self.model_metadata,
        }
        self._started = False
        self._last_epoch = 0

    def start(
        self,
        resume_from: str | PathLike[str] | None = None,
        *,
        initial_state: Mapping[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Prepare output and restore the same recipe, optionally for more epochs."""
        if self._started:
            raise RuntimeError("TrainingSession.start() may only be called once.")

        resume_path = Path(resume_from) if resume_from is not None else None
        if resume_path is not None and not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}.")

        if resume_path is None:
            reset_dir(str(self.output_dir))
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        default_state = dict(initial_state or {})
        if resume_path is None:
            self._started = True
            return 1, default_state

        checkpoint = load_training_checkpoint(
            resume_path,
            models=self.models,
            optimizers=self.optimizers,
            expected_metadata={
                "run": self.metadata,
                "models": self.model_metadata,
            },
        )
        if self.total_epochs < checkpoint["metadata"]["total_epochs"]:
            raise ValueError(
                "A resumed run may extend, but not shorten, its epoch budget."
            )
        resumed_epoch = checkpoint["epoch"]
        if resumed_epoch > self.total_epochs:
            raise ValueError("Checkpoint epoch exceeds the configured total epochs.")
        saved_state = dict(checkpoint["training_state"])
        if initial_state is not None and set(saved_state) != set(default_state):
            raise ValueError(
                "Checkpoint training-state keys do not match this run; "
                f"saved={sorted(saved_state)}, "
                f"expected={sorted(default_state)}."
            )

        self._started = True
        self._last_epoch = resumed_epoch
        return resumed_epoch + 1, saved_state

    def checkpoint(
        self,
        epoch: int,
        training_state: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path | None] | None:
        """Record one completed epoch and save it when the schedule fires."""
        self._require_started()
        expected_epoch = self._last_epoch + 1
        if epoch != expected_epoch:
            raise ValueError(f"Expected completed epoch {expected_epoch}, got {epoch}.")
        result = save_periodic_checkpoint(
            self.checkpoint_dir,
            epoch=epoch,
            total_epochs=self.total_epochs,
            every_epochs=self.checkpoint_every_epochs,
            archive_every_epochs=self.archive_every_epochs,
            models=self.models,
            optimizers=self.optimizers,
            training_state=training_state,
            metadata=self._checkpoint_metadata,
        )
        self._last_epoch = epoch
        return result

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Save one final metadata-rich weight file per registered model."""
        self._require_started()
        if self._last_epoch != self.total_epochs:
            raise RuntimeError(
                "Cannot save final weights before all configured epochs have completed."
            )

        shared_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "epoch": self.total_epochs,
        }
        paths = {}
        for name, model in self.models.items():
            paths[name] = save_model_weights(
                model,
                self.output_dir / f"{name}.pth",
                metadata={
                    **shared_metadata,
                    **self.model_metadata.get(name, {}),
                    "epoch": self.total_epochs,
                },
            )
        return paths

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Call TrainingSession.start() first.")
