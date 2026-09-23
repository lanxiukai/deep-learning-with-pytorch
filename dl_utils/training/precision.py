"""FP32 and BF16 training contexts and optimizer-step policies."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BF16Precision:
    """Own the BF16 autocast and optimizer-step behavior."""

    device: torch.device

    @property
    def name(self) -> str:
        return "bf16"

    def autocast(self):
        """Return a fresh forward-pass autocast context."""
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

    def backward_step(self, loss, optimizer) -> None:
        """Finish one CUDA backward pass before updating its parameters."""
        loss.backward()
        # PyTorch 2.13 + CUDA 13 needs an explicit completion boundary here for
        # stable BF16 GAN updates that include higher-order regularization.
        torch.cuda.synchronize(self.device)
        optimizer.step()


@dataclass(frozen=True)
class FP32Precision:
    """Own the ordinary FP32 forward and optimizer-step behavior."""

    device: torch.device

    @property
    def name(self) -> str:
        return "fp32"

    def autocast(self):
        """Return a no-op context matching the mixed-precision interface."""
        return nullcontext()

    def backward_step(self, loss, optimizer) -> None:
        """Backpropagate and update parameters in FP32."""
        loss.backward()
        optimizer.step()


def resolve_bf16_precision(device: torch.device) -> BF16Precision:
    """Require CUDA BF16 support and construct the training context."""
    if device.type != "cuda":
        raise ValueError("BF16 training requires a CUDA device.")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("the selected CUDA device does not support BF16.")
    return BF16Precision(device)


def resolve_training_precision(
    device: torch.device,
    requested: str = "auto",
) -> BF16Precision | FP32Precision:
    """Resolve ``auto``, ``bf16``, or ``fp32`` for one training run."""
    if requested not in {"auto", "bf16", "fp32"}:
        raise ValueError("requested precision must be auto, bf16, or fp32.")
    if requested == "fp32":
        return FP32Precision(device)
    if requested == "bf16":
        return resolve_bf16_precision(device)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return BF16Precision(device)
    return FP32Precision(device)


__all__ = [
    "BF16Precision",
    "FP32Precision",
    "resolve_bf16_precision",
    "resolve_training_precision",
]
