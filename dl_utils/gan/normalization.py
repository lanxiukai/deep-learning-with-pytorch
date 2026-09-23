"""Normalization-buffer calibration for conditional GAN generators."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def refresh_generator_statistics(
    generator: nn.Module,
    *,
    z_dim: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 64,
    num_batches: int = 32,
) -> None:
    """Refresh an EMA generator's BatchNorm and spectral-norm buffers.

    Averaged parameters can disagree with buffers copied from the live model.
    Forward-only calibration updates those buffers without changing parameters
    or consuming the training loop's random-number stream.
    """
    random_generator = torch.Generator(device=device).manual_seed(771)
    was_training = generator.training
    generator.train()
    try:
        for _ in range(num_batches):
            noise = torch.randn(
                batch_size, z_dim, device=device, generator=random_generator
            )
            labels = torch.randint(
                num_classes,
                (batch_size,),
                device=device,
                generator=random_generator,
            )
            generator(noise, labels)
    finally:
        generator.train(was_training)


__all__ = ["refresh_generator_statistics"]
