"""Fixed-latent sample grids for BF16 GAN lesson runs."""

from __future__ import annotations

import os
from collections.abc import Callable
from os import PathLike

import torch
from torch import nn

from dl_utils.gan.stylegan_layers import denormalize
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot.images import save_grid
from dl_utils.training.accelerator import BF16Precision


def save_gan_samples(
    generator: nn.Module,
    fixed_z: torch.Tensor,
    output_path: str | PathLike[str],
    *,
    precision: BF16Precision,
    generate: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    nrow: int,
    title: str,
) -> None:
    """Render a fixed-latent EMA sample grid through BF16 inference."""

    def generate_bf16(z_batch: torch.Tensor) -> torch.Tensor:
        with precision.autocast():
            return generate(z_batch).float()

    samples = generate_in_batches(
        fixed_z,
        batch_size,
        generate_bf16,
        module=generator,
    )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("GAN sample generation produced non-finite pixels.")
    save_grid(
        denormalize(samples),
        os.fspath(output_path),
        nrow=nrow,
        title=title,
    )


__all__ = ["save_gan_samples"]
