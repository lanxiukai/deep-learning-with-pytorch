"""Exponential moving averages shared by model families."""

import torch
from torch import nn


def ema_decay_for_batch(batch_size: int, half_life_kimg: float) -> float:
    """Convert an image-count half-life into one batch's EMA decay."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if half_life_kimg <= 0:
        raise ValueError("half_life_kimg must be positive.")
    return 0.5 ** (batch_size / (half_life_kimg * 1_000))


@torch.no_grad()
def update_ema(
    target: nn.Module,
    source: nn.Module,
    decay: float,
    *,
    copy_buffers: bool = True,
) -> None:
    """Move target parameters toward source and optionally copy buffers."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be within [0, 1).")
    for target_parameter, source_parameter in zip(
        target.parameters(),
        source.parameters(),
        strict=True,
    ):
        # decay * target_parameter + (1 - decay) * source_parameter
        target_parameter.lerp_(source_parameter, 1 - decay)
    # parameters() contains only trainable parameters optimized during training,
    # whereas buffers() contains non-trainable module state that still affects
    # the forward pass.
    if copy_buffers:
        for target_buffer, source_buffer in zip(
            target.buffers(),
            source.buffers(),
            strict=True,
        ):
            target_buffer.copy_(source_buffer)


def update_ema_by_images(
    target: nn.Module,
    source: nn.Module,
    batch_size: int,
    half_life_kimg: float,
    *,
    copy_buffers: bool = True,
) -> None:
    """Update an EMA using a half-life expressed in thousands of images."""
    update_ema(
        target,
        source,
        ema_decay_for_batch(batch_size, half_life_kimg),
        copy_buffers=copy_buffers,
    )


__all__ = ["ema_decay_for_batch", "update_ema", "update_ema_by_images"]
