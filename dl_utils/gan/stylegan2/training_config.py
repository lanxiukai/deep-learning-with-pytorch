"""Fixed-resolution image budgets and runtime options for StyleGAN2."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dl_utils.data.loading import resolve_num_workers


@dataclass(frozen=True)
class FixedResolutionGANOptions:
    """Resolved runtime options for the fixed-resolution GAN lesson."""

    total_kimg: int
    batch_size: int
    r1_batch_shrink: int
    path_batch_shrink: int
    num_workers: int
    prefetch_factor: int


def resolve_fixed_resolution_gan_options(
    *,
    total_kimg: int | None,
    batch_scale: int | None,
    r1_batch_shrink: int | None,
    path_batch_shrink: int | None,
    num_workers: int | None,
    prefetch_factor: int,
    base_batch_size: int,
    default_total_kimg: int,
    default_r1_batch_shrink: int,
    default_path_batch_shrink: int,
    default_num_workers: int,
) -> FixedResolutionGANOptions:
    """Validate and resolve the shared StyleGAN2 runtime options."""
    resolved_total_kimg = default_total_kimg if total_kimg is None else total_kimg
    resolved_batch_scale = 1 if batch_scale is None else batch_scale
    resolved_r1_shrink = (
        default_r1_batch_shrink if r1_batch_shrink is None else r1_batch_shrink
    )
    resolved_path_shrink = (
        default_path_batch_shrink if path_batch_shrink is None else path_batch_shrink
    )
    if (
        min(
            resolved_total_kimg,
            resolved_batch_scale,
            resolved_r1_shrink,
            resolved_path_shrink,
            base_batch_size,
            prefetch_factor,
        )
        < 1
    ):
        raise ValueError("training counts and scales must be positive.")
    return FixedResolutionGANOptions(
        total_kimg=resolved_total_kimg,
        batch_size=base_batch_size * resolved_batch_scale,
        r1_batch_shrink=resolved_r1_shrink,
        path_batch_shrink=resolved_path_shrink,
        num_workers=resolve_num_workers(num_workers, default_num_workers),
        prefetch_factor=prefetch_factor,
    )


@dataclass(frozen=True)
class TrainingEpoch:
    """One data epoch in a fixed-resolution image budget."""

    num_batches: int
    num_images: int
    final_batch_size: int

    def batch_size_at(self, batch_index, batch_size):
        if not 0 <= batch_index < self.num_batches:
            raise ValueError("batch_index is outside this training epoch.")
        if batch_index == self.num_batches - 1:
            return self.final_batch_size
        return batch_size


def build_training_schedule(total_kimg, batch_size, dataset_size):
    """Build data epochs with an exact final image and batch count."""
    if min(total_kimg, batch_size, dataset_size) < 1:
        raise ValueError("total_kimg, batch_size, and dataset_size must be positive.")
    if dataset_size < batch_size:
        raise ValueError("dataset_size must contain at least one complete batch.")
    total_images = total_kimg * 1_000
    total_batches = math.ceil(total_images / batch_size)
    final_training_batch = total_images - batch_size * (total_batches - 1)
    batches_per_epoch = dataset_size // batch_size
    schedule = []
    for batch_start in range(0, total_batches, batches_per_epoch):
        num_batches = min(batches_per_epoch, total_batches - batch_start)
        is_last = batch_start + num_batches == total_batches
        final_batch_size = final_training_batch if is_last else batch_size
        num_images = num_batches * batch_size
        if is_last:
            num_images -= batch_size - final_training_batch
        schedule.append(
            TrainingEpoch(
                num_batches=num_batches,
                num_images=num_images,
                final_batch_size=final_batch_size,
            )
        )
    return tuple(schedule)


__all__ = [
    "FixedResolutionGANOptions",
    "TrainingEpoch",
    "build_training_schedule",
    "resolve_fixed_resolution_gan_options",
]
