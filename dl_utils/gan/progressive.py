"""Progressive stages and runtime options shared by ProGAN and StyleGAN."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dl_utils.data.loading import resolve_num_workers


@dataclass(frozen=True)
class ProgressivePhase:
    """One fade-in or stabilization phase at a fixed resolution."""

    resolution: int
    name: str
    batch_size: int
    num_batches: int

    @property
    def num_images(self):
        return self.num_batches * self.batch_size


def build_progressive_schedule(
    *,
    resolutions: Sequence[int],
    batch_sizes: Mapping[int, int],
    phase_kimg: int,
) -> tuple[ProgressivePhase, ...]:
    """Build the fade-in and stabilization phases used by ProGAN."""
    resolutions = tuple(int(value) for value in resolutions)
    phase_kimg = int(phase_kimg)
    if phase_kimg < 1:
        raise ValueError("phase_kimg must be positive.")
    phases = []
    for resolution in resolutions:
        batch_size = int(batch_sizes[resolution])
        target_images = phase_kimg * 1_000
        if batch_size < 1:
            raise ValueError("progressive batch sizes must be positive.")
        names = (
            ("stabilization",)
            if resolution == resolutions[0]
            else ("fade-in", "stabilization")
        )
        phases.extend(
            ProgressivePhase(
                resolution=resolution,
                name=name,
                batch_size=batch_size,
                num_batches=math.ceil(target_images / batch_size),
            )
            for name in names
        )
    return tuple(phases)


def phase_alpha(phase: ProgressivePhase, batch_index: int) -> float:
    """Return the linear fade-in coefficient for one phase batch."""
    if phase.name != "fade-in" or phase.num_batches == 1:
        return 1.0
    return batch_index / (phase.num_batches - 1)


@dataclass(frozen=True)
class ProgressiveGANOptions:
    """Resolved runtime options for a progressive GAN lesson."""

    phase_kimg: int
    batch_sizes: dict[int, int]
    d_reg_every: int
    reg_batch_shrink: int
    num_workers: int
    prefetch_factor: int


def resolve_progressive_gan_options(
    *,
    phase_kimg: int | None,
    batch_scale: int | None,
    d_reg_every: int | None,
    reg_batch_shrink: int | None,
    num_workers: int | None,
    prefetch_factor: int,
    base_batch_sizes: Mapping[int, int],
    default_phase_kimg: int,
    default_d_reg_every: int,
    default_reg_batch_shrink: int,
    default_num_workers: int,
) -> ProgressiveGANOptions:
    """Validate and resolve the shared ProGAN/StyleGAN runtime options."""
    resolved_phase_kimg = default_phase_kimg if phase_kimg is None else phase_kimg
    resolved_batch_scale = 1 if batch_scale is None else batch_scale
    resolved_d_reg_every = default_d_reg_every if d_reg_every is None else d_reg_every
    resolved_reg_batch_shrink = (
        default_reg_batch_shrink if reg_batch_shrink is None else reg_batch_shrink
    )
    if (
        min(
            resolved_phase_kimg,
            resolved_batch_scale,
            resolved_d_reg_every,
            resolved_reg_batch_shrink,
            prefetch_factor,
        )
        < 1
    ):
        raise ValueError("training counts and scales must be positive.")
    batch_sizes = {
        int(resolution): int(batch_size) * resolved_batch_scale
        for resolution, batch_size in base_batch_sizes.items()
    }
    if not batch_sizes or min(batch_sizes.values()) < 1:
        raise ValueError("base_batch_sizes must contain positive values.")
    return ProgressiveGANOptions(
        phase_kimg=resolved_phase_kimg,
        batch_sizes=batch_sizes,
        d_reg_every=resolved_d_reg_every,
        reg_batch_shrink=resolved_reg_batch_shrink,
        num_workers=resolve_num_workers(num_workers, default_num_workers),
        prefetch_factor=prefetch_factor,
    )


__all__ = [
    "ProgressiveGANOptions",
    "ProgressivePhase",
    "build_progressive_schedule",
    "phase_alpha",
    "resolve_progressive_gan_options",
]
