"""Fixed latent grids for comparisons across class-conditional models."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def make_fixed_class_latent_grid(
    class_indices: Sequence[int],
    samples_per_class: int,
    z_dim: int,
    device: torch.device,
    *,
    base_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat the same latent columns for each requested class row."""
    if not class_indices:
        raise ValueError("class_indices must not be empty.")
    if base_noise is None:
        base_noise = torch.randn(samples_per_class, z_dim, device=device)
    elif tuple(base_noise.shape) != (samples_per_class, z_dim):
        raise ValueError(
            "base_noise must have shape "
            f"({samples_per_class}, {z_dim}), got {tuple(base_noise.shape)}."
        )
    else:
        base_noise = base_noise.to(device)

    classes = torch.tensor(class_indices, dtype=torch.long, device=device)
    labels = classes.repeat_interleave(samples_per_class)
    noise = base_noise.repeat(len(class_indices), 1)
    return noise, labels


__all__ = ["make_fixed_class_latent_grid"]
