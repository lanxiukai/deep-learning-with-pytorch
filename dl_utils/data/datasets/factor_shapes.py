"""Small procedurally controlled 32x32 shape dataset for representation lessons."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

FACTOR_NAMES = ("shape", "scale", "rotation", "x_position", "y_position")
FACTOR_SIZES = (3, 4, 6, 7, 7)


def all_factor_combinations() -> Tensor:
    """Return the complete independent Cartesian product of factor indices."""
    values = [torch.arange(size) for size in FACTOR_SIZES]
    return torch.cartesian_prod(*values).to(dtype=torch.long)


def render_factor_shapes(factors: Tensor, *, supersample: int = 2) -> Tensor:
    """Render factor rows as antialiased grayscale images in [0, 1]."""
    if factors.ndim != 2 or factors.shape[1] != len(FACTOR_SIZES):
        raise ValueError("factors must have shape [batch, 5]")
    if supersample < 1:
        raise ValueError("supersample must be positive")
    for column, size in enumerate(FACTOR_SIZES):
        if factors.numel() and (
            int(factors[:, column].min()) < 0 or int(factors[:, column].max()) >= size
        ):
            raise ValueError(f"factor column {column} is out of range")

    dtype = torch.float32
    side = 32 * supersample
    coordinates = (torch.arange(side, dtype=dtype) + 0.5) / supersample - 16.0
    y_grid, x_grid = torch.meshgrid(coordinates, coordinates, indexing="ij")
    scale_values = torch.tensor([3.0, 4.0, 5.0, 6.0])
    scale = scale_values[factors[:, 1]][:, None, None]
    angle = (factors[:, 2].to(dtype) * (math.pi / FACTOR_SIZES[2]))[:, None, None]
    center_x = ((factors[:, 3].to(dtype) - 3.0) * 2.0)[:, None, None]
    center_y = ((factors[:, 4].to(dtype) - 3.0) * 2.0)[:, None, None]
    dx = x_grid[None, :, :] - center_x
    dy = y_grid[None, :, :] - center_y
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    horizontal = cosine * dx + sine * dy
    vertical = -sine * dx + cosine * dy

    ellipse = (horizontal / (1.25 * scale)).square() + (
        vertical / (0.65 * scale)
    ).square() <= 1.0
    rectangle = (horizontal.abs() <= 1.15 * scale) & (vertical.abs() <= 0.55 * scale)
    triangle = (
        (horizontal >= -1.15 * scale)
        & (horizontal <= 1.15 * scale)
        & (vertical.abs() <= 0.65 * (1.15 * scale - horizontal))
    )
    shapes = torch.stack((ellipse, rectangle, triangle), dim=1)
    selector = factors[:, 0].reshape(-1, 1, 1, 1)
    mask = shapes.gather(1, selector.expand(-1, 1, side, side)).to(dtype)
    if supersample > 1:
        mask = F.avg_pool2d(mask, supersample, supersample)
    return mask


class FactorShapes32(Dataset[tuple[Tensor, Tensor]]):
    """Complete factor grid with a deterministic train/test combination split."""

    factor_names: Sequence[str] = FACTOR_NAMES
    factor_sizes: Sequence[int] = FACTOR_SIZES

    def __init__(
        self,
        *,
        split: str,
        train_fraction: float = 0.8,
        split_seed: int = 2026,
        render_batch_size: int = 256,
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be 'train', 'test', or 'all'")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        if render_batch_size < 1:
            raise ValueError("render_batch_size must be positive")

        factors = all_factor_combinations()
        if split != "all":
            generator = torch.Generator().manual_seed(split_seed)
            permutation = torch.randperm(len(factors), generator=generator)
            split_index = round(train_fraction * len(factors))
            indices = (
                permutation[:split_index]
                if split == "train"
                else permutation[split_index:]
            )
            factors = factors[indices]
        self.factors = factors
        self.images = torch.cat(
            [
                render_factor_shapes(factors[start : start + render_batch_size])
                for start in range(0, len(factors), render_batch_size)
            ]
        )

    def __len__(self) -> int:
        return len(self.factors)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.factors[index]


__all__ = [
    "FACTOR_NAMES",
    "FACTOR_SIZES",
    "FactorShapes32",
    "all_factor_combinations",
    "render_factor_shapes",
]
