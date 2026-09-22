"""Shared 256x256 RGB backbone for the conditional and hierarchical VAEs."""

from __future__ import annotations

from itertools import pairwise

from torch import Tensor, nn


class ImageEncoder(nn.Sequential):
    """Reduce an RGB image to a flattened 4x4 feature map."""

    def __init__(self, hidden_channels: int) -> None:
        channels = (
            3,
            hidden_channels // 8,
            hidden_channels // 4,
            hidden_channels // 2,
            hidden_channels,
            hidden_channels,
            hidden_channels,
        )
        layers: list[nn.Module] = []
        for in_channels, out_channels in pairwise(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            if in_channels != 3:
                layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.SiLU())
        layers.append(nn.Flatten())
        super().__init__(*layers)  # (B, 3, 256, 256) -> (B, hidden_channels * 4 * 4)


class ImageDecoder(nn.Module):
    """Decode a vector into an RGB Gaussian mean in [0, 1]."""

    def __init__(self, latent_dim: int, hidden_channels: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 4 * 4),
            nn.SiLU(),
        )
        channels = (
            hidden_channels,
            hidden_channels,
            hidden_channels,
            hidden_channels // 2,
            hidden_channels // 4,
            hidden_channels // 8,
            3,
        )
        layers: list[nn.Module] = [nn.Unflatten(1, (hidden_channels, 4, 4))]
        for in_channels, out_channels in pairwise(channels):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            if out_channels != 3:
                layers.extend((nn.GroupNorm(8, out_channels), nn.SiLU()))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)  # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256

    def forward(self, z: Tensor) -> Tensor:
        return self.net(self.input(z))  # (B, latent_dim) -> (B, 3, 256, 256)
