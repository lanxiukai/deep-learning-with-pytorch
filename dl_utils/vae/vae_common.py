"""Shared RGB networks, Gaussian primitives, and VAE latent-use diagnostics."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import Tensor, nn

LOG_2PI = math.log(2.0 * math.pi)


class ImageEncoder(nn.Sequential):
    """Reduce an RGB image to a flattened 4x4 feature map."""

    def __init__(self, hidden_channels: int) -> None:
        if hidden_channels < 128 or hidden_channels % 128:
            raise ValueError("hidden_channels must be a positive multiple of 128")
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
        super().__init__(*layers)

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[1:] != (3, 256, 256):
            raise ValueError("Expected 256x256 RGB images with shape (B, 3, 256, 256)")
        # (B, 3, 256, 256) -> (B, hidden_channels * 4 * 4)
        return super().forward(images)


class ImageDecoder(nn.Module):
    """Decode (B, latent_dim) into (B, 3, 256, 256) RGB means in [0, 1]."""

    def __init__(self, latent_dim: int, hidden_channels: int) -> None:
        super().__init__()
        if hidden_channels < 128 or hidden_channels % 128:
            raise ValueError("hidden_channels must be a positive multiple of 128")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
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
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent vectors with shape (B, {self.latent_dim})"
            )
        # (B, latent_dim) -> (B, 3, 256, 256)
        return self.net(self.input(z))


class ActiveUnitAccumulator:
    """Count coordinates whose population variance exceeds a threshold.

    Pass a (batch, units) tensor to update(), normally posterior means.
    Hierarchical models can instead pass posterior-minus-prior means for
    a conditional layer. Use one accumulator per layer or statistic; the
    caller chooses its meaning. This measures variance across examples,
    not posterior variance or per-coordinate KL.
    """

    def __init__(self) -> None:
        self.num_examples = 0
        self._sum: Tensor | None = None
        self._square_sum: Tensor | None = None

    def update(self, values: Tensor) -> None:
        """Accumulate detached float64 moments without storing all examples."""
        if values.ndim != 2:
            raise ValueError("values must have shape (batch, units)")
        if self._sum is not None and values.shape[1:] != self._sum.shape:
            raise ValueError("the number of units must stay constant across batches")
        if values.shape[0] == 0:
            return
        values = values.detach().double()
        batch_sum = values.sum(dim=0)
        batch_square_sum = values.square().sum(dim=0)
        if self._sum is None:
            self._sum = batch_sum
            self._square_sum = batch_square_sum
        else:
            assert self._square_sum is not None
            self._sum += batch_sum
            self._square_sum += batch_square_sum
        self.num_examples += values.shape[0]

    def count(self, *, variance_threshold: float = 1e-2) -> int:
        """Return the active count; fewer than two examples yield zero."""
        if self.num_examples < 2:
            return 0
        assert self._sum is not None and self._square_sum is not None
        variance = (
            self._square_sum / self.num_examples
            - (self._sum / self.num_examples).square()
        ).clamp_min(0.0)
        return int((variance > variance_threshold).sum().item())


def split_gaussian_parameters(
    raw: Tensor, *, dimension: int = 1, minimum: float = -12.0, maximum: float = 12.0
) -> tuple[Tensor, Tensor]:
    """Split a tensor into mean/log-variance and bound the exponential range."""
    # raw: (B, 2 * latent_dim)
    mu, logvar = raw.chunk(2, dim=dimension)
    return mu, logvar.clamp(minimum, maximum)  # (B, latent_dim)


def reparameterize_logvar(
    mu: Tensor, logvar: Tensor, *, noise: Tensor | None = None
) -> Tensor:
    """Draw ``N(mu, exp(logvar))`` using fresh or caller-supplied base noise."""
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have matching shapes")
    if noise is None:
        noise = torch.randn_like(mu)
    elif noise.shape != mu.shape:
        raise ValueError("noise must have the same shape as mu")
    return mu + torch.exp(0.5 * logvar) * noise  # (B, latent_dim)


def diagonal_gaussian_kl_from_logvar(
    q_mu: Tensor,
    q_logvar: Tensor,
    p_mu: Tensor | None = None,
    p_logvar: Tensor | None = None,
) -> Tensor:
    """Return elementwise ``KL(q || p)`` for diagonal Gaussian distributions.

    When ``p_mu`` and ``p_logvar`` are omitted, ``p`` is ``N(0, I)``.  No
    dimensions are reduced so each lesson can make its own per-sample and
    per-layer reduction explicit.
    """
    if (p_mu is None) != (p_logvar is None):
        raise ValueError("p_mu and p_logvar must both be provided or omitted")
    if p_mu is None:
        return 0.5 * (q_mu.square() + q_logvar.exp() - 1.0 - q_logvar)
    assert p_logvar is not None
    return 0.5 * (
        p_logvar - q_logvar
        + torch.exp(q_logvar - p_logvar)
        + (q_mu - p_mu).square() * torch.exp(-p_logvar)
        - 1.0
    )  # (B, latent_dim)


def diagonal_gaussian_log_density(
    sample: Tensor, mu: Tensor, logvar: Tensor
) -> Tensor:
    """Return elementwise ``log N(sample; mu, exp(logvar))`` with broadcasting."""
    return -0.5 * (
        LOG_2PI + logvar + (sample - mu).square() * torch.exp(-logvar)
    )


def fuse_diagonal_gaussians(
    mu_a: Tensor,
    logvar_a: Tensor,
    mu_b: Tensor,
    logvar_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the normalized product of two diagonal Gaussians.

    Each Gaussian is parameterized by its mean and log-variance. The returned
    mean is precision-weighted; the second result is the fused log-variance.
    """
    if not (
        mu_a.shape
        == logvar_a.shape
        == mu_b.shape
        == logvar_b.shape
    ):
        raise ValueError("all Gaussian parameters must have matching shapes")
    precision_a = torch.exp(-logvar_a)
    precision_b = torch.exp(-logvar_b)
    fused_precision = precision_a + precision_b
    fused_variance = fused_precision.reciprocal()
    fused_mu = fused_variance * (
        precision_a * mu_a + precision_b * mu_b
    )
    return fused_mu, fused_variance.log()


__all__ = [
    "LOG_2PI",
    "ActiveUnitAccumulator",
    "ImageDecoder",
    "ImageEncoder",
    "diagonal_gaussian_kl_from_logvar",
    "diagonal_gaussian_log_density",
    "fuse_diagonal_gaussians",
    "reparameterize_logvar",
    "split_gaussian_parameters",
]
