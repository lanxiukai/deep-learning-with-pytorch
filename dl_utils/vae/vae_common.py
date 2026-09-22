"""Gaussian primitives and latent-use diagnostics shared by the VAE lessons."""

from __future__ import annotations

import math

import torch
from torch import Tensor

LOG_2PI = math.log(2.0 * math.pi)


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
    prior_mu: Tensor,
    prior_logvar: Tensor,
    evidence_mu: Tensor,
    evidence_logvar: Tensor,
) -> tuple[Tensor, Tensor]:
    """Normalize the product of two diagonal Gaussians.

    This is the precision-weighted update used by the Ladder VAE posterior.
    """
    if not (
        prior_mu.shape
        == prior_logvar.shape
        == evidence_mu.shape
        == evidence_logvar.shape
    ):
        raise ValueError("all Gaussian parameters must have matching shapes")
    prior_precision = torch.exp(-prior_logvar)
    evidence_precision = torch.exp(-evidence_logvar)
    fused_precision = prior_precision + evidence_precision
    fused_variance = fused_precision.reciprocal()
    fused_mu = fused_variance * (
        prior_precision * prior_mu + evidence_precision * evidence_mu
    )
    return fused_mu, fused_variance.log()


__all__ = [
    "LOG_2PI",
    "ActiveUnitAccumulator",
    "diagonal_gaussian_kl_from_logvar",
    "diagonal_gaussian_log_density",
    "fuse_diagonal_gaussians",
    "reparameterize_logvar",
    "split_gaussian_parameters",
]
