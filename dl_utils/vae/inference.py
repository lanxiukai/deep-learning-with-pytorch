"""Shared 32x32 Gaussian VAE and importance-weighting primitives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    LOG_2PI,
    diagonal_gaussian_log_density,
    split_gaussian_parameters,
)


class GaussianVAE32(nn.Module):
    """Compact Bernoulli VAE with an explicit particle-shaped posterior API."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        hidden_channels: int = 128,
        context_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4, context_dim),
            nn.SiLU(),
        )
        self.base_posterior = nn.Linear(context_dim, 2 * latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 4 * 4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (hidden_channels, 4, 4)),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, 2, 1),
            nn.GroupNorm(4, hidden_channels // 4),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        context = self.context(self.encoder(x))
        mu, logvar = split_gaussian_parameters(self.base_posterior(context))
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        leading_shape = z.shape[:-1]
        flat_z = z.reshape(-1, self.latent_dim)
        images = self.decoder(self.decoder_input(flat_z))
        return images.reshape(*leading_shape, 1, 32, 32)

    def sample_from_statistics(
        self,
        mu: Tensor,
        logvar: Tensor,
        *,
        particles: int,
    ) -> tuple[Tensor, Tensor]:
        epsilon = torch.randn(
            mu.shape[0],
            particles,
            self.latent_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        z = mu[:, None, :] + torch.exp(0.5 * logvar[:, None, :]) * epsilon
        log_q = diagonal_gaussian_log_density(
            z, mu[:, None, :], logvar[:, None, :]
        ).sum(dim=-1)
        return z, log_q

    def reconstruct(self, x: Tensor) -> Tensor:
        mu, _ = self.encode(x)
        return self.decode(mu)

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        return self.decode(torch.randn(count, self.latent_dim, device=device))


def standard_normal_log_density(z: Tensor) -> Tensor:
    """Return log N(z; 0, I), reduced only over the final event dimension."""
    return -0.5 * (LOG_2PI + z.square()).sum(dim=-1)


def bernoulli_log_density(mean: Tensor, target: Tensor) -> Tensor:
    """Return log p(target | mean) for particle-shaped Bernoulli means."""
    expanded_target = target[:, None, ...].expand_as(mean)
    return (
        -F.binary_cross_entropy(mean, expanded_target, reduction="none")
        .flatten(2)
        .sum(dim=-1)
    )


def importance_log_weights(
    model: GaussianVAE32,
    x: Tensor,
    *,
    particles: int,
    particle_chunk_size: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute per-example importance log weights entirely in log space."""
    chunk_size = particles if particle_chunk_size is None else particle_chunk_size
    mu, logvar = model.encode(x)
    log_weight_chunks = []
    log_px_chunks = []
    log_pz_chunks = []
    log_q_chunks = []
    remaining = particles
    while remaining:
        count = min(remaining, chunk_size)
        z, log_q = model.sample_from_statistics(mu, logvar, particles=count)
        reconstruction = model.decode(z)
        log_px = bernoulli_log_density(reconstruction, x)
        log_pz = standard_normal_log_density(z)
        log_weight_chunks.append(log_px + log_pz - log_q)
        log_px_chunks.append(log_px)
        log_pz_chunks.append(log_pz)
        log_q_chunks.append(log_q)
        remaining -= count

    log_weights = torch.cat(log_weight_chunks, dim=1)
    return log_weights, {
        "log_px": torch.cat(log_px_chunks, dim=1),
        "log_pz": torch.cat(log_pz_chunks, dim=1),
        "log_q": torch.cat(log_q_chunks, dim=1),
        "mu": mu,
        "logvar": logvar,
    }


def log_mean_exp(log_weights: Tensor) -> Tensor:
    """Reduce only the particle axis of [batch, particles] log weights."""
    return torch.logsumexp(log_weights, dim=1) - math.log(log_weights.shape[1])


def importance_diagnostics(
    log_weights: Tensor, terms: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Return detached diagnostics without changing the IWAE gradient."""
    normalized = torch.softmax(log_weights, dim=1)
    ess = normalized.square().sum(dim=1).reciprocal()
    return {
        "bound": log_mean_exp(log_weights).mean().detach(),
        "log_px": terms["log_px"].mean().detach(),
        "log_pz": terms["log_pz"].mean().detach(),
        "log_q": terms["log_q"].mean().detach(),
        "ess_fraction": (ess / log_weights.shape[1]).mean().detach(),
        "log_weight_range": (
            log_weights.max(dim=1).values - log_weights.min(dim=1).values
        )
        .mean()
        .detach(),
    }


def model_config(model: GaussianVAE32) -> dict[str, int]:
    return {
        "latent_dim": model.latent_dim,
        "hidden_channels": model.hidden_channels,
        "context_dim": model.context_dim,
    }


__all__ = [
    "GaussianVAE32",
    "bernoulli_log_density",
    "importance_diagnostics",
    "importance_log_weights",
    "log_mean_exp",
    "model_config",
    "standard_normal_log_density",
]
