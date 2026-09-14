"""Shared 32x32 Gaussian VAE and importance-weighting primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.vae_common import (
    LOG_2PI,
    diagonal_gaussian_log_density,
    split_gaussian_parameters,
)


class GaussianVAE(nn.Module):
    """Compact Bernoulli VAE with an explicit particle-shaped posterior API."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
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
            nn.Linear(hidden_channels * 4 * 4, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * latent_dim),
        )  # (B, 1, 32, 32) -> (B, 2 * latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 4 * 4),
            nn.SiLU(),
            nn.Unflatten(1, (hidden_channels, 4, 4)),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, 2, 1),
            nn.GroupNorm(4, hidden_channels // 4),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )  # (B, latent_dim) -> (B, 1, 32, 32)

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = split_gaussian_parameters(self.encoder(images))
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        # z: (B, latent_dim) for VAE or (B, K, latent_dim) for IWAE
        leading_shape = z.shape[:-1]
        flat_z = z.reshape(-1, self.latent_dim)
        images = self.decoder(flat_z)
        # output: (B, 1, 32, 32) for VAE or (B, K, 1, 32, 32) for IWAE
        return images.reshape(*leading_shape, 1, 32, 32)

    def sample_from_statistics(
        self,
        mu: Tensor,
        logvar: Tensor,
        *,
        particles: int,
    ) -> tuple[Tensor, Tensor]:
        # epsilon ~ N(0, I): (B, K, latent_dim), with K independent samples per input.
        epsilon = torch.randn(
            mu.shape[0],
            particles,
            self.latent_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        # z: (B, K, latent_dim)
        z = mu[:, None, :] + torch.exp(0.5 * logvar[:, None, :]) * epsilon
        log_q = diagonal_gaussian_log_density(
            z, mu[:, None, :], logvar[:, None, :]
        ).sum(dim=-1)  # (B, K)
        return z, log_q

    def reconstruct(self, images: Tensor) -> Tensor:
        mu, _ = self.encode(images)
        return self.decode(mu)

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        return self.decode(torch.randn(count, self.latent_dim, device=device))


def standard_normal_log_density(z: Tensor) -> Tensor:
    """Return log N(z; 0, I), reduced only over the final event dimension."""
    return -0.5 * (LOG_2PI + z.square()).sum(dim=-1)


def bernoulli_log_density(mean: Tensor, real_images: Tensor) -> Tensor:
    """Return log p(real_images | mean) for particle-shaped Bernoulli means."""
    expanded_real_images = real_images[:, None, ...].expand_as(mean)  # (B, K, C, H, W)
    return (
        -F.binary_cross_entropy(mean, expanded_real_images, reduction="none")
        .flatten(2)
        .sum(dim=-1)
    )  # (B, K)


def importance_statistics(
    model: GaussianVAE,
    images: Tensor,
    *,
    particles: int,
    particle_chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Return the batch IWAE loss and detached monitoring metrics in log space."""
    chunk_size = particles if particle_chunk_size is None else particle_chunk_size
    mu, logvar = model.encode(images)
    log_weight_chunks = []
    log_px_chunks = []
    log_pz_chunks = []
    log_q_chunks = []
    remaining = particles
    while remaining:
        count = min(remaining, chunk_size)
        z, log_q = model.sample_from_statistics(mu, logvar, particles=count)
        reconstruction = model.decode(z)
        log_px = bernoulli_log_density(reconstruction, images)
        log_pz = standard_normal_log_density(z)
        log_weight_chunks.append(log_px + log_pz - log_q)
        log_px_chunks.append(log_px)
        log_pz_chunks.append(log_pz)
        log_q_chunks.append(log_q)
        remaining -= count

    log_weights = torch.cat(log_weight_chunks, dim=1)
    log_px = torch.cat(log_px_chunks, dim=1)
    log_pz = torch.cat(log_pz_chunks, dim=1)
    log_q = torch.cat(log_q_chunks, dim=1)
    normalized = torch.softmax(log_weights, dim=1)
    ess_fraction = normalized.square().sum(dim=1).reciprocal() / particles
    loss = -log_mean_exp(log_weights).mean()
    return {
        "loss": loss,
        "reconstruction_loss": (-log_px.mean()).detach(),
        "kl_loss": (log_q - log_pz).mean().detach(),
        "ess_fraction": ess_fraction.mean().detach(),
    }


@torch.inference_mode()
def evaluate_iwae(
    model: GaussianVAE,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    particles: int,
    particle_chunk_size: int | None,
    max_examples: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate batch IWAE metrics with a bounded number of examples."""
    model.eval()
    accumulator = MetricAccumulator(
        ("loss", "reconstruction_loss", "kl_loss", "ess_fraction"),
        device=device,
    )
    examples = 0
    for images, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        metrics = importance_statistics(
            model,
            images,
            particles=particles,
            particle_chunk_size=particle_chunk_size,
        )
        accumulator.update(
            (
                metrics["loss"],
                metrics["reconstruction_loss"],
                metrics["kl_loss"],
                metrics["ess_fraction"],
            ),
            num_examples=images.shape[0],
        )
        examples += images.shape[0]
    return accumulator.compute()


def log_mean_exp(log_weights: Tensor) -> Tensor:
    """Return each example's single K-particle IWAE estimate.

    This is a Monte Carlo estimate of the importance-weighted ELBO.
    """
    # log_weights: (B, K)
    # output: (B,)
    return torch.logsumexp(log_weights, dim=1) - math.log(log_weights.shape[1])


__all__ = [
    "GaussianVAE",
    "bernoulli_log_density",
    "evaluate_iwae",
    "importance_statistics",
    "log_mean_exp",
    "standard_normal_log_density",
]
