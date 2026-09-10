"""Compact conditional latent-variable models shared by the cVAE lessons."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    reparameterize_logvar,
    split_gaussian_parameters,
)


class ConditionalDecoder(nn.Module):
    """Decode a latent and class representation into a Bernoulli mean."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_channels * 4 * 4),
            nn.SiLU(),
        )  # (B, latent_dim + condition_dim) -> (B, hidden_channels * 4 * 4)
        self.net = nn.Sequential(
            nn.Unflatten(1, (hidden_channels, 4, 4)),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, 4, 2, 1),
            nn.GroupNorm(4, hidden_channels // 4),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )  # (B, hidden_channels, 4, 4) -> (B, 1, 32, 32)

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        # z: (B, latent_dim), condition: (B, condition_dim)
        features = torch.cat((z, condition), dim=1)
        return self.net(self.input(features))  # (B, 1, 32, 32)


class ConditionalVAE(nn.Module):
    """Class-conditional VAE with explicit prior, posterior, and decoder APIs."""

    def __init__(
        self,
        *,
        num_classes: int = 10,
        latent_dim: int = 16,
        condition_dim: int = 32,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels

        # The posterior, conditional prior, and decoder share the class embedding.
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 4, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Flatten(),
        )  # (B, 1, 32, 32) -> (B, hidden_channels * 4 * 4)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4 + condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )  # (B, hidden_channels * 4 * 4 + condition_dim) -> (B, 2 * latent_dim)
        self.prior_network = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )  # (B, condition_dim) -> (B, 2 * latent_dim)
        self.decoder = ConditionalDecoder(
            latent_dim, condition_dim, hidden_channels
        )

    def prior(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return p(z | c) parameters predicted from the class condition."""
        condition = self.condition_embedding(labels)  # (B, condition_dim)
        return split_gaussian_parameters(
            self.prior_network(condition)
        )  # p_mu, p_logvar: (B, latent_dim)

    def encode(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return q(z | x, c) parameters; this is the only target-aware API."""
        # images: (B, 1, 32, 32), labels: (B,)
        condition = self.condition_embedding(labels)  # (B, condition_dim)
        features = self.image_encoder(images)  # (B, hidden_channels * 4 * 4)
        return split_gaussian_parameters(
            self.posterior(torch.cat((features, condition), dim=1))
        )  # q_mu, q_logvar (B, latent_dim)

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        """Return the Bernoulli mean p(x | z, c)."""
        # z: (B, latent_dim), labels: (B,)
        condition = self.condition_embedding(labels)  # (B, condition_dim)
        return self.decoder(z, condition)  # (B, 1, 32, 32)

    def generate(self, labels: Tensor) -> Tensor:
        """Sample z from p(z | c), then decode it with the requested class."""
        # The prior p(z | c) is the latent reference for the KL term and generation.
        p_mu, p_logvar = self.prior(labels)
        z = reparameterize_logvar(p_mu, p_logvar)  # (B, latent_dim)
        return self.decode(z, labels)  # (B, 1, 32, 32)

    def forward(self, images: Tensor, labels: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        # images: (B, 1, 32, 32), labels: (B,)
        # mu, logvar: (B, latent_dim)
        q_mu, q_logvar = self.encode(images, labels)
        p_mu, p_logvar = self.prior(labels)
        z = reparameterize_logvar(q_mu, q_logvar)  # (B, latent_dim)
        reconstruction = self.decode(z, labels)    # (B, 1, 32, 32)
        return reconstruction, {
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
        }  # reconstruction, statistics


__all__ = [
    "ConditionalVAE",
]
