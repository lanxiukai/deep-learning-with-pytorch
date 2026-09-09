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
        # Typical shapes: z: (B, latent_dim), condition: (B, condition_dim).
        # For S latent samples per example, use z: (S, B, latent_dim) and
        # condition: (S, B, condition_dim).
        # Preserve all leading dimensions so they can be restored after decoding.
        leading_shape = z.shape[:-1]
        features = torch.cat((z, condition), dim=-1).reshape(
            -1, z.shape[-1] + condition.shape[-1]
        )
        images = self.net(self.input(features))
        return images.reshape(*leading_shape, 1, 32, 32)


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

        # The posterior and decoder share the class embedding.
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
        )  # (B, 1, 32, 32) -> (B, hidden_channels, 4, 4)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4 + condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )
        self.decoder = ConditionalDecoder(latent_dim, condition_dim, hidden_channels)

    def condition(self, labels: Tensor) -> Tensor:
        return self.condition_embedding(labels)

    def prior(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return the standard normal prior, independent of the class label."""
        zeros = self.condition_embedding.weight.new_zeros(
            labels.shape[0], self.latent_dim
        )
        return zeros, zeros

    def encode(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return q(z | x, c) parameters; this is the only target-aware API."""
        condition = self.condition_embedding(labels)
        features = self.image_encoder(x)
        return split_gaussian_parameters(
            self.posterior(torch.cat((features, condition), dim=1))
        )

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        """Return the Bernoulli mean p(x | z, c)."""
        flat_labels = labels.reshape(-1)
        condition = self.condition_embedding(flat_labels).reshape(
            *labels.shape, self.condition_dim
        )
        return self.decoder(z, condition)

    def reconstruct(self, x: Tensor, labels: Tensor, *, sample: bool = False) -> Tensor:
        q_mu, q_logvar = self.encode(x, labels)
        z = reparameterize_logvar(q_mu, q_logvar) if sample else q_mu
        return self.decode(z, labels)

    def generate(self, labels: Tensor) -> Tensor:
        """Sample z ~ N(0, I), then decode it with the requested class."""
        z = torch.randn(
            labels.shape[0],
            self.latent_dim,
            device=labels.device,
        )
        return self.decode(z, labels)

    def forward(self, x: Tensor, labels: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        q_mu, q_logvar = self.encode(x, labels)
        p_mu, p_logvar = self.prior(labels)
        z = reparameterize_logvar(q_mu, q_logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, {
            "z": z,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
        }


__all__ = [
    "ConditionalVAE",
]
