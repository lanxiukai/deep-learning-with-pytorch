"""Quantizers and reusable image-tokenizer blocks for VQ/FSQ lessons."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _token_usage_from_counts(counts: Tensor) -> dict[str, Tensor]:
    probabilities = counts / counts.sum().clamp_min(1.0)
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return {
        "perplexity": entropy.exp().detach(),
        "active_codes": nonzero.sum().detach(),
        "usage_fraction": nonzero.float().mean().detach(),
        "token_entropy_nats": entropy.detach(),
    }


def token_usage(indices: Tensor, vocabulary_size: int) -> dict[str, Tensor]:
    """Return marginal token entropy diagnostics for one observation window."""
    counts = torch.bincount(indices.reshape(-1), minlength=vocabulary_size).float()
    return _token_usage_from_counts(counts)


class TokenUsageAccumulator:
    """Accumulate exact token counts over an epoch or evaluation window."""

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size
        self.counts: Tensor | None = None

    def update(self, indices: Tensor) -> None:
        counts = torch.bincount(
            indices.detach().reshape(-1), minlength=self.vocabulary_size
        )
        if self.counts is None:
            self.counts = torch.zeros_like(counts)
        self.counts += counts

    def statistics(self) -> dict[str, Tensor]:
        return _token_usage_from_counts(self.counts.float())


class VectorQuantizer(nn.Module):
    """Nearest-neighbour VQ with the original gradient-updated codebook."""

    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 64,
        commitment: float = 0.25,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment = commitment
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size
        )

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        flat = z_e.permute(0, 2, 3, 1).contiguous().reshape(-1, self.embedding_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(
            z_e.shape[0], z_e.shape[2], z_e.shape[3], self.embedding_dim
        )
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = self.commitment * F.mse_loss(z_e, z_q.detach())
        # Forward is exactly z_q; the decoder gradient sees identity wrt z_e.
        z_st = z_e + (z_q - z_e).detach()
        index_grid = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        diagnostics = token_usage(index_grid, self.codebook_size)
        diagnostics.update(
            {
                "codebook_loss": codebook_loss.detach(),
                "commitment_loss": commitment_loss.detach(),
                "quantization_mse": F.mse_loss(z_e, z_q).detach(),
            }
        )
        return z_st, index_grid, codebook_loss + commitment_loss, diagnostics

    def lookup(self, indices: Tensor) -> Tensor:
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()


class FiniteScalarQuantizer(nn.Module):
    """FSQ with fixed scalar levels and reversible mixed-radix indices."""

    def __init__(self, levels: Sequence[int] = (8, 8, 5, 5)) -> None:
        super().__init__()
        levels = tuple(int(level) for level in levels)
        levels_tensor = torch.tensor(levels, dtype=torch.long)
        basis = torch.ones_like(levels_tensor)
        if len(levels) > 1:
            basis[1:] = torch.cumprod(levels_tensor[:-1], dim=0)
        self.register_buffer("levels", levels_tensor)
        self.register_buffer("basis", basis)
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

    def _digits_to_values(self, digits: Tensor) -> Tensor:
        levels = self.levels.to(dtype=digits.dtype)
        return 2.0 * digits / (levels - 1.0) - 1.0

    def _values_to_digits(self, bounded: Tensor) -> Tensor:
        levels = self.levels.to(dtype=bounded.dtype)
        return torch.round((bounded + 1.0) * (levels - 1.0) / 2.0)

    def pack(self, digits: Tensor) -> Tensor:
        return (digits.long() * self.basis).sum(dim=-1)

    def unpack(self, indices: Tensor) -> Tensor:
        return (indices[..., None] // self.basis % self.levels).long()

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        bounded = torch.tanh(z_e).permute(0, 2, 3, 1).contiguous()
        digits = self._values_to_digits(bounded)
        quantized = self._digits_to_values(digits)
        z_st = bounded + (quantized - bounded).detach()
        indices = self.pack(digits)
        diagnostics = token_usage(indices, self.codebook_size)
        diagnostics["quantization_mse"] = F.mse_loss(bounded, quantized).detach()
        return z_st.permute(0, 3, 1, 2).contiguous(), indices, diagnostics

    def indices_to_values(self, indices: Tensor) -> Tensor:
        digits = self.unpack(indices).to(dtype=torch.float32)
        values = self._digits_to_values(digits)
        return values.permute(0, 3, 1, 2).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.net(inputs)


class ImageEncoder(nn.Module):
    """Image encoder with an explicit power-of-two spatial compression."""

    def __init__(
        self,
        out_channels: int,
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, hidden_channels // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
        ]
        in_channels = hidden_channels // 2
        for step in range(1, downsample_steps):
            layers.append(nn.Conv2d(in_channels, hidden_channels, 4, 2, 1))
            if step < downsample_steps - 1:
                layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels
        layers.extend(
            [
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                ResidualBlock(hidden_channels),
                ResidualBlock(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


class ImageDecoder(nn.Module):
    """Mirror an ``ImageEncoder`` and map features to RGB in [-1, 1]."""

    def __init__(
        self,
        in_channels: int,
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(downsample_steps - 2):
            layers.extend(
                [
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels // 2, image_channels, 4, 2, 1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class VQVAE(nn.Module):
    """VQ-VAE tokenizer with configurable spatial compression."""

    def __init__(
        self,
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        codebook_size: int = 512,
        commitment: float = 0.25,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        self.downsample_steps = downsample_steps
        self.encoder = ImageEncoder(
            embedding_dim,
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
        )
        self.quantizer = VectorQuantizer(codebook_size, embedding_dim, commitment)
        self.decoder = ImageDecoder(
            embedding_dim,
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(images))

    def encode_indices(self, images: Tensor) -> Tensor:
        """Encode images and return only their discrete token grid."""
        return self.encode(images)[1]

    def decode_indices(self, indices: Tensor) -> Tensor:
        return self.decoder(self.quantizer.lookup(indices))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, quantizer_loss, diagnostics = self.encode(images)
        return self.decoder(z_st), indices, quantizer_loss, diagnostics


class FSQAutoencoder(nn.Module):
    """The same configurable tokenizer with FSQ replacing learned VQ."""

    def __init__(
        self,
        levels: Sequence[int] = (8, 8, 5, 5),
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        self.downsample_steps = downsample_steps
        self.quantizer = FiniteScalarQuantizer(levels)
        self.encoder = ImageEncoder(
            self.quantizer.dim,
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
        )
        self.decoder = ImageDecoder(
            self.quantizer.dim,
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(images))

    def encode_indices(self, images: Tensor) -> Tensor:
        """Encode images and return only their discrete token grid."""
        return self.encode(images)[1]

    def decode_indices(self, indices: Tensor) -> Tensor:
        return self.decoder(self.quantizer.indices_to_values(indices))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, diagnostics = self.encode(images)
        return self.decoder(z_st), indices, diagnostics


__all__ = [
    "VQVAE",
    "FSQAutoencoder",
    "FiniteScalarQuantizer",
    "ImageDecoder",
    "ImageEncoder",
    "ResidualBlock",
    "TokenUsageAccumulator",
    "VectorQuantizer",
    "token_usage",
]
