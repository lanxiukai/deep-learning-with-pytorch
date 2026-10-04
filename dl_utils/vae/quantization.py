"""Quantizers for VQ-VAE, FSQ, and VQGAN, plus VQ-VAE/FSQ image blocks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# VQ-VAE, FSQ, and VQGAN map 256x256 images to 16x16 token grids.
TOKENIZER_DOWNSAMPLE_STEPS = 4


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, C, H, W)
        return inputs + self.net(inputs)


def validate_image_size(height: int, width: int, downsample_steps: int) -> None:
    """Require positive spatial dimensions divisible by the compression factor."""
    factor = 2**downsample_steps
    if height < factor or width < factor or height % factor or width % factor:
        raise ValueError(
            f"Image height and width must be positive multiples of {factor}; "
            f"got {height}x{width}."
        )


class ImageEncoder(nn.Module):
    """Image encoder with configurable compression; the default is 16x."""

    def __init__(
        self,
        out_channels: int,
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        downsample_steps: int = TOKENIZER_DOWNSAMPLE_STEPS,
    ) -> None:
        super().__init__()
        if hidden_channels < 2 or downsample_steps < 2:
            raise ValueError(
                "hidden_channels and downsample_steps must be at least two"
            )
        self.downsample_steps = downsample_steps
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
        validate_image_size(images.shape[-2], images.shape[-1], self.downsample_steps)
        # (B, image_channels, H, W) -> (B, out_channels, H//F, W//F)
        # F = 2**downsample_steps
        return self.net(images)


class ImageDecoder(nn.Module):
    """Mirror an ``ImageEncoder`` and reconstruct images in [-1, 1]."""

    def __init__(
        self,
        in_channels: int,
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        downsample_steps: int = TOKENIZER_DOWNSAMPLE_STEPS,
    ) -> None:
        super().__init__()
        if hidden_channels < 2 or downsample_steps < 2:
            raise ValueError(
                "hidden_channels and downsample_steps must be at least two"
            )
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
        # (B, in_channels, H, W) -> (B, image_channels, H*F, W*F)
        # F = 2**downsample_steps
        # Output values are in [-1, 1]
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Nearest-neighbour VQ with training-only EMA codebook updates.

    The embedding is frozen for autograd; quantizer loss is commitment loss.
    Counts and vector sums are persistent model buffers.
    """

    ema_cluster_size: Tensor
    ema_embedding_sum: Tensor

    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 64,
        commitment: float = 0.25,
        *,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if not math.isfinite(ema_epsilon) or ema_epsilon <= 0.0:
            raise ValueError("ema_epsilon must be finite and positive.")
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment = commitment
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size
        )
        self.embedding.requires_grad_(False)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer(
            "ema_embedding_sum", torch.zeros_like(self.embedding.weight)
        )

    @torch.no_grad()
    def _update_ema(self, flat: Tensor, indices: Tensor) -> None:
        # flat: (N, D), where N = B * H * W; all latent vectors in the batch
        # indices: (N,), the codebook index selected for each latent vector
        counts = torch.bincount(indices, minlength=self.codebook_size).to(
            self.ema_cluster_size
        )  # counts[i] is the number of occurrences of i in indices
        sums = torch.zeros_like(self.ema_embedding_sum)
        # Accumulate in the buffer dtype, including under mixed precision.
        sums.index_add_(0, indices, flat.to(sums))
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            counts, alpha=1.0 - self.ema_decay
        )  # new = old * ema_decay + (1 - ema_decay) * counts
        self.ema_embedding_sum.mul_(self.ema_decay).add_(
            sums, alpha=1.0 - self.ema_decay
        )  # new = old * ema_decay + (1 - ema_decay) * sums
        # Apply additive smoothing to the EMA counts while preserving the total count
        total = self.ema_cluster_size.sum()
        smoothed_counts = (
            (self.ema_cluster_size + self.ema_epsilon)
            / (total + self.codebook_size * self.ema_epsilon)
            * total
        )
        # Never-assigned codes keep their initialization instead of collapsing
        # to zero or dividing an arbitrary initial vector by a tiny count.
        self.embedding.weight.copy_(
            torch.where(
                self.ema_cluster_size[:, None] > 0,
                self.ema_embedding_sum / smoothed_counts[:, None],
                self.embedding.weight,
            )
        )

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        flat = z_e.permute(0, 2, 3, 1).contiguous().reshape(-1, self.embedding_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2.0 * flat @ self.embedding.weight.t()
        )  # (N, K), N = B*h*w
        indices = distances.argmin(dim=1)  # (N,)
        z_q = self.embedding(indices).view(
            z_e.shape[0], z_e.shape[2], z_e.shape[3], self.embedding_dim
        )
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        quantization_mse = F.mse_loss(z_e, z_q.detach())  # scaler ()
        commitment_loss = self.commitment * quantization_mse
        # Forward is exactly z_q; the decoder gradient sees identity wrt z_e.
        z_st = z_e + (z_q - z_e).detach()
        index_grid = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        diagnostics = {
            "quantization_mse": quantization_mse.detach(),
        }
        # This batch uses the pre-update vectors for its outputs and losses.
        if self.training:
            self._update_ema(flat, indices)
        return z_st, index_grid, commitment_loss, diagnostics

    def indices_to_values(self, indices: Tensor) -> Tensor:
        """Restore quantized latents (B, C, H, W) from token indices (B, H, W)."""
        # indices: (B, h, w), return z_q.detach(): (B, C, h, w), C = embedding_dim
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()


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
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        downsample_steps: int = TOKENIZER_DOWNSAMPLE_STEPS,
    ) -> None:
        super().__init__()
        self.downsample_steps = downsample_steps
        self.encoder = ImageEncoder(
            embedding_dim,
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
        )
        self.quantizer = VectorQuantizer(
            codebook_size,
            embedding_dim,
            commitment,
            ema_decay=ema_decay,
            ema_epsilon=ema_epsilon,
        )
        self.decoder = ImageDecoder(
            embedding_dim,
            image_channels=image_channels,
            hidden_channels=hidden_channels,
            downsample_steps=downsample_steps,
        )

    def encode(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        # images (B, C_img, H, W) -> quantizer z_st ...
        return self.quantizer(self.encoder(images))

    def encode_indices(self, images: Tensor) -> Tensor:
        """Encode images and return only their discrete token grid."""
        # images (B, C_img, H, W) -> indices (B, h, w)
        return self.encode(images)[1]

    def decode_indices(self, indices: Tensor) -> Tensor:
        # indices (B, h, w) -> gen_images (B, C_img, H, W)
        return self.decoder(self.quantizer.indices_to_values(indices))

    def forward(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, quantizer_loss, diagnostics = self.encode(images)
        # return:
        # reconstructions (B, C_img, H, W), indices (B, h, w)
        # commitment_loss (), diagnostics dict[str, ()]
        return self.decoder(z_st), indices, quantizer_loss, diagnostics


class FiniteScalarQuantizer(nn.Module):
    """FSQ with centered integer levels and reversible mixed-radix indices.

    Even counts use an offset: eight levels give [-4, -3, ..., 3] / 4.
    See the FSQ paper, Appendix A.1, for the bounded-rounding construction.
    """

    levels: Tensor
    basis: Tensor
    half_width: Tensor

    def __init__(self, levels: Sequence[int] = (8, 8, 8)) -> None:
        super().__init__()
        levels = tuple(int(level) for level in levels)
        if not levels or any(level < 2 for level in levels):
            raise ValueError("FSQ needs nonempty levels, each at least two.")
        levels_tensor = torch.tensor(levels, dtype=torch.long)
        basis = torch.ones_like(levels_tensor)
        if len(levels) > 1:
            basis[1:] = torch.cumprod(levels_tensor[:-1], dim=0)  # [1, 8, 64]
        self.register_buffer("levels", levels_tensor)
        self.register_buffer("basis", basis)
        # Derived from levels; keep the checkpoint format unchanged.
        self.register_buffer("half_width", levels_tensor // 2, persistent=False)
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

    def _digits_to_values(self, digits: Tensor) -> Tensor:
        half_width = self.half_width.to(dtype=digits.dtype)
        return (digits - half_width) / half_width

    def bound(self, values: Tensor) -> Tensor:
        """Use the bounding function published in the FSQ paper, Appendix A.1."""
        # values: encoder output z_e rearranged to (B, H, W, C).
        levels = self.levels.to(dtype=values.dtype)
        half_range = (levels - 1.0) * (1.0 - 1e-3) / 2.0  # a_r
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)  # o_r
        shift = torch.tan(offset / half_range)  # delta_r = tan(o_r/a_r)
        # Output: bounded continuous values, same shape; not yet rounded.
        return torch.tanh(values + shift) * half_range - offset

    def pack(self, digits: Tensor) -> Tensor:
        # Pack nonnegative digits (B, H, W, C) into indices (B, H, W).
        return (digits.long() * self.basis).sum(dim=-1)

    def unpack(self, indices: Tensor) -> Tensor:
        # Restore nonnegative digits (B, H, W, C) from indices (B, H, W).
        return (indices[..., None] // self.basis % self.levels).long()

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        bounded = self.bound(z_e.permute(0, 2, 3, 1).contiguous())
        rounded = bounded.round()
        half_width = self.half_width.to(dtype=bounded.dtype)
        z_st = (bounded + (rounded - bounded).detach()) / half_width
        indices = self.pack(rounded + half_width)
        # FSQ latent MSE is not directly comparable with VQ latent MSE.
        diagnostics = {
            "quantization_mse": F.mse_loss(
                bounded / half_width, rounded / half_width
            ).detach()
        }
        return z_st.permute(0, 3, 1, 2).contiguous(), indices, diagnostics

    def indices_to_values(self, indices: Tensor) -> Tensor:
        """Restore quantized latents (B, C, H, W) from token indices (B, H, W)."""
        # Restore quantized values (B, C, H, W) from indices (B, H, W), C = self.dim.
        digits = self.unpack(indices).to(dtype=torch.float32)
        values = self._digits_to_values(digits)
        return values.permute(0, 3, 1, 2).contiguous()


class FSQAutoencoder(nn.Module):
    """FSQ image tokenizer using the same encoder/decoder as VQ-VAE."""

    def __init__(
        self,
        levels: Sequence[int] = (8, 8, 8),
        *,
        image_channels: int = 3,
        hidden_channels: int = 128,
        downsample_steps: int = TOKENIZER_DOWNSAMPLE_STEPS,
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
    "TOKENIZER_DOWNSAMPLE_STEPS",
    "VQVAE",
    "FSQAutoencoder",
    "FiniteScalarQuantizer",
    "ImageDecoder",
    "ImageEncoder",
    "ResidualBlock",
    "VectorQuantizer",
    "validate_image_size",
]
