"""Masked convolution, residual blocks, and raster sampling for PixelCNN priors."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskedConv2d(nn.Conv2d):
    """PixelCNN mask: type A excludes the current token; B includes it."""

    mask: Tensor

    def __init__(self, mask_type: str, *args: Any, **kwargs: Any) -> None:
        if mask_type not in {"A", "B"}:
            raise ValueError("PixelCNN mask type must be A or B.")
        super().__init__(*args, **kwargs)
        # kernel shape: (C_out, C_in/groups, K_h, K_w)
        mask = torch.ones_like(self.weight)
        center_h = self.kernel_size[0] // 2
        center_w = self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        first_blocked = center_w if mask_type == "A" else center_w + 1
        mask[:, :, center_h, first_blocked:] = 0
        self.register_buffer("mask", mask)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNResidualBlock(nn.Module):
    """A plain 1x1 / masked 3x3 / 1x1 bottleneck residual block."""

    def __init__(self, channels: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, bottleneck, 1)
        self.masked = MaskedConv2d("B", bottleneck, bottleneck, 3, padding=1)
        self.expand = nn.Conv2d(bottleneck, channels, 1)
        self.dropout = dropout

    def forward(self, hidden: Tensor) -> Tensor:
        update = self.reduce(F.relu(hidden))
        update = self.masked(F.relu(update))
        update = F.dropout(F.relu(update), self.dropout, self.training)
        return hidden + self.expand(update)


class PixelCNNPrior(nn.Module):
    """Plain bottleneck PixelCNN, following the residual layout of PixelRNN.

    The initial mask-A convolution can cover the whole token grid's past.
    Grouping this large kernel reduces its cost; the following 1x1 convolutions
    mix channels. Blocks contain only convolution, ReLU, dropout, and addition.
    Sampling recomputes the full grid in raster order.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        hidden_channels: int = 256,
        embedding_dim: int = 32,
        blocks: int = 15,
        bottleneck: int = 128,
        head_channels: int = 1024,
        first_kernel_size: int = 63,
        first_groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if first_kernel_size < 1 or first_kernel_size % 2 == 0:
            raise ValueError("PixelCNN kernel size must be positive and odd.")
        if blocks < 1:
            raise ValueError("Residual PixelCNN needs at least one block.")
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)  # (K, C)
        self.first = MaskedConv2d(
            "A",
            embedding_dim,
            hidden_channels,
            first_kernel_size,
            padding=first_kernel_size // 2,
            groups=first_groups,
        )
        self.blocks = nn.ModuleList(
            PixelCNNResidualBlock(hidden_channels, bottleneck, dropout)
            for _ in range(blocks)
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, head_channels, 1),
            nn.ReLU(),
            nn.Conv2d(head_channels, vocabulary_size, 1),
        )

    def forward(self, indices: Tensor) -> Tensor:
        # indices (B, h, w) -> hidden (B, hidden_channels, h, w)
        hidden = self.first(self.embedding(indices).permute(0, 3, 1, 2).contiguous())
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample in raster order with dropout disabled; restore the caller's mode."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive.")
        if min(count, height, width) < 1:
            raise ValueError("Sample count and grid dimensions must be positive.")
        was_training = self.training
        self.eval()
        try:
            indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
            # Recompute the full grid, using only the current position's logits.
            for i in range(height):
                for j in range(width):
                    logits = self(indices)[:, :, i, j] / temperature
                    tokens = torch.multinomial(logits.softmax(dim=1), 1).squeeze(1)
                    indices[:, i, j] = tokens
            return indices
        finally:
            self.train(was_training)


__all__ = ["MaskedConv2d", "PixelCNNPrior", "PixelCNNResidualBlock"]
