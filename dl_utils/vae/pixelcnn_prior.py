"""Basic single-stream PixelCNN token prior and raster-order sampling."""

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


class PixelCNNPrior(nn.Module):
    """Token embeddings, masked convolutions with ReLU, and a 1x1 class head.

    The first convolution uses mask A; later convolutions use mask B on
    already-causal features. The single stream has a receptive-field blind
    spot and does not cover every earlier token, even with sixteen layers.
    The default prior is unconditional. A positive num_classes explicitly
    enables class conditioning and requires labels for forward/sampling calls.
    Former gated-prior checkpoints require retraining.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        hidden_channels: int = 64,
        layers: int = 16,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("PixelCNN needs at least one masked layer.")
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocabulary_size, hidden_channels)
        self.class_embedding = (
            nn.Embedding(num_classes, hidden_channels) if num_classes else None
        )
        self.causal = nn.ModuleList(
            [
                MaskedConv2d(
                    "A" if i == 0 else "B",
                    hidden_channels,
                    hidden_channels,
                    7 if i == 0 else 3,
                    padding=3 if i == 0 else 1,
                )
                for i in range(layers)
            ]
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, vocabulary_size, 1),
        )

    def forward(self, indices: Tensor, *, labels: Tensor | None = None) -> Tensor:
        hidden = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        condition: Tensor | float = 0.0
        if self.class_embedding is not None:
            if labels is None:
                raise ValueError("A conditional PixelCNN requires class labels.")
            condition = self.class_embedding(labels)[:, :, None, None]
        for convolution in self.causal:
            hidden = F.relu(convolution(hidden) + condition)
        return self.head(hidden)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        labels: Tensor | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """Predict one token at a time by recomputing the full grid's logits."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive.")
        if min(count, height, width) < 1:
            raise ValueError("Sample count and grid dimensions must be positive.")
        indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
        for row in range(height):
            for column in range(width):
                logits = self(indices, labels=labels)[:, :, row, column] / temperature
                tokens = torch.multinomial(logits.softmax(dim=1), 1).squeeze(1)
                indices[:, row, column] = tokens
        return indices


__all__ = [
    "MaskedConv2d",
    "PixelCNNPrior",
]
