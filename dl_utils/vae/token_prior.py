"""Compact causal token priors shared by VQ-VAE, FSQ, and VQGAN."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.quantization import VQVAE, FSQAutoencoder


def make_fixed_class_labels(
    num_classes: int,
    samples_per_class: int,
    device: torch.device,
) -> Tensor:
    """Return adjacent, balanced labels for readable generated-image grids."""
    return torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)


class MaskedConv2d(nn.Conv2d):
    """PixelCNN mask: type A excludes the current token; B includes it."""

    mask: Tensor

    def __init__(
        self, mask_type: str, *args: Any, stack: str = "raster", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        mask = torch.ones_like(self.weight)
        center_h = self.kernel_size[0] // 2
        center_w = self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        first_blocked = center_w if mask_type == "A" else center_w + 1
        mask[:, :, center_h, first_blocked:] = 0
        if stack == "vertical":
            # A excludes the whole current row; B receives already shifted features.
            mask[:, :, center_h, :] = 0 if mask_type == "A" else 1
        elif stack == "horizontal":
            mask[:, :, :center_h, :] = 0
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


class GatedPixelCNNBlock(nn.Module):
    """Separate above-row and left-row streams to remove the raster blind spot."""

    def __init__(self, channels: int, mask_type: str, kernel_size: int) -> None:
        super().__init__()
        self.vertical = MaskedConv2d(
            mask_type,
            channels,
            2 * channels,
            kernel_size,
            padding=kernel_size // 2,
            stack="vertical",
        )
        self.horizontal = MaskedConv2d(
            mask_type,
            channels,
            2 * channels,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            stack="horizontal",
        )
        self.vertical_to_horizontal = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        self.residual = mask_type == "B"

    @staticmethod
    def gate(features: Tensor) -> Tensor:
        value, gate = features.chunk(2, dim=1)
        return value.tanh() * gate.sigmoid()

    def forward(self, vertical: Tensor, horizontal: Tensor, condition: Tensor | float):
        above = self.vertical(vertical)
        left = self.horizontal(horizontal)
        update = self.gate(left + self.vertical_to_horizontal(above) + condition)
        update = self.output(update)
        # The first block cannot bypass its A mask with unmasked token embeddings.
        horizontal = horizontal + update if self.residual else update
        return self.gate(above + condition), horizontal


class PixelCNNPrior(nn.Module):
    """Gated PixelCNN with distinct vertical/horizontal streams.

    Sixteen layers (7x7, then fifteen 3x3 blocks) cover the entire 16x16
    causal context. Smaller lesson experiments may explicitly use fewer layers.
    Set num_classes=0 for an unconditional prior without a class embedding.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        hidden_channels: int = 64,
        layers: int = 16,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("PixelCNN needs at least one masked layer.")
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocabulary_size, hidden_channels)
        self.class_embedding = (
            nn.Embedding(num_classes, 2 * hidden_channels) if num_classes else None
        )
        self.causal = nn.ModuleList(
            [
                GatedPixelCNNBlock(
                    hidden_channels, "A" if i == 0 else "B", 7 if i == 0 else 3
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
        vertical = horizontal = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        condition: Tensor | float = 0.0
        if self.class_embedding is not None:
            if labels is None:
                raise ValueError("A conditional PixelCNN requires class labels.")
            condition = self.class_embedding(labels)[:, :, None, None]
        for block in self.causal:
            vertical, horizontal = block(vertical, horizontal, condition)
        return self.head(horizontal)

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
        indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
        for row in range(height):
            for column in range(width):
                logits = self(indices, labels=labels)[:, :, row, column] / temperature
                indices[:, row, column] = torch.multinomial(
                    logits.softmax(dim=1), 1
                ).squeeze(1)
        return indices


def train_pixelcnn_prior_epoch(
    prior: PixelCNNPrior,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    *,
    progress: tqdm,
    log_every: int = 100,
) -> float:
    """Train one causal-prior epoch over cached frozen-token grids."""
    prior.train()
    metrics = MetricAccumulator(("nll",), device=device)
    for batch_index, (indices, labels) in enumerate(loader, 1):
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if prior.num_classes else None
        loss = F.cross_entropy(prior(indices, labels=labels), indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.add_batch_means((loss,), num_examples=indices.shape[0])
        if batch_index % log_every == 0:
            nll = metrics.compute_weighted_means(require_finite=True)["nll"]
            progress.set_postfix(
                nll=f"{nll:.4f}", bpt=f"{nll / math.log(2):.3f}", refresh=False
            )
        progress.update(1)
    return metrics.compute_weighted_means(require_finite=True)["nll"]


@torch.inference_mode()
def evaluate_pixelcnn_prior(
    prior: PixelCNNPrior,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    tokens_per_image: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure PixelCNN NLL for one frozen tokenizer."""
    prior.eval()
    metrics = MetricAccumulator(("nll",), device=device)
    for indices, labels in loader:
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if prior.num_classes else None
        loss = F.cross_entropy(prior(indices, labels=labels), indices)
        metrics.add_batch_means((loss,), num_examples=indices.shape[0])
    nll = metrics.compute_weighted_means(require_finite=True)["nll"]
    return {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": tokens_per_image * nll / math.log(2),
    }


@torch.inference_mode()
def sample_pixelcnn_prior_images(
    tokenizer: VQVAE | FSQAutoencoder,
    prior: PixelCNNPrior,
    count: int,
    *,
    grid_size: int,
    device: torch.device,
    temperature: float,
    labels: Tensor | None = None,
) -> Tensor:
    """Sample a square token grid and decode it to an image batch."""
    indices = prior.sample(
        count,
        grid_size,
        grid_size,
        device=device,
        labels=labels.to(device) if labels is not None else None,
        temperature=temperature,
    )
    return tokenizer.decode_indices(indices)


class CausalTransformerPrior(nn.Module):
    """Teacher-forced causal Transformer over a fixed-length token sequence."""

    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        *,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.bos_token = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size + 1, model_dim)
        self.class_embedding = nn.Embedding(num_classes, model_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, sequence_length, model_dim) / math.sqrt(model_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        # TransformerEncoder clones one prototype, so initialize each clone separately.
        for encoder_layer in self.transformer.layers:
            block = cast(nn.TransformerEncoderLayer, encoder_layer)
            nn.init.xavier_uniform_(cast(Tensor, block.self_attn.in_proj_weight))
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.normalization = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocabulary_size, bias=False)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, input_tokens: Tensor, labels: Tensor) -> Tensor:
        length = input_tokens.shape[1]
        hidden = self.token_embedding(input_tokens)
        hidden = hidden + self.position_embedding[:, :length]
        hidden = hidden + self.class_embedding(labels)[:, None, :]
        hidden = self.transformer(
            hidden, mask=self._causal_mask(length, input_tokens.device)
        )
        return self.head(self.normalization(hidden))

    def teacher_forcing(
        self,
        indices: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return logits and targets for ``[B,H,W]`` or ``[B,T]`` indices."""
        targets = indices.flatten(1)
        bos = targets.new_full((targets.shape[0], 1), self.bos_token)
        inputs = torch.cat((bos, targets[:, :-1]), dim=1)
        return self(inputs, labels), targets

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        sequence = torch.full(
            (count, 1), self.bos_token, dtype=torch.long, device=device
        )
        generated: list[Tensor] = []
        for _ in range(self.sequence_length):
            logits = self(sequence, labels)[:, -1] / temperature
            token = torch.multinomial(logits.softmax(dim=1), 1)
            generated.append(token)
            if len(generated) < self.sequence_length:
                sequence = torch.cat((sequence, token), dim=1)
        return torch.cat(generated, dim=1)


__all__ = [
    "CausalTransformerPrior",
    "MaskedConv2d",
    "PixelCNNPrior",
    "evaluate_pixelcnn_prior",
    "make_fixed_class_labels",
    "sample_pixelcnn_prior_images",
    "train_pixelcnn_prior_epoch",
]
