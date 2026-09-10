"""Compact causal token priors shared by VQ-VAE, FSQ, and VQGAN."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

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

    def __init__(self, mask_type: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
    """Class-conditional causal prior over an image-token grid."""

    def __init__(
        self,
        vocabulary_size: int,
        *,
        hidden_channels: int = 128,
        layers: int = 7,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocabulary_size, hidden_channels)
        self.class_embedding = nn.Embedding(num_classes, hidden_channels)
        blocks: list[nn.Module] = [
            MaskedConv2d("A", hidden_channels, hidden_channels, 7, padding=3),
            nn.ReLU(inplace=True),
        ]
        for _ in range(layers - 1):
            blocks.extend(
                [
                    MaskedConv2d("B", hidden_channels, hidden_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        self.causal = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, vocabulary_size, 1),
        )

    def forward(
        self,
        indices: Tensor,
        *,
        labels: Tensor,
    ) -> Tensor:
        hidden = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        hidden = self.causal(hidden)
        hidden = hidden + self.class_embedding(labels)[:, :, None, None]
        return self.head(hidden)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        labels: Tensor,
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
    tokenizer: VQVAE | FSQAutoencoder,
    prior: PixelCNNPrior,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    *,
    progress_desc: str = "PixelCNN",
    progress_interval: float = 0.5,
) -> float:
    """Train one causal-prior epoch over frozen tokenizer indices."""
    prior.train()
    nll_sum = 0.0
    examples = 0
    progress = tqdm(loader, desc=progress_desc, mininterval=progress_interval)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # The prior's embedding backward must be able to save these indices.
        with torch.no_grad():
            indices = tokenizer.encode_indices(images)
        loss = F.cross_entropy(prior(indices, labels=labels), indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        nll_sum += loss.item() * images.shape[0]
        examples += images.shape[0]
        progress.set_postfix(
            nll=f"{nll_sum / examples:.4f}",
            bpt=f"{nll_sum / examples / math.log(2):.3f}",
            refresh=False,
        )
    return nll_sum / examples


@torch.inference_mode()
def evaluate_pixelcnn_prior(
    tokenizer: VQVAE | FSQAutoencoder,
    prior: PixelCNNPrior,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    tokens_per_image: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure conditional PixelCNN NLL for one frozen tokenizer."""
    prior.eval()
    nll_sum = 0.0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = tokenizer.encode_indices(images)
        loss = F.cross_entropy(prior(indices, labels=labels), indices)
        nll_sum += float(loss) * images.shape[0]
        examples += images.shape[0]
    nll = nll_sum / examples
    return {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": tokens_per_image * nll / math.log(2),
    }


@torch.inference_mode()
def sample_pixelcnn_prior_images(
    tokenizer: VQVAE | FSQAutoencoder,
    prior: PixelCNNPrior,
    labels: Tensor,
    *,
    grid_size: int,
    device: torch.device,
    temperature: float,
) -> Tensor:
    """Sample a square token grid and decode it to an image batch."""
    labels = labels.to(device)
    indices = prior.sample(
        labels.shape[0],
        grid_size,
        grid_size,
        device=device,
        labels=labels,
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
