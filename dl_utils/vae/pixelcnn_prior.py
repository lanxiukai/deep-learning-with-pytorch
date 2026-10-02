"""Gated PixelCNN token prior and cached raster sampling.

Inference caches belong to a single sampling call and are absent from checkpoints.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
            # A excludes the current image row. B combines vertical features
            # that already depend only on earlier image rows.
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
    The default prior is unconditional. A positive num_classes explicitly
    enables class conditioning and requires labels for forward/sampling calls.
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
        """Generate with caches: CUDA Graph replay on CUDA, eager steps on CPU."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive.")
        if min(count, height, width) < 1:
            raise ValueError("Sample count and grid dimensions must be positive.")
        indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
        return sample_pixelcnn_cached(
            self, indices, labels=labels, temperature=temperature
        )


@torch.inference_mode()
def sample_pixelcnn_cached(
    prior: PixelCNNPrior,
    indices: Tensor,
    *,
    labels: Tensor | None,
    temperature: float,
) -> Tensor:
    count, height, width = indices.shape
    cache = _PixelCNNCache(
        prior, count, height, width, device=indices.device, labels=labels
    )
    if indices.is_cuda:
        return _sample_with_cuda_graph(cache, indices, temperature)
    for row in range(height):
        cache.begin_row(row)
        for column in range(width):
            logits = cache.logits(column) / temperature
            tokens = torch.multinomial(logits.softmax(dim=1), 1).squeeze(1)
            indices[:, row, column] = tokens
            cache.append_token(column, tokens)
        cache.end_row(row)
    return indices


class _PixelCNNCache:
    """One raster scan over a batch; use inside an inference-mode context."""

    def __init__(
        self,
        prior: PixelCNNPrior,
        count: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        labels: Tensor | None,
    ) -> None:
        self.prior = prior
        self.blocks = [cast("GatedPixelCNNBlock", block) for block in prior.causal]
        self.condition: Tensor | float = 0.0
        if prior.class_embedding is not None:
            if labels is None:
                raise ValueError("A conditional PixelCNN requires class labels.")
            self.condition = prior.class_embedding(labels)[:, :, None, None]
        channels = prior.embedding.embedding_dim
        self.vertical_inputs: list[Tensor] = []
        self.vertical_weights: list[Tensor] = []
        self.horizontal_weights: list[Tensor] = []
        self.history: list[Tensor] = []
        self.row_context: list[Tensor] = []
        for block in self.blocks:
            top = block.vertical.kernel_size[0] // 2
            left = block.horizontal.kernel_size[1] // 2
            self.vertical_inputs.append(
                prior.embedding.weight.new_zeros(
                    count, channels, height + top, width, device=device
                )
            )
            # A excludes the current input; B sees an already-causal feature.
            rows = top + int(block.residual)
            columns = left + int(block.residual)
            self.vertical_weights.append(
                (block.vertical.weight * block.vertical.mask)[:, :, :rows].contiguous()
            )
            self.horizontal_weights.append(
                (block.horizontal.weight * block.horizontal.mask)[:, :, 0, :columns]
                .permute(0, 2, 1)
                .flatten(1)
                .contiguous()
            )
            self.history.append(
                prior.embedding.weight.new_zeros(count, left, channels, device=device)
            )
            self.row_context.append(
                prior.embedding.weight.new_empty(
                    count, width, 2 * channels, device=device
                )
            )
        self.token_row = prior.embedding.weight.new_empty(
            count, channels, width, device=device
        )

    def begin_row(self, row: int) -> None:
        """Cache every vertical-to-horizontal contribution for the next row."""
        for index, block in enumerate(self.blocks):
            weight = self.vertical_weights[index]
            above = F.conv2d(
                self.vertical_inputs[index][:, :, row : row + weight.shape[2]],
                weight,
                block.vertical.bias,
                padding=(0, block.vertical.kernel_size[1] // 2),
            )
            context = block.vertical_to_horizontal(above) + self.condition
            self.row_context[index].copy_(context.squeeze(2).transpose(1, 2))
            if index + 1 < len(self.blocks):
                next_top = self.blocks[index + 1].vertical.kernel_size[0] // 2
                self.vertical_inputs[index + 1][:, :, row + next_top].copy_(
                    block.gate(above + self.condition).squeeze(2)
                )
            self.history[index].zero_()

    def logits(self, column: int | Tensor) -> Tensor:
        """Advance hidden horizontal features by exactly one position."""
        horizontal: Tensor | None = None
        for index, block in enumerate(self.blocks):
            context = self.history[index]
            if horizontal is not None:
                context = torch.cat((context, horizontal[:, None]), dim=1)
                self.history[index].copy_(context[:, 1:])
            left = F.linear(
                context.flatten(1),
                self.horizontal_weights[index],
                block.horizontal.bias,
            )
            above = (
                self.row_context[index].index_select(1, column).squeeze(1)
                if isinstance(column, Tensor)
                else self.row_context[index][:, column]
            )
            update = block.gate(left + above)
            update = F.linear(
                update, block.output.weight[:, :, 0, 0], block.output.bias
            )
            horizontal = horizontal + update if horizontal is not None else update
        assert horizontal is not None
        hidden_head = cast(nn.Conv2d, self.prior.head[1])
        output_head = cast(nn.Conv2d, self.prior.head[3])
        hidden = F.linear(
            self.prior.head[0](horizontal),
            hidden_head.weight[:, :, 0, 0],
            hidden_head.bias,
        )
        return F.linear(
            self.prior.head[2](hidden),
            output_head.weight[:, :, 0, 0],
            output_head.bias,
        )

    def append_token(self, column: int | Tensor, tokens: Tensor) -> None:
        embedded = self.prior.embedding(tokens)
        self.history[0].copy_(
            torch.cat((self.history[0][:, 1:], embedded[:, None]), dim=1)
        )
        if isinstance(column, Tensor):
            self.token_row.index_copy_(2, column, embedded[:, :, None])
        else:
            self.token_row[:, :, column].copy_(embedded)

    def end_row(self, row: int) -> None:
        top = self.blocks[0].vertical.kernel_size[0] // 2
        self.vertical_inputs[0][:, :, row + top].copy_(self.token_row)


def _sample_with_cuda_graph(
    cache: _PixelCNNCache, indices: Tensor, temperature: float
) -> Tensor:
    """Replay a fixed one-token computation with a device-side column counter."""
    count, height, width = indices.shape
    column = torch.zeros(1, dtype=torch.long, device=indices.device)
    sampled_row = indices.new_empty(count, width)

    def step() -> None:
        logits = cache.logits(column) / temperature
        tokens = torch.multinomial(logits.softmax(dim=1), 1).squeeze(1)
        cache.append_token(column, tokens)
        sampled_row.index_copy_(1, column, tokens[:, None])
        column.add_(1)

    with torch.cuda.device(indices.device):
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=indices.device)
        current = torch.cuda.current_stream(indices.device)
        # Warmup/capture must not spend the caller's sampling random numbers.
        with torch.random.fork_rng(devices=[indices.device]):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                cache.begin_row(0)
                for _ in range(3):
                    column.zero_()
                    step()
            current.wait_stream(stream)
            column.zero_()
            with torch.cuda.graph(graph, stream=stream):
                step()
        for row in range(height):
            cache.begin_row(row)
            column.zero_()
            for _ in range(width):
                graph.replay()
            indices[:, row].copy_(sampled_row)
            cache.end_row(row)
    return indices


__all__ = [
    "MaskedConv2d",
    "PixelCNNPrior",
]
