"""Per-call inference caches for the existing gated PixelCNN token prior.

Vertical features depend only on earlier token rows and are computed once
per row. Horizontal features retain only the preceding kernel-width context.
No cache is stored in the model or checkpoint, and training remains parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from dl_utils.vae.token_prior import GatedPixelCNNBlock, PixelCNNPrior


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
