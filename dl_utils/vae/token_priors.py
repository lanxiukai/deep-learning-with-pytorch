"""PixelCNN and causal Transformer priors for discrete image tokens."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

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
        return self.head(hidden)  # (B, vocabulary_size, h, w)

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


class CausalTransformerPrior(nn.Module):
    """Unconditional causal Transformer with tied image-token input/output weights."""

    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        *,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.bos_token = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size, model_dim)
        self.bos_embedding = nn.Parameter(torch.empty(model_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, model_dim)
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
        self.head.weight = self.token_embedding.weight
        # Unit-scale tied embeddings saturate the initial output distribution.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.bos_embedding, std=0.02)

    def _logits(self, hidden: Tensor) -> Tensor:
        return self.head(self.normalization(hidden))

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def _embed_tokens(self, input_tokens: Tensor, position: int = 0) -> Tensor:
        length = input_tokens.shape[1]
        # BOS has its own vector, outside the shared image-token vocabulary.
        is_bos = input_tokens == self.bos_token
        hidden = self.token_embedding(input_tokens.masked_fill(is_bos, 0))
        hidden = torch.where(is_bos.unsqueeze(-1), self.bos_embedding, hidden)
        hidden = hidden + self.position_embedding[:, position : position + length]
        return hidden

    def forward(self, input_tokens: Tensor) -> Tensor:
        length = input_tokens.shape[1]
        hidden = self._embed_tokens(input_tokens)
        hidden = self.transformer(
            hidden, mask=self._causal_mask(length, input_tokens.device)
        )
        return self._logits(hidden)

    def _decode_step(
        self,
        token: Tensor,
        position: int,
        cache: list[tuple[Tensor, Tensor] | None],
    ) -> Tensor:
        """Decode one position in eval mode using a cache owned by the caller."""
        hidden = self._embed_tokens(token, position)
        batch_size = token.shape[0]
        for index, encoder_layer in enumerate(self.transformer.layers):
            # Use the same pre-norm attention and feedforward weights as training.
            block = cast(nn.TransformerEncoderLayer, encoder_layer)
            attention = block.self_attn
            qkv = F.linear(
                block.norm1(hidden),
                cast(Tensor, attention.in_proj_weight),
                attention.in_proj_bias,
            )
            query, key, value = (
                projected.reshape(
                    batch_size, 1, attention.num_heads, attention.head_dim
                ).transpose(1, 2)
                for projected in qkv.chunk(3, dim=-1)
            )
            layer_cache = cache[index]
            if layer_cache is None:
                # Allocate once per layer/call, in the projected (autocast) dtype.
                shape = (
                    batch_size,
                    attention.num_heads,
                    self.sequence_length,
                    attention.head_dim,
                )
                layer_cache = (key.new_empty(shape), value.new_empty(shape))
                cache[index] = layer_cache
            keys, values = layer_cache
            keys[:, :, position : position + 1].copy_(key)
            values[:, :, position : position + 1].copy_(value)
            # The single query can see every cached position, including itself.
            # is_causal=True would align a non-square mask to the first key.
            attended = F.scaled_dot_product_attention(
                query,
                keys[:, :, : position + 1],
                values[:, :, : position + 1],
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape(batch_size, 1, -1)
            hidden = hidden + block.dropout1(attention.out_proj(attended))
            update = block.linear2(
                block.dropout(block.activation(block.linear1(block.norm2(hidden))))
            )
            hidden = hidden + block.dropout2(update)
        return self._logits(hidden)[:, 0, :]

    def teacher_forcing(
        self,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return logits and targets for ``[B,H,W]`` or ``[B,T]`` indices."""
        targets = indices.flatten(1)
        bos = targets.new_full((targets.shape[0], 1), self.bos_token)
        inputs = torch.cat((bos, targets[:, :-1]), dim=1)
        return self(inputs), targets

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        temperature: float = 1.0,
        method: Literal["multinomial", "greedy", "top_k", "top_p"] = "multinomial",
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tensor:
        """Sample one token at a time with a per-call KV cache.

        Greedy decoding uses argmax without temperature. Stochastic methods
        apply temperature to output logits before top-k or nucleus filtering.
        Sampling temporarily disables dropout and restores the previous mode.
        Training and likelihood evaluation use the parallel forward pass.
        Cache tensors are transient and never become part of the checkpoint.
        """
        if method not in {"multinomial", "greedy", "top_k", "top_p"}:
            raise ValueError("Unknown Transformer sampling method.")
        if method != "greedy" and (not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Sampling temperature must be finite and positive.")
        if method == "top_k" and not 1 <= top_k <= self.vocabulary_size:
            raise ValueError("top_k must be between 1 and the vocabulary size.")
        if method == "top_p" and (not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be finite and in (0, 1].")
        if count < 1 or self.sequence_length < 1:
            raise ValueError("Sample count and sequence length must be positive.")
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full(
                (count, self.sequence_length + 1),
                self.bos_token,
                dtype=torch.long,
                device=device,
            )
            cache: list[tuple[Tensor, Tensor] | None] = [
                None for _ in self.transformer.layers
            ]
            for position in range(self.sequence_length):
                logits = self._decode_step(
                    tokens[:, position : position + 1], position, cache
                )
                if method == "greedy":
                    sampled = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if method == "top_k":
                        values, indices = logits.topk(top_k, dim=-1)
                        logits = torch.full_like(logits, -torch.inf).scatter(
                            -1, indices, values
                        )
                    elif method == "top_p" and top_p < 1:
                        values, indices = logits.sort(dim=-1, descending=True)
                        remove = values.softmax(dim=-1).cumsum(dim=-1) >= top_p
                        # Keep the token that first reaches the requested mass.
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        logits = torch.empty_like(logits).scatter(
                            -1, indices, values.masked_fill(remove, -torch.inf)
                        )
                    sampled = torch.multinomial(logits.softmax(dim=-1), 1)
                tokens[:, position + 1 : position + 2] = sampled
            return tokens[:, 1:]
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def beam_search(self, beam_width: int, *, device: torch.device) -> Tensor:
        """Return up to ``beam_width`` fixed-length candidates, best score first.

        Each step keeps the best cumulative, untempered log probabilities.
        Cached prefixes follow their selected parents, including duplicates.
        There is no EOS or length penalty because every image has equal length.
        """
        if beam_width < 1 or self.sequence_length < 1:
            raise ValueError("Beam width and sequence length must be positive.")
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full(
                (1, self.sequence_length + 1),
                self.bos_token,
                dtype=torch.long,
                device=device,
            )
            scores = torch.zeros(1, device=device)
            cache: list[tuple[Tensor, Tensor] | None] = [
                None for _ in self.transformer.layers
            ]
            for position in range(self.sequence_length):
                logits = self._decode_step(
                    tokens[:, position : position + 1], position, cache
                )
                candidates = scores[:, None] + logits.log_softmax(dim=-1)
                scores, selected = candidates.flatten().topk(
                    min(beam_width, candidates.numel())
                )
                parents = selected // self.vocabulary_size
                tokens = tokens.index_select(0, parents)
                tokens[:, position + 1] = selected % self.vocabulary_size
                for index, layer_cache in enumerate(cache):
                    assert layer_cache is not None
                    keys, values = layer_cache
                    cache[index] = (
                        keys.index_select(0, parents),
                        values.index_select(0, parents),
                    )
            return tokens[:, 1:]
        finally:
            self.train(was_training)


__all__ = [
    "CausalTransformerPrior",
    "MaskedConv2d",
    "PixelCNNPrior",
    "PixelCNNResidualBlock",
]
