"""Causal image-token Transformer, reusable blocks, and per-call KV caches."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerKVCache:
    """Own one layer's K/V tensors, allocating in their projected dtype."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("KV cache capacity must be positive.")
        self.capacity = capacity
        self.length = 0
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    def append(
        self, key: Tensor, value: Tensor, position: int
    ) -> tuple[Tensor, Tensor]:
        """Append one position and return views of the populated prefix."""
        if position != self.length:
            raise ValueError("KV cache positions must be appended consecutively.")
        if position >= self.capacity:
            raise ValueError("KV cache capacity exceeded.")
        if key.ndim != 4 or key.shape != value.shape or key.shape[2] != 1:
            raise ValueError("Cached K/V must have matching [B, heads, 1, dim] shapes.")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("Cached K/V must share a dtype and device.")
        if self.keys is None:
            shape = (key.shape[0], key.shape[1], self.capacity, key.shape[3])
            self.keys = key.new_empty(shape)
            self.values = value.new_empty(shape)
        keys, values = self.keys, self.values
        assert keys is not None and values is not None
        if key.shape[:2] != keys.shape[:2] or key.shape[3] != keys.shape[3]:
            raise ValueError("K/V shape changed within a decoding call.")
        if key.dtype != keys.dtype or key.device != keys.device:
            raise ValueError("K/V dtype or device changed within a decoding call.")
        keys[:, :, position : position + 1].copy_(key)
        values[:, :, position : position + 1].copy_(value)
        self.length += 1
        return keys[:, :, : self.length], values[:, :, : self.length]

    def reorder(self, parents: Tensor) -> None:
        """Select beam parents along the batch axis, including duplicates."""
        if self.keys is not None:
            assert self.values is not None
            self.keys = self.keys.index_select(0, parents)
            self.values = self.values.index_select(0, parents)


class KVCache:
    """Group per-layer storage for a single sampling or beam-search call."""

    def __init__(self, layers: int, capacity: int) -> None:
        if layers < 1:
            raise ValueError("KV cache needs at least one layer.")
        self.layers = [LayerKVCache(capacity) for _ in range(layers)]

    def __getitem__(self, index: int) -> LayerKVCache:
        return self.layers[index]

    def reorder(self, parents: Tensor) -> None:
        """Reorder every layer to match the selected candidate prefixes."""
        for layer in self.layers:
            layer.reorder(parents)


class CausalMultiHeadAttention(nn.Module):
    """Self-attention over a full causal sequence or one cached position."""

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if model_dim < 1 or heads < 1 or model_dim % heads:
            raise ValueError("Model dimension must be positive and divisible by heads.")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        for linear in (self.qkv, self.out_proj):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        hidden: Tensor,
        *,
        cache: LayerKVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        batch_size, length, model_dim = hidden.shape
        if cache is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError(
                "Cached attention requires eval mode with gradients disabled."
            )
        query, key, value = (
            self.qkv(hidden)
            .reshape(batch_size, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if cache is not None:
            key, value = cache.append(key, value, position)
        # A cached single query sees the entire populated prefix, including itself.
        # A non-square is_causal mask would align that query to the first key.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=cache is None,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, length, model_dim)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    """Position-wise GELU network with a fourfold hidden expansion."""

    def __init__(self, model_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(model_dim, 4 * model_dim)
        self.linear2 = nn.Linear(4 * model_dim, model_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        for linear in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(hidden))))


class CausalTransformerBlock(nn.Module):
    """Pre-norm MHA/FFN residual block with optional caller-owned K/V storage."""

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.attention = CausalMultiHeadAttention(model_dim, heads, dropout)
        self.ffn = FeedForward(model_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        *,
        cache: LayerKVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        hidden = hidden + self.dropout1(
            self.attention(self.norm1(hidden), cache=cache, position=position)
        )
        return hidden + self.dropout2(self.ffn(self.norm2(hidden)))


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
        if layers < 1:
            raise ValueError("Causal Transformer needs at least one block.")
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(model_dim, heads, dropout) for _ in range(layers)
        )
        self.normalization = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, vocabulary_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # Unit-scale tied embeddings saturate the initial output distribution.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.bos_embedding, std=0.02)

    def _embed_tokens(self, input_tokens: Tensor, position: int = 0) -> Tensor:
        length = input_tokens.shape[1]
        # BOS has its own vector, outside the shared image-token vocabulary.
        is_bos = input_tokens == self.bos_token
        hidden = self.token_embedding(input_tokens.masked_fill(is_bos, 0))
        hidden = torch.where(is_bos.unsqueeze(-1), self.bos_embedding, hidden)
        hidden = hidden + self.position_embedding[:, position : position + length]
        return hidden

    def _logits(self, hidden: Tensor) -> Tensor:
        return self.head(self.normalization(hidden))

    def forward(self, input_tokens: Tensor) -> Tensor:
        hidden = self._embed_tokens(input_tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self._logits(hidden)

    def _decode_step(
        self,
        token: Tensor,
        position: int,
        cache: KVCache,
    ) -> Tensor:
        """Decode one position in eval mode using a cache owned by the caller."""
        hidden = self._embed_tokens(token, position)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cache=cache[index], position=position)
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
            cache = KVCache(len(self.blocks), self.sequence_length)
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
            cache = KVCache(len(self.blocks), self.sequence_length)
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
                cache.reorder(parents)
            return tokens[:, 1:]
        finally:
            self.train(was_training)


__all__ = [
    "CausalMultiHeadAttention",
    "CausalTransformerBlock",
    "CausalTransformerPrior",
    "FeedForward",
    "KVCache",
    "LayerKVCache",
]
