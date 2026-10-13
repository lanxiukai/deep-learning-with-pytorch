"""VQGAN causal Transformer blocks, token generation, and per-call KV caches."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerKVCache:
    """Own one layer's K/V tensors, allocating in their projected dtype."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.keys: Tensor
        self.values: Tensor

    def append(
        self, key: Tensor, value: Tensor, position: int
    ) -> tuple[Tensor, Tensor]:
        """Write consecutive single-token positions, starting at zero."""
        if position == 0:
            shape = (key.shape[0], key.shape[1], self.capacity, key.shape[3])
            self.keys = key.new_empty(shape)
            self.values = value.new_empty(shape)
        self.keys[:, :, position : position + 1].copy_(key)
        self.values[:, :, position : position + 1].copy_(value)
        # return two views
        return self.keys[:, :, : position + 1], self.values[:, :, : position + 1]

    def reorder(self, parents: Tensor) -> None:
        """Select beam parents after an append, including duplicates."""
        # parents: (K,)
        # keys_new[i] = keys_old[parents[i]] for i in range(K)
        self.keys = self.keys.index_select(0, parents)
        self.values = self.values.index_select(0, parents)


class KVCache:
    """Group per-layer storage for a single sampling or beam-search call."""

    def __init__(self, layers: int, capacity: int) -> None:
        self.layers = [LayerKVCache(capacity) for _ in range(layers)]

    def __getitem__(self, index: int) -> LayerKVCache:
        return self.layers[index]

    def reorder(self, parents: Tensor) -> None:
        """Reorder every layer to match the selected candidate prefixes."""
        for layer in self.layers:
            layer.reorder(parents)


def _linear(in_features: int, out_features: int) -> nn.Linear:
    """Build a Transformer projection with Xavier weights and zero bias."""
    # Transform the last dimension while preserving all leading dimensions
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class CausalMultiHeadAttention(nn.Module):
    """Self-attention over a full causal sequence or one cached position."""

    def __init__(self, model_dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if model_dim < 1 or heads < 1 or model_dim % heads:
            raise ValueError("Model dimension must be positive and divisible by heads.")
        self.heads = heads
        self.head_dim = model_dim // heads
        self.dropout = dropout
        self.qkv = _linear(model_dim, 3 * model_dim)
        self.out_proj = _linear(model_dim, model_dim)

    def forward(
        self,
        hidden: Tensor,
        *,
        cache: LayerKVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        # hidden: (B, T, model_dim)
        batch_size, length, model_dim = hidden.shape
        # hidden (B, T, model_dim) -> qkv (B, T, 3*model_dim)
        # -> (B, T, 3, heads, model_dim // heads)
        # -> (3, B, heads, T, model_dim // heads)
        # -> Q, K, V: each (B, heads, T, model_dim // heads)
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
        )  # (B, heads, T, model_dim // heads), Dropout(attention_weights) @ V
        # -> (B, T, heads, model_dim // heads) -> (B, T, model_dim)
        attended = attended.transpose(1, 2).reshape(batch_size, length, model_dim)
        return self.out_proj(attended)  # (B, T, model_dim)


class FeedForward(nn.Sequential):
    """Position-wise GELU network with a fourfold hidden expansion."""

    def __init__(self, model_dim: int, dropout: float = 0.0) -> None:
        super().__init__(
            _linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            _linear(4 * model_dim, model_dim),
        )  # (B, T, model_dim) -> (B, T, model_dim)


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
        )  # (B, T, model_dim) -> (B, T, model_dim)
        return hidden + self.dropout2(self.ffn(self.norm2(hidden)))


class CausalInputEmbedding(nn.Module):
    """Embed image tokens and an input-only BOS, then add learned positions."""

    def __init__(
        self, vocabulary_size: int, sequence_length: int, model_dim: int
    ) -> None:
        super().__init__()
        self.bos_token = vocabulary_size
        # token_embedding: (vocabulary_size, model_dim)
        self.token_embedding = nn.Embedding(vocabulary_size, model_dim)
        self.bos_embedding = nn.Parameter(torch.empty(model_dim))  # (model_dim,)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, model_dim)
        )  # learned position embedding: (1, T, model_dim)
        # Unit-scale tied embeddings saturate the initial output distribution.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.bos_embedding, std=0.02)

    def forward(self, input_tokens: Tensor, position: int = 0) -> Tensor:
        """Return (B,T,D) embeddings starting at the supplied sequence position."""
        # input_tokens (token IDs): (B, length)
        length = input_tokens.shape[1]  # T or 1
        # BOS has its own vector, outside the shared image-token vocabulary.
        is_bos = input_tokens == self.bos_token  # Boolean (B, length)
        # (B, length, model_dim)
        hidden = self.token_embedding(input_tokens.masked_fill(is_bos, 0))
        hidden = torch.where(is_bos.unsqueeze(-1), self.bos_embedding, hidden)
        # output: (B, length, model_dim)
        return hidden + self.position_embedding[:, position : position + length]


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
        self.input_embedding = CausalInputEmbedding(
            vocabulary_size, sequence_length, model_dim
        )  # (B, T) -> (B, T, model_dim)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(model_dim, heads, dropout) for _ in range(layers)
        )  # (B, T, model_dim) -> (B, T, model_dim)
        self.normalization = nn.LayerNorm(model_dim)
        # (B, T, model_dim) -> (B, T, V), V = vocabulary_size
        self.head = nn.Linear(model_dim, vocabulary_size, bias=False)
        self.head.weight = self.input_embedding.token_embedding.weight

    def forward(
        self,
        input_tokens: Tensor,
        *,
        cache: KVCache | None = None,
        position: int = 0,
    ) -> Tensor:
        """Return logits for a full sequence or one cached token.
        Cached calls require eval mode and disabled gradients.
        """
        # input_tokens (token IDs): (B, T)
        hidden = self.input_embedding(input_tokens, position)
        for index, block in enumerate(self.blocks):
            layer_cache = cache[index] if cache is not None else None
            hidden = block(hidden, cache=layer_cache, position=position)
        return self.head(self.normalization(hidden))  # (B, T, V)

    def teacher_forcing(
        self,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return logits and targets for (B,H,W) or (B,T) indices."""
        # indices: (B, H, W) or (B, T), T = H * W
        targets = indices.flatten(1)  # flatten indices or targets (B, T)
        bos = targets.new_full((targets.shape[0], 1), self.bos_token)  # (B, 1)
        inputs = torch.cat((bos, targets[:, :-1]), dim=1)  # shift right (B, T)
        return self(inputs), targets  # logits (B, T, V), labels

    @contextmanager
    def _generation(
        self, count: int, device: torch.device
    ) -> Generator[tuple[Tensor, KVCache]]:
        """Create generation buffers and restore the previous dropout mode."""
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full(
                (count, self.sequence_length + 1),
                self.bos_token,
                dtype=torch.long,
                device=device,
            )  # (count, T+1)
            yield tokens, KVCache(len(self.blocks), self.sequence_length)
        finally:
            self.train(was_training)

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
        """Generate tokens with a per-call KV cache; VQGAN validates options.
        Temperature scales logits before filtering; greedy ignores it.
        """
        with self._generation(count, device) as (tokens, cache):
            for position in range(self.sequence_length):
                logits = self(
                    tokens[:, position : position + 1], cache=cache, position=position
                )[:, 0]
                tokens[:, position + 1 : position + 2] = _sample_token(
                    logits, method, temperature, top_k, top_p
                )
            return tokens[:, 1:]

    @torch.inference_mode()
    def beam_search(self, beam_width: int, *, device: torch.device) -> Tensor:
        """Return up to beam_width fixed-length candidates, best score first.
        Rank by cumulative log probability, without temperature or length penalty.
        """
        with self._generation(1, device) as (tokens, cache):
            scores = torch.zeros(1, device=device)
            for position in range(self.sequence_length):
                logits = self(
                    tokens[:, position : position + 1], cache=cache, position=position
                )[:, 0]
                candidates = scores[:, None] + logits.log_softmax(dim=-1)
                scores, selected = candidates.flatten().topk(
                    min(beam_width, candidates.numel())
                )
                parents = selected // self.vocabulary_size
                tokens = tokens.index_select(0, parents)
                tokens[:, position + 1] = selected % self.vocabulary_size
                cache.reorder(parents)
            return tokens[:, 1:]


def _sample_token(
    logits: Tensor, method: str, temperature: float, top_k: int, top_p: float
) -> Tensor:
    """Choose one token per batch item from already validated options."""
    # logits: (B, V), method: ["multinomial", "greedy", "top_k", "top_p"]
    if method == "greedy":
        return logits.argmax(dim=-1, keepdim=True)  # (B, 1)
    logits = logits / temperature
    if method == "top_k":
        values, indices = logits.topk(top_k, dim=-1)  # filter and sort, (B, top_k)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, indices, values)
    elif method == "top_p" and top_p < 1:
        # Each sample is filtered independently.
        values, indices = logits.sort(dim=-1, descending=True)  # (B, V)
        remove = values.softmax(dim=-1).cumsum(dim=-1) >= top_p
        # Keep the token that first reaches the requested mass.
        remove[:, 1:] = remove[:, :-1].clone()  # shift right
        # Keep the first token if it reaches the requested mass.
        remove[:, 0] = False
        logits = torch.empty_like(logits).scatter(
            -1, indices, values.masked_fill(remove, -torch.inf)
        )
    return torch.multinomial(logits.softmax(dim=-1), 1)  # (B, 1)


__all__ = [
    "CausalInputEmbedding",
    "CausalMultiHeadAttention",
    "CausalTransformerBlock",
    "CausalTransformerPrior",
    "FeedForward",
    "KVCache",
    "LayerKVCache",
]
