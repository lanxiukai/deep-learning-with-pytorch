"""Incremental KV-cache inference for the VQGAN Transformer token prior.

Reuse the prior's own PyTorch layer weights. Caches belong to one sampling
call, carry no learned parameters, and never enter a checkpoint. Training
continues to use the existing parallel TransformerEncoder forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from dl_utils.vae.token_prior import CausalTransformerPrior


class _TransformerKVCache:
    """Append one input token at a time to an eval-mode prior's key/value cache."""

    def __init__(self, prior: CausalTransformerPrior, *, labels: Tensor | None) -> None:
        self.prior = prior
        self.layers = [
            cast(nn.TransformerEncoderLayer, layer)
            for layer in prior.transformer.layers
        ]
        self.condition: Tensor | float = 0.0
        if prior.class_embedding is not None:
            if labels is None:
                raise ValueError("A class-conditional prior requires labels.")
            self.condition = prior.class_embedding(labels)
        self.position = 0
        self.positions = torch.arange(
            prior.sequence_length, device=prior.position_embedding.device
        )
        self.keys: list[Tensor] = []
        self.values: list[Tensor] = []

    def logits(self, token: Tensor, *, position: Tensor | None = None) -> Tensor:
        """Use an eager prefix or a fixed-shape masked cache for CUDA Graph replay."""
        if position is None:
            if self.position >= self.prior.sequence_length:
                raise ValueError(
                    "The KV cache has reached the prior's sequence length."
                )
            positional = self.prior.position_embedding[0, self.position]
            mask = None
        else:
            positional = (
                self.prior.position_embedding[0].index_select(0, position).squeeze(0)
            )
            mask = (self.positions <= position)[None, None, None, :]
        hidden = self.prior.token_embedding(token) + positional + self.condition
        count = token.shape[0]
        end = self.position + 1
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            normalized = layer.norm1(hidden)
            query, key, value = F.linear(
                normalized,
                cast(Tensor, attention.in_proj_weight),
                attention.in_proj_bias,
            ).chunk(3, dim=-1)
            shape = (count, attention.num_heads, 1, attention.head_dim)
            query, key, value = (
                tensor.reshape(shape) for tensor in (query, key, value)
            )
            if index == len(self.keys):
                # Allocate once in the projection's dtype. Future slots must be
                # finite for masked attention in the fixed-shape CUDA path.
                cache_shape = (
                    count,
                    attention.num_heads,
                    self.prior.sequence_length,
                    attention.head_dim,
                )
                self.keys.append(key.new_zeros(cache_shape))
                self.values.append(value.new_zeros(cache_shape))
            if position is None:
                self.keys[index][:, :, self.position : end].copy_(key)
                self.values[index][:, :, self.position : end].copy_(value)
                keys, values = (
                    self.keys[index][:, :, :end],
                    self.values[index][:, :, :end],
                )
            else:
                self.keys[index].index_copy_(2, position, key)
                self.values[index].index_copy_(2, position, value)
                keys, values = self.keys[index], self.values[index]
            # Slicing or the explicit mask excludes future keys. is_causal=True
            # would incorrectly align this one-query row to the first key only.
            attended = F.scaled_dot_product_attention(
                query, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=False
            ).reshape(count, attention.embed_dim)
            hidden = hidden + attention.out_proj(attended)
            hidden = hidden + layer.linear2(
                layer.activation(layer.linear1(layer.norm2(hidden)))
            )
        if position is None:
            self.position = end
        return self.prior.head(self.prior.normalization(hidden))


def _sample_with_cuda_graph(
    cache: _TransformerKVCache, token: Tensor, generated: Tensor, temperature: float
) -> Tensor:
    """Replay one token step against fixed-capacity K/V storage and a valid-prefix mask."""
    position = torch.zeros(1, dtype=torch.long, device=token.device)

    def step() -> None:
        logits = cache.logits(token, position=position) / temperature
        sampled = torch.multinomial(logits.softmax(dim=-1), 1)
        generated.index_copy_(1, position, sampled)
        token.copy_(sampled.squeeze(1))
        position.add_(1)

    with torch.cuda.device(token.device):
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=token.device)
        current = torch.cuda.current_stream(token.device)
        # Warmup/capture initialize kernels and buffers without spending the
        # caller's random draws. Only position zero is touched during setup.
        with torch.random.fork_rng(devices=[token.device]):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    token.fill_(cache.prior.bos_token)
                    position.zero_()
                    step()
            current.wait_stream(stream)
            # KV buffers were allocated on the warmup stream and are replayed
            # on the caller's stream. Keep their storage alive until replay ends.
            for buffer in cache.keys + cache.values:
                buffer.record_stream(current)
            token.fill_(cache.prior.bos_token)
            position.zero_()
            with torch.cuda.graph(graph, stream=stream):
                step()
        token.fill_(cache.prior.bos_token)
        position.zero_()
        for _ in range(generated.shape[1]):
            graph.replay()
    return generated


@torch.inference_mode()
def sample_transformer_cached(
    prior: CausalTransformerPrior,
    count: int,
    *,
    device: torch.device,
    labels: Tensor | None,
    temperature: float,
) -> Tensor:
    """Generate a full sequence with fixed-capacity, per-layer KV buffers."""
    cache = _TransformerKVCache(prior, labels=labels)
    token = torch.full((count,), prior.bos_token, dtype=torch.long, device=device)
    generated = token.new_empty(count, prior.sequence_length)
    if token.is_cuda:
        return _sample_with_cuda_graph(cache, token, generated, temperature)
    for position in range(prior.sequence_length):
        logits = cache.logits(token) / temperature
        token = torch.multinomial(logits.softmax(dim=-1), 1).squeeze(1)
        generated[:, position] = token
    return generated
