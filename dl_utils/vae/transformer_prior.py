"""Causal Transformer token prior with full-prefix autoregressive sampling."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


class CausalTransformerPrior(nn.Module):
    """Fixed-length causal token Transformer, unconditional by default.

    A positive num_classes explicitly enables class conditioning.
    """

    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        *,
        model_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.0,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.bos_token = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size + 1, model_dim)
        self.class_embedding = (
            nn.Embedding(num_classes, model_dim) if num_classes else None
        )
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

    def forward(self, input_tokens: Tensor, labels: Tensor | None = None) -> Tensor:
        length = input_tokens.shape[1]
        hidden = self.token_embedding(input_tokens)
        hidden = hidden + self.position_embedding[:, :length]
        if self.class_embedding is not None:
            if labels is None:
                raise ValueError("A class-conditional prior requires labels.")
            hidden = hidden + self.class_embedding(labels)[:, None, :]
        hidden = self.transformer(
            hidden, mask=self._causal_mask(length, input_tokens.device)
        )
        return self.head(self.normalization(hidden))

    def teacher_forcing(
        self,
        indices: Tensor,
        labels: Tensor | None = None,
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
        labels: Tensor | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """Recompute the full prefix and sample its next token at each step.

        Sampling temporarily disables dropout and restores the previous mode.
        Training and likelihood evaluation use the parallel forward pass.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive.")
        if count < 1 or self.sequence_length < 1:
            raise ValueError("Sample count and sequence length must be positive.")
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full(
                (count, 1), self.bos_token, dtype=torch.long, device=device
            )
            for _ in range(self.sequence_length):
                logits = self(tokens, labels)[:, -1, :] / temperature
                sampled = torch.multinomial(logits.softmax(dim=-1), 1)
                tokens = torch.cat((tokens, sampled), dim=1)
            return tokens[:, 1:]
        finally:
            self.train(was_training)


__all__ = ["CausalTransformerPrior"]
