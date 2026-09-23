'''
Encoder-Decoder Architecture

A self-contained demonstration of the encoder-decoder pattern using tiny
concrete subclasses of the abstract Encoder, Decoder, and EncoderDecoder
interfaces.  No dataset, training loop, GPU, or network access required.
'''

from __future__ import annotations

import torch
import torch.nn as nn

from dl_utils.d2l.seq2seq import Decoder, Encoder, EncoderDecoder


# ── Concrete encoder ────────────────────────────────────────────────────


class TinyEncoder(Encoder):
    """Minimal encoder: embedding → GRU → (outputs, state)."""

    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, batch_first=False)

    def forward(self, X: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        # X shape: (batch, seq_len) → embed → (batch, seq_len, embed_size)
        embedded = self.embedding(X)
        # GRU expects (seq_len, batch, embed_size)
        permuted = embedded.permute(1, 0, 2)
        output, state = self.rnn(permuted)
        # output: (seq_len, batch, num_hiddens)
        # state:  (num_layers, batch, num_hiddens)
        return output, state


# ── Concrete decoder ────────────────────────────────────────────────────


class TinyDecoder(Decoder):
    """Minimal decoder: init_state extracts encoder hidden, then GRU + linear."""

    def __init__(self, vocab_size: int, embed_size: int, num_hiddens: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, batch_first=False)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs: tuple[torch.Tensor, torch.Tensor], *args):
        """Return the encoder's final hidden state as the initial decoder state."""
        # enc_outputs is (output, state) from TinyEncoder.forward
        return enc_outputs[1]

    def forward(
        self, X: torch.Tensor, state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # X shape: (batch, dec_seq_len) → embed → (batch, dec_seq_len, embed_size)
        embedded = self.embedding(X)
        permuted = embedded.permute(1, 0, 2)  # (dec_seq_len, batch, embed_size)
        output, state = self.rnn(permuted, state)  # (dec_seq_len, batch, num_hiddens), (1, batch, num_hiddens)
        output = self.dense(output)                # (dec_seq_len, batch, vocab_size)
        output = output.permute(1, 0, 2)           # (batch, dec_seq_len, vocab_size)
        return output, state


# ── End-to-end demonstration ────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(42)

    # Hyper-parameters for a genuinely tiny model.
    vocab_size = 10
    embed_size = 8
    num_hiddens = 16
    batch_size = 2
    enc_seq_len = 4
    dec_seq_len = 3

    encoder = TinyEncoder(vocab_size, embed_size, num_hiddens)
    decoder = TinyDecoder(vocab_size, embed_size, num_hiddens)
    model = EncoderDecoder(encoder, decoder)

    # Fixed CPU-only inputs.
    enc_X = torch.randint(0, vocab_size, (batch_size, enc_seq_len))
    dec_X = torch.randint(0, vocab_size, (batch_size, dec_seq_len))

    model.eval()
    with torch.no_grad():
        # EncoderDecoder.forward(enc_X, dec_X, *args) exercises the full chain:
        #   encoder → decoder.init_state → decoder.forward
        output, state = model(enc_X, dec_X)

    print(f'output shape: {output.shape}')
    print(f'state shape:  {state.shape}')


if __name__ == '__main__':
    main()
