"""RBM primitive functions: binarization, fixed-point quantization, and fast energy computation.

This module holds leaf-level helpers that depend only on :class:`BinaryRBM`
via a TYPE_CHECKING-only import (``from __future__ import annotations``),
so it stays dependency-free at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from .rbm_model import BinaryRBM


def binarize(
    x: torch.Tensor, mode: str = "stochastic", threshold: float = 0.5
) -> torch.Tensor:
    """
    x: float in [0,1]
    mode: "stochastic" or "threshold"
    threshold: the threshold for the threshold mode
    Returns: {0,1} float tensor
    """
    if mode == "threshold":
        return (x >= threshold).to(x.dtype)
    if mode == "stochastic":
        # Sample Bernoulli with probabilities x
        return torch.bernoulli(x)
    raise ValueError(f"Unknown binarize_mode: {mode}")


@torch.no_grad()
def _quantize_int16_q_(x: torch.Tensor, q_frac: int) -> torch.Tensor:
    """
    In-place quantize a tensor to signed int16 fixed-point with ``q_frac``
    fractional bits, then dequantize back to float (i.e., keep the tensor
    dtype but snap values to the int16 grid and saturate to int16 range).

    This matches the FPGA side expectation in ``rbm_sampling_fpga/``:

    * parameters stored as int16 (suggested Q4.12 ⇒ ``q_frac=12``)
    * quantization rule on PC host: round(x * 2^q_frac) then clamp
      to [-32768, 32767]
    """
    qf = int(q_frac)
    if qf < 0:
        raise ValueError(f"q_frac must be >= 0, got {q_frac}")

    # Use float32 math for stability/consistency; copy back to preserve
    # original dtype/device.
    scale = float(1 << qf)
    q = torch.round(x.to(torch.float32) * scale)
    q = torch.clamp(q, -32768.0, 32767.0)
    x.copy_((q / scale).to(dtype=x.dtype))
    return x


@torch.no_grad()
def rbm_energy_free_energy_fast(
    rbm: BinaryRBM,
    v: torch.Tensor,
    *,
    h: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Equivalent energy/free-energy computation for an RBM, reusing a single
    ``v @ W`` to avoid redundant matmul.

    * ``h`` is **None** — sample ``h`` from ``p(h|v)`` using the original
      logic (used for energy monitoring during RBM sampling).
    * ``h`` is **not None** — use the provided ``h`` (used to separately
      monitor RBM1/RBM2 during DBN sampling).

    Returns:
        energy: [B]
        free_energy: [B]
        h: [B, H] (sampled/provided hidden states)
    """
    vW = v @ rbm.W
    h_arg = rbm.bh + vW

    if h is None:
        p_h = torch.sigmoid(h_arg)
        h = torch.bernoulli(p_h)

    inter = torch.sum(vW * h, dim=1)
    v_bias = torch.sum(v * rbm.bv, dim=1)
    h_bias = torch.sum(h * rbm.bh, dim=1)
    energy = -(inter + v_bias + h_bias)

    v_term = torch.matmul(v, rbm.bv)
    h_term = torch.sum(F.softplus(h_arg), dim=1)
    free_energy = -v_term - h_term

    return energy, free_energy, h
