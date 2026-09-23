from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ._ebm_types import RBMUpdateMetrics
from .rbm_primitives import _quantize_int16_q_
from .rbm_update import cd_k_update as _cd_k_update_fn


class BinaryRBM(nn.Module):
    """
    Bernoulli-Bernoulli RBM layer.
    Energy: E(v,h) = -v^T W h - bv^T v - bh^T h
    Parameters:
      W: [V, H] matrix of weights
      bv: [V] vector of visible biases
      bh: [H] vector of hidden biases
    Conditional probabilities:
      p(h=1|v) = sigmoid(bh + v W)
      p(v=1|h) = sigmoid(bv + h W^T)
    """
    def __init__(self, n_visible: int, n_hidden: int, device: torch.device):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = device

        # Parameters
        # Small random init for W; zeros for biases (common practice)
        self.W = nn.Parameter(0.01 * torch.randn(n_visible, n_hidden, device=device))
        self.bv = nn.Parameter(torch.zeros(n_visible, device=device))
        self.bh = nn.Parameter(torch.zeros(n_hidden, device=device))

        # For momentum SGD (manual updates)
        self.W_vel = torch.zeros_like(self.W, device=device)
        self.bv_vel = torch.zeros_like(self.bv, device=device)
        self.bh_vel = torch.zeros_like(self.bh, device=device)

        # Persistent chain for PCD (initialized lazily when used)
        self.persistent_v: torch.Tensor | None = None

        # Optional: simulate FPGA-aligned fixed-point parameter precision.
        # When enabled, W/bv/bh are kept on an int16 Qm.q_frac grid (default Q4.12 => q_frac=12).
        self.param_fixed_point: bool = False
        self.param_q_frac: int = 12

    @torch.no_grad()
    def enable_param_fixed_point_(self, enable: bool, *, q_frac: int = 12, quantize_now: bool = True) -> None:
        """Enable/disable parameter fixed-point quantization (float params snapped to int16 grid)."""
        self.param_fixed_point = bool(enable)
        self.param_q_frac = int(q_frac)
        if self.param_fixed_point and quantize_now:
            self.quantize_params_(q_frac=self.param_q_frac)

    @torch.no_grad()
    def quantize_params_(self, *, q_frac: int | None = None) -> None:
        """Quantize W/bv/bh in-place to int16 fixed-point with q_frac fractional bits."""
        qf = int(self.param_q_frac if q_frac is None else q_frac)
        self.param_q_frac = qf
        _quantize_int16_q_(self.W.data, qf)
        _quantize_int16_q_(self.bv.data, qf)
        _quantize_int16_q_(self.bh.data, qf)

    @torch.no_grad()
    def _maybe_quantize_params_(self) -> None:
        if self.param_fixed_point:
            self.quantize_params_(q_frac=self.param_q_frac)

    @torch.no_grad()
    def prob_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bh + v @ self.W)

    @torch.no_grad()
    def prob_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bv + h @ self.W.t())

    @torch.no_grad()
    def sample_h_given_v(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v: [B, V] in {0,1}
        returns (p_h, h_sample) both [B, H]
        """
        p_h = self.prob_h_given_v(v)
        h = torch.bernoulli(p_h)
        return p_h, h

    @torch.no_grad()
    def sample_v_given_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, H] in {0,1}
        returns (p_v, v_sample) both [B, V]
        """
        p_v = self.prob_v_given_h(h)
        v = torch.bernoulli(p_v)
        return p_v, v
    @torch.no_grad()
    def gibbs_vhv(self, v0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One full Gibbs step starting from v:
          v0 -> h -> v
        Returns: (p_h, h, p_v, v)
        """
        p_h, h = self.sample_h_given_v(v0)
        p_v, v = self.sample_v_given_h(h)
        return p_h, h, p_v, v

    @torch.no_grad()
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Free energy F(v) for Bernoulli-Bernoulli RBM:
          F(v) = -bv^T v - sum_j log(1 + exp(bh_j + (vW)_j))
        Returns: [B]
        """
        v_term = torch.matmul(v, self.bv)
        h_arg = self.bh + v @ self.W
        h_term = torch.sum(F.softplus(h_arg), dim=1)
        return -v_term - h_term

    @torch.no_grad()
    def pseudo_likelihood(self, v: torch.Tensor) -> torch.Tensor:
        """
        Approximate log pseudo-likelihood (mean over batch).
        Uses a random bit flip per sample:
          PL(v) ≈ V * log σ(F(v_flip) - F(v))
        """
        B, V = v.shape
        bit_idx = torch.randint(low=0, high=V, size=(B,), device=v.device)
        v_flip = v.clone()
        v_flip[torch.arange(B, device=v.device), bit_idx] = 1.0 - v_flip[torch.arange(B, device=v.device), bit_idx]

        fe = self.free_energy(v)
        fe_flip = self.free_energy(v_flip)
        # avoid log(0)
        return V * torch.log(torch.sigmoid(fe_flip - fe) + 1e-8).mean()

    @torch.no_grad()
    def cd_k_update(
        self,
        v0: torch.Tensor,
        lr: float,
        k: int,
        momentum: float,
        weight_decay: float,
        *,
        use_pcd: bool = False,
        pcd_init: str = "random",
        max_w_norm: float = 0.0,
        compute_pl: bool = False,
        compute_metrics: bool = True,
    ) -> RBMUpdateMetrics:
        """Contrastive Divergence CD-k parameter update — delegates to
        :func:`~dl_utils.ebm.rbm_update.cd_k_update`.

        See the standalone function for the full implementation.
        """
        return _cd_k_update_fn(
            self, v0, lr, k, momentum, weight_decay,
            use_pcd=use_pcd, pcd_init=pcd_init,
            max_w_norm=max_w_norm,
            compute_pl=compute_pl, compute_metrics=compute_metrics,
        )

    @torch.no_grad()
    def reconstruct(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step reconstruction:
          v -> h (sample) -> v_recon_prob
        Returns: (v_recon_prob, v_recon_sample)
        """
        _, _, p_v, v1 = self.gibbs_vhv(v)
        return p_v, v1

    @torch.no_grad()
    def generate(
        self,
        n: int,
        gibbs_steps: int,
        init_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Generate samples by running a Gibbs chain.
        Returns:
          - p_v: visible probabilities [n, V] from the final Gibbs step, or None
            when gibbs_steps is zero
          - v: binary visible samples [n, V]
        """
        if init_v is None:
            v = torch.bernoulli(0.5 * torch.ones(n, self.n_visible, device=self.device))
        else:
            v = init_v.to(self.device)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            m = v.size(0)
            if m != n:
                if m < n:
                    reps = (n + m - 1) // m
                    v = v.repeat(reps, 1)[:n]
                else:
                    v = v[:n]

        p_v = None
        for _ in range(gibbs_steps):
            _, _, p_v, v = self.gibbs_vhv(v)

        return p_v, v
