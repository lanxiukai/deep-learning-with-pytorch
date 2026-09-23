"""Standalone CD-k update function for :class:`~dl_utils.ebm.rbm_model.BinaryRBM`.

The :func:`cd_k_update` function is the parameter-update core (the ``cd_k``
in contrastive divergence) extracted from the :class:`BinaryRBM` method so
that the model class stays under the 250‑LOC ceiling.  The original method
on ``BinaryRBM`` is now a thin wrapper that delegates here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from ._ebm_types import RBMUpdateMetrics
from .rbm_primitives import rbm_energy_free_energy_fast

if TYPE_CHECKING:
    from .rbm_model import BinaryRBM


@torch.no_grad()
def cd_k_update(
    rbm: BinaryRBM,
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
    """Contrastive Divergence CD-k parameter update (manual, no autograd).

    ``v0`` must be binary ``{0, 1}``.

    Uses expectations:

        positive: v0^T p(h|v0)
        negative: vk^T p(h|vk)

    (Often uses samples for ``h``, but using probabilities reduces variance.)

    Returns:
        dict with monitoring stats ``{"mse", "bce", "energy", "free_energy", "pl"}``.
    """
    B = v0.size(0)

    # Positive phase (expectation): update uses probabilities.
    # Compute pre-activation once so we can reuse it for optional monitoring
    # metrics (avoids repeated matmul when metrics are enabled).
    h_arg0 = rbm.bh + v0 @ rbm.W
    p_h0 = torch.sigmoid(h_arg0)

    # Monitoring metrics are optional and can be expensive (also triggers GPU
    # sync via .item()).  Compute them sparsely in the training loop.
    if compute_metrics:
        h0 = torch.bernoulli(p_h0)
        p_v1 = rbm.prob_v_given_h(h0)
        mse_t = torch.mean((v0 - p_v1) ** 2)
        bce_t = F.binary_cross_entropy(p_v1.clamp(1e-6, 1 - 1e-6), v0)

        # Avoid an extra matmul by reusing h_arg0 (where h_arg0 = bh + v0@W).
        e_per_sample, fe_per_sample, _ = rbm_energy_free_energy_fast(
            rbm, v0, h=h0
        )
        energy_t = e_per_sample.mean()
        free_energy_t = fe_per_sample.mean()

        # One GPU→CPU sync instead of 4× .item()
        mse, bce, energy, free_energy = (
            float(metric)
            for metric in torch.stack(
                [mse_t, bce_t, energy_t, free_energy_t]
            )
            .detach()
            .cpu()
            .tolist()
        )
    else:
        mse = math.nan
        bce = math.nan
        energy = math.nan
        free_energy = math.nan

    # Negative phase: run k Gibbs steps (CD-k or PCD-k)
    if use_pcd:
        if rbm.persistent_v is None or rbm.persistent_v.size(0) != B:
            if pcd_init == "data":
                rbm.persistent_v = v0.clone()
            else:
                rbm.persistent_v = torch.bernoulli(
                    0.5 * torch.ones(B, rbm.n_visible, device=rbm.device)
                )
        vk = rbm.persistent_v
    else:
        vk = v0

    for _ in range(k):
        _, _, _, vk = rbm.gibbs_vhv(vk)

    p_hk = rbm.prob_h_given_v(vk)

    if use_pcd:
        rbm.persistent_v = vk.detach()

    pos = v0.t() @ p_h0
    neg = vk.t() @ p_hk
    dW = (pos - neg) / B

    dbv = (v0 - vk).mean(dim=0)
    dbh = (p_h0 - p_hk).mean(dim=0)

    if weight_decay > 0:
        dW = dW - weight_decay * rbm.W

    rbm.W_vel = momentum * rbm.W_vel + lr * dW
    rbm.bv_vel = momentum * rbm.bv_vel + lr * dbv
    rbm.bh_vel = momentum * rbm.bh_vel + lr * dbh

    rbm.W.add_(rbm.W_vel)
    rbm.bv.add_(rbm.bv_vel)
    rbm.bh.add_(rbm.bh_vel)

    if max_w_norm > 0:
        col_norms = torch.linalg.norm(rbm.W, dim=0)
        scale = torch.clamp(max_w_norm / (col_norms + 1e-8), max=1.0)
        rbm.W.mul_(scale.unsqueeze(0))

    rbm._maybe_quantize_params_()

    out: RBMUpdateMetrics = {
        "mse": mse,
        "bce": bce,
        "energy": energy,
        "free_energy": free_energy,
    }
    if compute_pl:
        out["pl"] = (
            rbm.pseudo_likelihood(v0).item()
            if compute_metrics
            else math.nan
        )
    return out
