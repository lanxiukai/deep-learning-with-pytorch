"""HVAE / Ladder VAE notation from reading guide 3.3c, sections 3-4.

Layers: 0 = x, 1 = z1, 2 = z2. mu is the mean; v = log(sigma**2).
Both models share these generative distributions::

    Distribution      Network       Parameters
    p(z2)             none          fixed N(0, I)
    p_theta(z1 | z2)  self.p_1(z2)  mu_p_1, v_p_1
    p_theta(x | z1)   self.p_0(z1)  mu_p_0; fixed variance 1/2
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    ImageDecoder,
    ImageEncoder,
    diagonal_gaussian_kl_from_logvar,
    fuse_diagonal_gaussians,
    reparameterize_logvar,
    split_gaussian_parameters,
)


class _TopPosterior(nn.Module):
    """q_2: x -> (h_1, mu_q_2, v_q_2); h_1 is reused by lower inference."""

    def __init__(
        self, hidden_channels: int, context_dim: int, z2_dim: int
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ImageEncoder(hidden_channels),
            nn.Linear(hidden_channels * 4 * 4, context_dim),
            nn.SiLU(),
        )  # (B, 3, 256, 256) -> (B, hidden_channels * 4 * 4)
           # -> h_1: (B, context_dim)
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z2_dim),
        )  # -> (B, context_dim) -> (B, 2 * z2_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return h_1: (B, context_dim), mu_q_2/v_q_2: (B, z2_dim)."""
        h_1 = self.features(x)
        mu_q_2, v_q_2 = split_gaussian_parameters(self.head(h_1))
        return h_1, mu_q_2, v_q_2


class HierarchicalVAE(nn.Module):
    """Top-down HVAE inference (guide section 3)::

        Distribution            Network              Parameters
        q_phi^down(z2 | x)      self.q_2(x)          mu_q_2, v_q_2
        q_phi^down(z1 | z2, x)  self.q_1([h_1, z2])  mu_q_1, v_q_1

    q_2(x) also returns h_1, the shared image feature. [h_1, z2] denotes
    concatenation. The common p distributions are listed in the module docstring.
    """

    posterior_family = "conditional_network"

    def __init__(
        self,
        *,
        z1_dim: int = 96,
        z2_dim: int = 32,
        hidden_channels: int = 256,
        context_dim: int = 512,
    ) -> None:
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.q_2 = _TopPosterior(hidden_channels, context_dim, z2_dim)

        self.p_1 = nn.Sequential(
            nn.Linear(z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        self.p_0 = ImageDecoder(z1_dim, hidden_channels)
        # Build shared blocks first so equal seeds match them across both models.
        self._init_lower_inference()

    def _init_lower_inference(self) -> None:
        """Build q_1; Ladder overrides this to build q_hat_1."""
        self.q_1 = nn.Sequential(
            nn.Linear(self.context_dim + self.z2_dim, self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, 2 * self.z1_dim),
        )

    def lower_parameters(
        self,
        h_1: Tensor,
        z2: Tensor,
    ) -> dict[str, Tensor]:
        """Return mu_p_1/v_p_1 and mu_q_1/v_q_1, each shaped (B, z1_dim)."""
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        mu_q_1, v_q_1 = split_gaussian_parameters(
            self.q_1(torch.cat((h_1, z2), dim=1))
        )
        return {
            "mu_p_1": mu_p_1,
            "v_p_1": v_p_1,
            "mu_q_1": mu_q_1,
            "v_q_1": v_q_1,
        }

    def infer(self, x: Tensor, *, sample: bool = True) -> dict[str, Tensor]:
        """Return z1/z2, their distribution parameters, and h_1.

        sample=False follows successive conditional means instead of sampling.
        """
        # q(z2 | x) -> z2.
        h_1, mu_q_2, v_q_2 = self.q_2(x)
        z2 = reparameterize_logvar(mu_q_2, v_q_2) if sample else mu_q_2
        # p(z1 | z2), q(z1 | z2, x) -> z1.
        parameters_1 = self.lower_parameters(h_1, z2)
        mu_q_1, v_q_1 = parameters_1["mu_q_1"], parameters_1["v_q_1"]
        z1 = reparameterize_logvar(mu_q_1, v_q_1) if sample else mu_q_1
        return {
            # Layer 2.
            "mu_q_2": mu_q_2,
            "v_q_2": v_q_2,
            "z2": z2,
            # Layer 1.
            **parameters_1,
            "z1": z1,
            "h_1": h_1,
        }

    def decode(self, z1: Tensor) -> Tensor:
        """Return mu_p_0(z1): (..., z1_dim) -> (..., 3, 256, 256)."""
        return self.p_0(z1)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return (mu_p_0, latents) from stochastic inference followed by decoding."""
        latents = self.infer(x, sample=True)
        mu_p_0 = self.decode(latents["z1"])
        return mu_p_0, latents

    def generate(self, z2: Tensor, lower_noise: Tensor) -> Tensor:
        """Return mu_p_0 using supplied z2 and standard-normal lower_noise."""
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        z1 = reparameterize_logvar(mu_p_1, v_p_1, noise=lower_noise)
        return self.decode(z1)

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        """Draw both prior latents and return count RGB observation means."""
        return self.generate(
            torch.randn(count, self.z2_dim, device=device),
            torch.randn(count, self.z1_dim, device=device),
        )


class LadderVAE(HierarchicalVAE):
    """Ladder inference (guide section 4); p distributions are unchanged::

        Distribution or factor       Network / operation  Parameters
        q_phi^down(z2 | x)           self.q_2(x)          mu_q_2, v_q_2
        q_tilde_phi(z1 | x)          self.q_hat_1(h_1)    mu_hat_q_1, v_hat_q_1
        q_{theta,phi}^L(z1 | z2, x)  precision fusion     mu_q_1, v_q_1

    hat denotes unfused evidence; the shared mu_q_1/v_q_1 keys denote the
    guide's mu_q_1^L/v_q_1^L. Fusion has no learned network of its own.
    """

    posterior_family = "ladder_precision_fusion"

    def _init_lower_inference(self) -> None:
        """Build q_hat_1, the Gaussian evidence network."""
        self.q_hat_1 = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            nn.SiLU(),
            nn.Linear(self.context_dim, 2 * self.z1_dim),
        )

    def lower_parameters(
        self,
        h_1: Tensor,
        z2: Tensor,
    ) -> dict[str, Tensor]:
        """Return prior, hatted evidence, and fused posterior mu/v for z1."""
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        mu_hat_q_1, v_hat_q_1 = split_gaussian_parameters(self.q_hat_1(h_1))
        # Equation (4.3): fused q,1; keep gradients through both inputs.
        mu_q_1, v_q_1 = fuse_diagonal_gaussians(
            mu_p_1,
            v_p_1,
            mu_hat_q_1,
            v_hat_q_1,
        )
        return {
            "mu_p_1": mu_p_1,
            "v_p_1": v_p_1,
            "mu_hat_q_1": mu_hat_q_1,
            "v_hat_q_1": v_hat_q_1,
            "mu_q_1": mu_q_1,
            "v_q_1": v_q_1,
        }


def hierarchical_vae_loss(
    mu_p_0: Tensor,
    x: Tensor,
    latents: dict[str, Tensor],
    *,
    kl_weight: float,
    free_bits: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return (loss, detached metrics), shared by HVAE and Ladder.

    Guide equations (3.4)/(4.5): distortion = mean(D_i),
    kl_z1 = mean(r_1(x_i, z2_i)), kl_z2 = mean(R_2(x_i)).
    The fixed-variance Gaussian observation constant is omitted.

    free_bits clamps each layer's batch-mean KL in nats; kl_weight scales
    their sum. kl_z1/z2 stay raw; kl_objective_z1/z2 are clamped, unweighted.
    """
    # First sum coordinates within each example; retain the batch dimension.
    distortion_per_example = (mu_p_0 - x).square().flatten(1).sum(dim=1)
    # Layer 1: KL(q(z1 | z2, x) || p(z1 | z2)) at the same sampled z2.
    kl_z1_per_example = diagonal_gaussian_kl_from_logvar(
        latents["mu_q_1"],
        latents["v_q_1"],
        latents["mu_p_1"],
        latents["v_p_1"],
    ).sum(dim=1)
    # Layer 2: KL(q(z2 | x) || N(0, I)).
    kl_z2_per_example = diagonal_gaussian_kl_from_logvar(
        latents["mu_q_2"], latents["v_q_2"]
    ).sum(dim=1)

    # Batch means are the raw metrics; free bits acts on each whole layer.
    distortion = distortion_per_example.mean()
    kl_z1 = kl_z1_per_example.mean()
    kl_z2 = kl_z2_per_example.mean()
    threshold = distortion.new_tensor(float(free_bits))
    kl_objective_z1 = torch.maximum(kl_z1, threshold)
    kl_objective_z2 = torch.maximum(kl_z2, threshold)
    loss = distortion + float(kl_weight) * (kl_objective_z1 + kl_objective_z2)
    return loss, {
        "distortion": distortion.detach(),
        "kl_z1": kl_z1.detach(),
        "kl_z2": kl_z2.detach(),
        "kl_objective_z1": kl_objective_z1.detach(),
        "kl_objective_z2": kl_objective_z2.detach(),
        "kl_weight": distortion.new_tensor(float(kl_weight)),
    }


def model_config(
    model: HierarchicalVAE,
) -> dict[str, int]:
    return {
        "z1_dim": model.z1_dim,
        "z2_dim": model.z2_dim,
        "hidden_channels": model.hidden_channels,
        "context_dim": model.context_dim,
    }


__all__ = [
    "HierarchicalVAE",
    "LadderVAE",
    "hierarchical_vae_loss",
    "model_config",
]
