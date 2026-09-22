"""Two-level hierarchy matching reading guide 3.3c, sections 3 and 4.

Layer 0 is the observation x, layer 1 is z1, and layer 2 is z2. Network
blocks p_ell/q_ell produce parameters for that layer's distribution.
mu_p_ell/mu_q_ell are means; v_p_ell/v_q_ell are log variances,
v = log(sigma**2). The conditioning inputs are shown at each block below.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from dl_utils.vae.image_networks import ImageDecoder, ImageEncoder
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    fuse_diagonal_gaussians,
    reparameterize_logvar,
    split_gaussian_parameters,
)


class HierarchicalVAE(nn.Module):
    """Section 3: q(z2 | x) q(z1 | z2, x) with a direct q_1 network.

    forward(x) returns (mu_p_0, latents). The latent dictionary groups
    q,2 parameters, p,1 and q,1 parameters, samples z2/z1, and the
    deterministic feature h_1. The fixed p(z2) = N(0, I) needs no outputs;
    p(x | z1) has fixed variance 1/2, so p_0 only predicts its mean.
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
        if hidden_channels < 128 or hidden_channels % 128:
            raise ValueError("hidden_channels must be a positive multiple of 128")
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        # Deterministic features: x -> h_0 -> h_1 -> h_2 (no sampling).
        self.bottom_up_0 = ImageEncoder(hidden_channels)
        self.bottom_up_1 = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4, context_dim),
            nn.SiLU(),
        )
        self.bottom_up_2 = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )
        # q,2: h_2(x) -> (mu_q_2(x), v_q_2(x)); section 3.3, step 1.
        self.q_2 = nn.Linear(context_dim, 2 * z2_dim)

        # p,1: z2 -> (mu_p_1(z2), v_p_1(z2)); section 3.3, step 2.
        self.p_1 = nn.Sequential(
            nn.Linear(z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        # q,1: [h_1(x), z2] -> (mu_q_1(x, z2), v_q_1(x, z2)).
        self.q_1 = nn.Sequential(
            nn.Linear(context_dim + z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        # p,0: z1 -> mu_p_0(z1), with sigma_p_0 = 1/sqrt(2); step 3.
        self.p_0 = ImageDecoder(z1_dim, hidden_channels)

    def bottom_up(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return h_1(x), mu_q_2(x), v_q_2(x), before any sampling."""
        if x.shape[1:] != (3, 256, 256):
            raise ValueError("Expected RGB images from the 256x256 glasses cache")
        h_0 = self.bottom_up_0(x)
        h_1 = self.bottom_up_1(h_0)
        h_2 = self.bottom_up_2(h_1)
        mu_q_2, v_q_2 = split_gaussian_parameters(self.q_2(h_2))
        return h_1, mu_q_2, v_q_2

    def lower_distributions(
        self,
        h_1: Tensor,
        z2: Tensor,
    ) -> dict[str, Tensor]:
        """Section 3.3, step 2: p,1 and q,1 share the same sampled z2."""
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

    def infer_from_top(
        self,
        h_1: Tensor,
        mu_q_2: Tensor,
        v_q_2: Tensor,
        z2: Tensor,
        *,
        sample_lower: bool,
    ) -> dict[str, Tensor]:
        parameters_1 = self.lower_distributions(h_1, z2)
        mu_q_1, v_q_1 = parameters_1["mu_q_1"], parameters_1["v_q_1"]
        z1 = reparameterize_logvar(mu_q_1, v_q_1) if sample_lower else mu_q_1
        return {
            # Layer 2: q(z2 | x), then its sample. p(z2) is fixed N(0, I).
            "mu_q_2": mu_q_2,
            "v_q_2": v_q_2,
            "z2": z2,
            # Layer 1: p(z1 | z2), q(z1 | z2, x), then the q sample.
            **parameters_1,
            "z1": z1,
            "h_1": h_1,
        }

    def infer(self, x: Tensor, *, sample: bool = True) -> dict[str, Tensor]:
        # Section 3.3, step 1 (also section 4.4): infer and sample layer 2.
        h_1, mu_q_2, v_q_2 = self.bottom_up(x)
        z2 = reparameterize_logvar(mu_q_2, v_q_2) if sample else mu_q_2
        # Step 2: construct layer 1 distributions, then sample from q,1.
        return self.infer_from_top(
            h_1,
            mu_q_2,
            v_q_2,
            z2,
            sample_lower=sample,
        )

    def decode(self, z1: Tensor) -> Tensor:
        """Section 3.3, step 3: return the observation mean mu_p_0(z1)."""
        leading_shape = z1.shape[:-1]
        flat_z1 = z1.reshape(-1, self.z1_dim)
        mu_p_0 = self.p_0(flat_z1)
        return mu_p_0.reshape(*leading_shape, 3, 256, 256)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        latents = self.infer(x, sample=True)
        mu_p_0 = self.decode(latents["z1"])
        return mu_p_0, latents

    def generate(self, z2: Tensor, lower_noise: Tensor) -> Tensor:
        """Sample p(z1 | z2) with explicit base noise, then decode its RGB mean."""
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        z1 = reparameterize_logvar(mu_p_1, v_p_1, noise=lower_noise)
        return self.decode(z1)

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        return self.generate(
            torch.randn(count, self.z2_dim, device=device),
            torch.randn(count, self.z1_dim, device=device),
        )


class LadderVAE(HierarchicalVAE):
    """Section 4: replace q_1 with q_hat_1 evidence and precision fusion.

    q_hat_1 predicts (mu_hat_q_1(x), v_hat_q_1(x)), the guide's hatted
    evidence parameters. Both are retained in the forward latent dictionary.
    Its shared mu_q_1/v_q_1 keys denote the guide's mu_q_1^L/v_q_1^L:
    the fused posterior used for sampling and KL. There is no learned q_1
    block in LadderVAE; the other distribution blocks match HierarchicalVAE.
    """

    posterior_family = "ladder_precision_fusion"

    def __init__(
        self,
        *,
        z1_dim: int = 96,
        z2_dim: int = 32,
        hidden_channels: int = 256,
        context_dim: int = 512,
    ) -> None:
        super().__init__(
            z1_dim=z1_dim,
            z2_dim=z2_dim,
            hidden_channels=hidden_channels,
            context_dim=context_dim,
        )
        del self.q_1
        # q-hat,1: h_1(x) -> evidence parameters; section 4.1, equation (4.2).
        self.q_hat_1 = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )

    def lower_distributions(
        self,
        h_1: Tensor,
        z2: Tensor,
    ) -> dict[str, Tensor]:
        """Section 4.4: p,1 + hatted q,1 evidence -> fused q,1 parameters."""
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        mu_hat_q_1, v_hat_q_1 = split_gaussian_parameters(
            self.q_hat_1(h_1)
        )
        # Gradients remain live through both the generative prior and the
        # bottom-up evidence path.
        # Section 4.2: these are mu_q_1^L(x, z2), v_q_1^L(x, z2).
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


class ActiveUnitAccumulator:
    """Dataset-window variance of top means and lower posterior corrections."""

    def __init__(self) -> None:
        self.count = 0
        self.lower_sum: Tensor | None = None
        self.lower_square_sum: Tensor | None = None
        self.top_sum: Tensor | None = None
        self.top_square_sum: Tensor | None = None

    def update(self, latents: dict[str, Tensor]) -> None:
        lower = (latents["mu_q_1"] - latents["mu_p_1"]).detach().double()
        top = latents["mu_q_2"].detach().double()
        if self.lower_sum is None:
            self.lower_sum = torch.zeros_like(lower[0])
            self.lower_square_sum = torch.zeros_like(lower[0])
            self.top_sum = torch.zeros_like(top[0])
            self.top_square_sum = torch.zeros_like(top[0])
        self.lower_sum += lower.sum(dim=0)
        self.lower_square_sum += lower.square().sum(dim=0)
        self.top_sum += top.sum(dim=0)
        self.top_square_sum += top.square().sum(dim=0)
        self.count += lower.shape[0]

    def counts(self, *, variance_threshold: float = 1e-2) -> tuple[int, int]:
        if self.count < 2 or self.lower_sum is None:
            return 0, 0
        lower_variance = (
            self.lower_square_sum / self.count - (self.lower_sum / self.count).square()
        )
        top_variance = (
            self.top_square_sum / self.count - (self.top_sum / self.count).square()
        )
        return (
            int((lower_variance > variance_threshold).sum()),
            int((top_variance > variance_threshold).sum()),
        )


def hierarchical_vae_loss(
    mu_p_0: Tensor,
    x: Tensor,
    latents: dict[str, Tensor],
    *,
    kl_weight: float,
    free_bits: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Summed RGB MSE + two KL terms, with warm-up and group-wise free bits.

    As in VAE/CVAE, observation variance is 1/2 and the Gaussian constant
    is omitted. Free bits clamp each layer's batch-mean KL, not each unit.
    Sections 3.2/4.3: distortion is mean(D_i), kl_z1 is mean(r_1(x_i, z2_i)),
    and kl_z2 is mean(R_2(x_i)). For Ladder, q,1 is the fused distribution.
    """
    # Layer 0: D_i = ||x_i - mu_p_0(z1_i)||^2, then average over the batch.
    distortion = (mu_p_0 - x).square().flatten(1).sum(dim=1).mean()
    # Layer 1: KL(q(z1 | z2, x) || p(z1 | z2)) at the same sampled z2.
    kl_z1 = (
        diagonal_gaussian_kl_from_logvar(
            latents["mu_q_1"],
            latents["v_q_1"],
            latents["mu_p_1"],
            latents["v_p_1"],
        )
        .sum(dim=1)
        .mean()
    )
    # Layer 2: KL(q(z2 | x) || N(0, I)); no learned p,2 parameters.
    kl_z2 = (
        diagonal_gaussian_kl_from_logvar(latents["mu_q_2"], latents["v_q_2"])
        .sum(dim=1)
        .mean()
    )
    threshold = distortion.new_tensor(float(free_bits))
    kl_objective_z1 = torch.maximum(kl_z1, threshold)
    kl_objective_z2 = torch.maximum(kl_z2, threshold)
    kl_objective = kl_objective_z1 + kl_objective_z2
    loss = distortion + float(kl_weight) * kl_objective
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
    "ActiveUnitAccumulator",
    "HierarchicalVAE",
    "LadderVAE",
    "hierarchical_vae_loss",
    "model_config",
]
