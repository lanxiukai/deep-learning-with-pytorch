"""Reusable two-level Gaussian hierarchy for the HVAE and Ladder lessons."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    fuse_diagonal_gaussians,
    reparameterize_logvar,
    split_gaussian_parameters,
)


class HierarchicalVAE32(nn.Module):
    """Two-level q(z2 | x) q(z1 | z2, x) teaching baseline."""

    posterior_family = "conditional_network"

    def __init__(
        self,
        *,
        z1_dim: int = 24,
        z2_dim: int = 12,
        hidden_channels: int = 64,
        context_dim: int = 192,
    ) -> None:
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.hidden_channels = hidden_channels
        self.context_dim = context_dim
        self.bottom_up_image = nn.Sequential(
            nn.Conv2d(1, hidden_channels // 2, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.bottom_up_lower = nn.Sequential(
            nn.Linear(hidden_channels * 2 * 4 * 4, context_dim),
            nn.SiLU(),
        )
        self.bottom_up_top = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )
        self.top_posterior = nn.Linear(context_dim, 2 * z2_dim)
        self.lower_prior = nn.Sequential(
            nn.Linear(z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        self.lower_posterior = nn.Sequential(
            nn.Linear(context_dim + z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(z1_dim, hidden_channels * 2 * 4 * 4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (hidden_channels * 2, 4, 4)),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden_channels // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden_channels // 2, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def bottom_up(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        lower_evidence = self.bottom_up_lower(self.bottom_up_image(images))
        top_evidence = self.bottom_up_top(lower_evidence)
        q2_mu, q2_logvar = split_gaussian_parameters(self.top_posterior(top_evidence))
        return lower_evidence, q2_mu, q2_logvar

    def lower_distributions(
        self,
        lower_evidence: Tensor,
        z2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return p(z1 | z2) and q(z1 | z2, x) parameters."""
        p1_mu, p1_logvar = split_gaussian_parameters(self.lower_prior(z2))
        q1_mu, q1_logvar = split_gaussian_parameters(
            self.lower_posterior(torch.cat((lower_evidence, z2), dim=1))
        )
        return p1_mu, p1_logvar, q1_mu, q1_logvar

    def infer_from_top(
        self,
        lower_evidence: Tensor,
        q2_mu: Tensor,
        q2_logvar: Tensor,
        z2: Tensor,
        *,
        sample_lower: bool,
    ) -> dict[str, Tensor]:
        p1_mu, p1_logvar, q1_mu, q1_logvar = self.lower_distributions(
            lower_evidence, z2
        )
        z1 = reparameterize_logvar(q1_mu, q1_logvar) if sample_lower else q1_mu
        return {
            "z1": z1,
            "z2": z2,
            "q1_mu": q1_mu,
            "q1_logvar": q1_logvar,
            "p1_mu": p1_mu,
            "p1_logvar": p1_logvar,
            "q2_mu": q2_mu,
            "q2_logvar": q2_logvar,
            "p2_mu": torch.zeros_like(q2_mu),
            "p2_logvar": torch.zeros_like(q2_logvar),
            "lower_evidence": lower_evidence,
        }

    def infer(self, images: Tensor, *, sample: bool = True) -> dict[str, Tensor]:
        lower_evidence, q2_mu, q2_logvar = self.bottom_up(images)
        z2 = reparameterize_logvar(q2_mu, q2_logvar) if sample else q2_mu
        return self.infer_from_top(
            lower_evidence,
            q2_mu,
            q2_logvar,
            z2,
            sample_lower=sample,
        )

    def decode(self, z1: Tensor) -> Tensor:
        leading_shape = z1.shape[:-1]
        flat_z1 = z1.reshape(-1, self.z1_dim)
        images = self.decoder(self.decoder_input(flat_z1))
        return images.reshape(*leading_shape, 1, 32, 32)

    def forward(self, images: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        latents = self.infer(images, sample=True)
        return self.decode(latents["z1"]), latents

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        z2 = torch.randn(count, self.z2_dim, device=device)
        p1_mu, p1_logvar = split_gaussian_parameters(self.lower_prior(z2))
        z1 = reparameterize_logvar(p1_mu, p1_logvar)
        return self.decode(z1)


class LadderVAE32(HierarchicalVAE32):
    """Replace the lower conditional network with prior-evidence fusion."""

    posterior_family = "ladder_precision_fusion"

    def __init__(
        self,
        *,
        z1_dim: int = 24,
        z2_dim: int = 12,
        hidden_channels: int = 64,
        context_dim: int = 192,
    ) -> None:
        super().__init__(
            z1_dim=z1_dim,
            z2_dim=z2_dim,
            hidden_channels=hidden_channels,
            context_dim=context_dim,
        )
        del self.lower_posterior
        self.lower_evidence_distribution = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )

    def lower_distributions(
        self,
        lower_evidence: Tensor,
        z2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p1_mu, p1_logvar = split_gaussian_parameters(self.lower_prior(z2))
        evidence_mu, evidence_logvar = split_gaussian_parameters(
            self.lower_evidence_distribution(lower_evidence)
        )
        # Gradients remain live through both the generative prior and the
        # bottom-up evidence path.
        q1_mu, q1_logvar = fuse_diagonal_gaussians(
            p1_mu,
            p1_logvar,
            evidence_mu,
            evidence_logvar,
        )
        return p1_mu, p1_logvar, q1_mu, q1_logvar


class ActiveUnitAccumulator:
    """Dataset-window variance of top means and lower posterior corrections."""

    def __init__(self) -> None:
        self.count = 0
        self.lower_sum: Tensor | None = None
        self.lower_square_sum: Tensor | None = None
        self.top_sum: Tensor | None = None
        self.top_square_sum: Tensor | None = None

    def update(self, latents: dict[str, Tensor]) -> None:
        lower = (latents["q1_mu"] - latents["p1_mu"]).detach().double()
        top = latents["q2_mu"].detach().double()
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
    reconstruction: Tensor,
    target: Tensor,
    latents: dict[str, Tensor],
    *,
    kl_weight: float,
    free_bits: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Negative two-level ELBO with group-wise free bits after batch means."""
    distortion = (
        F.binary_cross_entropy(reconstruction, target, reduction="none")
        .flatten(1)
        .sum(dim=1)
        .mean()
    )
    kl_z1 = (
        diagonal_gaussian_kl_from_logvar(
            latents["q1_mu"],
            latents["q1_logvar"],
            latents["p1_mu"],
            latents["p1_logvar"],
        )
        .sum(dim=1)
        .mean()
    )
    kl_z2 = (
        diagonal_gaussian_kl_from_logvar(latents["q2_mu"], latents["q2_logvar"])
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
    model: HierarchicalVAE32,
) -> dict[str, int]:
    return {
        "z1_dim": model.z1_dim,
        "z2_dim": model.z2_dim,
        "hidden_channels": model.hidden_channels,
        "context_dim": model.context_dim,
    }


__all__ = [
    "ActiveUnitAccumulator",
    "HierarchicalVAE32",
    "LadderVAE32",
    "hierarchical_vae_loss",
    "model_config",
]
