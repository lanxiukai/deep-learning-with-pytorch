"""HVAE and Ladder VAE from reading guide 3.3c, sections 3 and 4.

Layer numbers follow the generated variable: 0 = observation x, 1 = lower
latent z1, 2 = upper latent z2. Both models use the same generative chain::

    p_theta(x, z1, z2) = p(z2) p_theta(z1 | z2) p_theta(x | z1)

The guide's generative distributions map to these network blocks::

    Distribution      Network call  Distribution parameters
    p(z2)             none          fixed N(0, I)
    p_theta(z1 | z2)  self.p_1(z2)  mu_p_1, v_p_1
    p_theta(x | z1)   self.p_0(z1)  mu_p_0; fixed variance 1/2

Each learned latent Gaussian uses N(mu, diag(exp(v))), with v = log(sigma**2).
The latent networks predict means and log-variances; split_gaussian_parameters
separates their concatenated predictions and bounds the log-variance. q_2
performs this split internally and also returns the shared image feature h_1.
The p_0 decoder predicts only the RGB observation mean.

The classes below map the guide's inference distributions to their q_*
blocks. In both models, deterministic features flow upward from x, while
latent sampling proceeds downward: first z2, then z1, then decode mu_p_0.
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


class _TopPosterior(nn.Module):
    """Complete x -> q(z2 | x) network, retaining shared lower-layer features."""

    def __init__(
        self, hidden_channels: int, context_dim: int, z2_dim: int
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ImageEncoder(hidden_channels),
            nn.Linear(hidden_channels * 4 * 4, context_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z2_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return h_1(x), mu_q_2(x), v_q_2(x) from one image-encoder pass."""
        if x.shape[1:] != (3, 256, 256):
            raise ValueError("Expected RGB images from the 256x256 glasses cache")
        h_1 = self.features(x)
        mu_q_2, v_q_2 = split_gaussian_parameters(self.head(h_1))
        return h_1, mu_q_2, v_q_2


class HierarchicalVAE(nn.Module):
    """Infer z2 first, then predict z1 from the image and z2 (guide section 3).

    The guide's top-down inference factorization, equation (3.1), is::

        q_phi^down(z1, z2 | x) = q_phi^down(z2 | x) q_phi^down(z1 | z2, x)

    The complete generative (p) and inference (q) mapping is::

        Distribution            Network call         Parameters
        p(z2)                   none                 fixed N(0, I)
        p_theta(z1 | z2)        self.p_1(z2)         mu_p_1, v_p_1
        p_theta(x | z1)         self.p_0(z1)         mu_p_0; variance 1/2
        q_phi^down(z2 | x)      self.q_2(x)          mu_q_2, v_q_2
        q_phi^down(z1 | z2, x)  self.q_1([h_1, z2])  mu_q_1, v_q_1

    q_2 contains the full image encoder and upper posterior head. It also
    returns the intermediate feature h_1, reused by the lower posterior;
    [h_1, z2] means concatenation. No separate p_2 network is needed because
    the upper prior is fixed. p_0 predicts only the observation mean.
    The q_1 network predicts the lower posterior directly. The p_1 parameters
    supply the reference distribution for its conditional KL term.

    Read the training path as q_2(x) -> infer_from_top -> decode,
    matching the three steps in section 3.3. forward returns the decoded
    mean together with the latent samples and parameters needed by the loss.
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
        # q(z2 | x): encode x once and retain h_1 for the lower posterior.
        self.q_2 = _TopPosterior(hidden_channels, context_dim, z2_dim)

        # p(z1 | z2): the generative branch reads only the upper latent.
        self.p_1 = nn.Sequential(
            nn.Linear(z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        # q(z1 | z2, x): the lower posterior reads both image features and z2.
        self.q_1 = nn.Sequential(
            nn.Linear(context_dim + z2_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 2 * z1_dim),
        )
        # p(x | z1): decode the observation mean; variance is fixed at 1/2.
        self.p_0 = ImageDecoder(z1_dim, hidden_channels)

    def bottom_up(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Expose q_2(x) for inference and layer-intervention diagnostics.

        For x shaped (B, 3, 256, 256), return h_1 shaped (B, context_dim)
        and the upper mean/log-variance pair, each shaped (B, z2_dim).
        h_1 carries the image information used again by the lower posterior.
        """
        return self.q_2(x)

    def lower_distributions(
        self,
        h_1: Tensor,
        z2: Tensor,
    ) -> dict[str, Tensor]:
        """Predict p(z1 | z2) and q(z1 | z2, x) at the same supplied z2.

        h_1 carries the dependence on x. Each returned parameter has shape
        (B, z1_dim); sampling is deferred to infer_from_top.
        """
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
        """Complete lower inference for a supplied upper latent z2.

        Normally z2 is drawn from q(z2 | x); diagnostics can supply a mean
        or an intervention value. The supplied mu_q_2/v_q_2 remain the
        original upper posterior parameters used to evaluate its KL.
        sample_lower selects a q(z1 | z2, x) draw or its conditional mean.
        """
        parameters_1 = self.lower_distributions(h_1, z2)
        mu_q_1, v_q_1 = parameters_1["mu_q_1"], parameters_1["v_q_1"]
        z1 = reparameterize_logvar(mu_q_1, v_q_1) if sample_lower else mu_q_1
        return {
            # Layer 2: posterior parameters and the supplied upper latent.
            "mu_q_2": mu_q_2,
            "v_q_2": v_q_2,
            "z2": z2,
            # Layer 1: prior/posterior parameters and the inferred lower latent.
            **parameters_1,
            "z1": z1,
            "h_1": h_1,
        }

    def infer(self, x: Tensor, *, sample: bool = True) -> dict[str, Tensor]:
        """Run the top-down posterior path and retain its latent parameters.

        sample=True uses reparameterized draws at both layers. sample=False
        follows successive conditional means for deterministic diagnostics;
        this mean path does not integrate over the upper posterior.
        """
        # Step 1: predict q(z2 | x), then choose z2.
        h_1, mu_q_2, v_q_2 = self.q_2(x)
        z2 = reparameterize_logvar(mu_q_2, v_q_2) if sample else mu_q_2
        # Step 2: construct the lower distributions, then choose z1 from q.
        return self.infer_from_top(
            h_1,
            mu_q_2,
            v_q_2,
            z2,
            sample_lower=sample,
        )

    def decode(self, z1: Tensor) -> Tensor:
        """Decode mu_p_0(z1), the mean of p(x | z1), without observation noise.

        Preserve any leading dimensions: (..., z1_dim) -> (..., 3, 256, 256).
        """
        leading_shape = z1.shape[:-1]
        flat_z1 = z1.reshape(-1, self.z1_dim)
        mu_p_0 = self.p_0(flat_z1)
        return mu_p_0.reshape(*leading_shape, 3, 256, 256)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return (mu_p_0, latents) for one stochastic reconstruction pass.

        mu_p_0 has the input image shape (B, 3, 256, 256). latents contains
        z1/z2, the mu_q_2/v_q_2, mu_p_1/v_p_1 and mu_q_1/v_q_1 pairs, and h_1.
        Ladder also retains mu_hat_q_1/v_hat_q_1 before fusion. The loss uses
        the posterior/prior pairs and compares mu_p_0 with the real image x.
        """
        latents = self.infer(x, sample=True)
        mu_p_0 = self.decode(latents["z1"])
        return mu_p_0, latents

    def generate(self, z2: Tensor, lower_noise: Tensor) -> Tensor:
        """Generate an RGB mean from supplied z2 and standard-normal lower noise.

        Use z1 = mu_p_1(z2) + exp(v_p_1(z2) / 2) * lower_noise, then decode.
        This path uses only p_1 and p_0; inference and Ladder fusion are unused.
        """
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        z1 = reparameterize_logvar(mu_p_1, v_p_1, noise=lower_noise)
        return self.decode(z1)

    def sample(self, count: int, *, device: torch.device) -> Tensor:
        """Draw z2 from N(0, I), draw z1 from p(z1 | z2), and return RGB means."""
        return self.generate(
            torch.randn(count, self.z2_dim, device=device),
            torch.randn(count, self.z1_dim, device=device),
        )


class LadderVAE(HierarchicalVAE):
    """Fuse image evidence with p(z1 | z2) to infer z1 (guide section 4).

    Keep the inherited p_0, p_1 and complete q_2 image encoder. Replace the
    direct q_1 network with evidence and fusion, equations (4.1)-(4.3).
    The complete distribution-to-network mapping is::

        Distribution or factor       Implementation     Parameters
        p(z2)                        none               fixed N(0, I)
        p_theta(z1 | z2)             self.p_1(z2)       mu_p_1, v_p_1
        p_theta(x | z1)              self.p_0(z1)       mu_p_0; variance 1/2
        q_phi^down(z2 | x)           self.q_2(x)        mu_q_2, v_q_2
        q_tilde_phi(z1 | x)          self.q_hat_1(h_1)  mu_hat_q_1, v_hat_q_1
        q_{theta,phi}^L(z1 | z2, x)  precision fusion   mu_q_1, v_q_1

    q_2(x) also returns h_1 for the evidence network. The p distributions have
    the same roles as in HierarchicalVAE: p(z2) is fixed, p_1 predicts the
    lower prior, and p_0 predicts the observation mean with fixed variance 1/2.

    q_tilde_phi is the learned Gaussian evidence factor. Its parameters carry
    hats in the guide; it contributes to fusion without being sampled itself.
    The lower posterior is the normalized product::

        q_{theta,phi}^L(z1 | z2, x) proportional to
            p_theta(z1 | z2) * q_tilde_phi(z1 | x)

    Fusion computes the guide's mu_q_1^L/v_q_1^L. The returned dictionary uses
    mu_q_1/v_q_1 so the inherited sampling path and loss consume this posterior.
    There is no learned fusion network. Gradients flow through both p_1 and
    q_hat_1, while q(z2 | x) is used directly without fusion with N(0, I).
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
        # q_tilde(z1 | x): image evidence for fusion, replacing the direct q_1.
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
        """Return the lower prior, image evidence, and fused posterior parameters.

        All six tensors have shape (B, z1_dim). The mu_q_1/v_q_1 pair is the
        fused q^L(z1 | z2, x) used to sample z1 and compute the lower KL.
        """
        mu_p_1, v_p_1 = split_gaussian_parameters(self.p_1(z2))
        mu_hat_q_1, v_hat_q_1 = split_gaussian_parameters(
            self.q_hat_1(h_1)
        )
        # Add precisions and precision-weight the means (guide equation 4.3).
        # Keep both inputs attached: reconstruction gradients also train p_1.
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
    """Combine reconstruction and layer KL terms (guide equations 3.4 and 4.5).

    The returned metrics map to the guide as follows::

        distortion   mean(D_i)              summed RGB squared error
        kl_z1        mean(r_1(x_i, z2_i))   lower conditional KL
        kl_z2        mean(R_2(x_i))         upper KL against N(0, I)

    Observation variance is 1/2, with the Gaussian constant omitted. Each
    term sums over its coordinates before averaging over the batch. Ladder
    uses its fused lower posterior in kl_z1; its evidence has no separate KL.

    Free bits clamp each layer's batch-mean KL, then kl_weight scales their
    sum. The returned distortion/kl_z1/kl_z2 metrics remain unweighted and
    unclamped, and all returned metrics are detached from the training graph.
    """
    # Layer 0: compare the decoded observation mean with the real image.
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
    # Layer 2: KL(q(z2 | x) || N(0, I)).
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
    "HierarchicalVAE",
    "LadderVAE",
    "hierarchical_vae_loss",
    "model_config",
]
