"""256x256 RGB conditional VAE and glasses-dataset lesson utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from dl_utils.data.glasses import GLASSES_CLASS_NAMES, GLASSES_IMAGE_SIZE
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.gan.inference import generate_in_batches, make_fixed_class_latent_grid
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.images import save_image_row_grid
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.image_networks import ImageDecoder, ImageEncoder
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
    split_gaussian_parameters,
)

CVAE_OBJECTIVE = "summed_rgb_mse_plus_conditional_kl"


class ConditionalDecoder(ImageDecoder):
    """Decode a latent and class representation into an RGB Gaussian mean."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_channels: int,
    ) -> None:
        super().__init__(latent_dim + condition_dim, hidden_channels)

    def forward(self, z: Tensor, condition: Tensor) -> Tensor:
        # z: (B, latent_dim), condition: (B, condition_dim)
        return super().forward(torch.cat((z, condition), dim=1))


class ConditionalVAE(nn.Module):
    """Class-conditional VAE with explicit prior, posterior, and decoder APIs."""

    def __init__(
        self,
        *,
        num_classes: int = 2,
        latent_dim: int = 128,
        condition_dim: int = 32,
        hidden_channels: int = 256,
        posterior_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if hidden_channels < 128 or hidden_channels % 128:
            raise ValueError("hidden_channels must be a positive multiple of 128")
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels
        self.posterior_hidden_dim = posterior_hidden_dim

        # The posterior, conditional prior, and decoder share the class embedding.
        self.condition_embedding = nn.Embedding(num_classes, condition_dim)
        self.image_encoder = ImageEncoder(hidden_channels)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 4 + condition_dim, posterior_hidden_dim),
            nn.SiLU(),
            nn.Linear(posterior_hidden_dim, 2 * latent_dim),
        )  # (B, hidden_channels * 4 * 4 + condition_dim) -> (B, 2 * latent_dim)
        self.prior_network = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2 * latent_dim),
        )  # (B, condition_dim) -> (B, 2 * latent_dim)
        self.decoder = ConditionalDecoder(latent_dim, condition_dim, hidden_channels)

    def prior(self, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return p(z | c) parameters predicted from the class condition."""
        condition = self.condition_embedding(labels)  # (B, condition_dim)
        return split_gaussian_parameters(
            self.prior_network(condition)
        )  # p_mu, p_logvar: (B, latent_dim)

    def encode(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return q(z | x, c) parameters; this is the only target-aware API."""
        # images: (B, 3, 256, 256), labels: (B,)
        if images.shape[1:] != (3, GLASSES_IMAGE_SIZE, GLASSES_IMAGE_SIZE):
            raise ValueError(
                f"Expected RGB images from the {GLASSES_IMAGE_SIZE}x"
                f"{GLASSES_IMAGE_SIZE} glasses cache"
            )
        condition = self.condition_embedding(labels)  # (B, condition_dim)
        features = self.image_encoder(images)  # (B, hidden_channels * 4 * 4)
        return split_gaussian_parameters(
            self.posterior(torch.cat((features, condition), dim=1))
        )  # q_mu, q_logvar (B, latent_dim)

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        """Return the fixed-scale Gaussian mean for p(x | z, c), in [0, 1]."""
        # z: (B, latent_dim), labels: (B,)
        condition = self.condition_embedding(labels)  # (B, condition_dim)
        return self.decoder(z, condition)  # (B, 3, 256, 256)

    def generate(self, labels: Tensor, *, noise: Tensor | None = None) -> Tensor:
        """Sample z from p(z | c), then decode it with the requested class."""
        # The prior p(z | c) is the latent reference for the KL term and generation.
        p_mu, p_logvar = self.prior(labels)
        z = reparameterize_logvar(p_mu, p_logvar, noise=noise)
        return self.decode(z, labels)  # (B, 3, 256, 256)

    def forward(
        self, images: Tensor, labels: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # images: (B, 3, 256, 256), labels: (B,)
        # mu, logvar: (B, latent_dim)
        q_mu, q_logvar = self.encode(images, labels)
        p_mu, p_logvar = self.prior(labels)
        z = reparameterize_logvar(q_mu, q_logvar)  # (B, latent_dim)
        reconstruction = self.decode(z, labels)  # (B, 3, 256, 256)
        return reconstruction, {
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
        }  # reconstruction, statistics


def conditional_vae_loss(
    reconstruction: Tensor,
    real_images: Tensor,
    statistics: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return summed RGB MSE + KL(q || p), omitting the Gaussian constant.

    The fixed observation variance is 1/2, giving a unit MSE coefficient,
    as in the introductory face VAE. Both terms are averaged over images.
    """
    distortion = (reconstruction - real_images).square().flatten(1).sum(dim=1).mean()
    rate = (
        diagonal_gaussian_kl_from_logvar(
            statistics["q_mu"],
            statistics["q_logvar"],
            statistics["p_mu"],
            statistics["p_logvar"],
        )
        .sum(dim=1)
        .mean()
    )
    return distortion + rate, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
    }


@torch.inference_mode()
def evaluate_cvae(
    model: ConditionalVAE,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate conditional-ELBO metrics with sample-count-weighted means."""
    model.eval()
    metrics = MetricAccumulator(("loss", "distortion", "rate"), device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction, statistics = model(images, labels)
        loss, terms = conditional_vae_loss(
            reconstruction,
            images,
            statistics,
        )
        metrics.update(
            (
                loss,
                terms["distortion"],
                terms["rate"],
            ),
            num_examples=images.shape[0],
        )
    return metrics.compute()


@torch.inference_mode()
def save_conditional_samples(
    model: ConditionalVAE,
    path: Path,
    *,
    device: torch.device,
    samples_per_class: int = 8,
    noise: Tensor | None = None,
    class_names: tuple[str, ...] = GLASSES_CLASS_NAMES,
) -> None:
    """Save G then NoG rows, sharing base noise across corresponding columns.

    The conditional prior transforms the noise separately for each class;
    this comparison does not assume identity preservation.
    """
    noise, labels = make_fixed_class_latent_grid(
        tuple(range(model.num_classes)),
        samples_per_class,
        model.latent_dim,
        device,
        base_noise=noise,
    )
    samples = generate_in_batches(
        (noise, labels),
        samples_per_class,
        lambda base_noise, conditions: model.generate(conditions, noise=base_noise),
        module=model,
    )
    # The shared renderer accepts [-1, 1]; CVAE probabilities stay in [0, 1].
    save_image_row_grid(
        samples.mul(2).sub(1).split(samples_per_class),
        class_names,
        path,
        title="Conditional prior samples",
        column_labels=[
            f"Shared noise {index + 1}" for index in range(samples_per_class)
        ],
        dpi=200,
    )


@torch.inference_mode()
def save_conditional_reconstructions(
    model: ConditionalVAE,
    dataset: ImageFolder,
    path: Path,
    *,
    device: torch.device,
    samples_per_class: int,
) -> None:
    """Save balanced class rows of original images and posterior-mean decodes."""
    for_class = [
        [index for index, target in enumerate(dataset.targets) if target == label]
        for label in range(model.num_classes)
    ]
    count = min(samples_per_class, *(len(indices) for indices in for_class))
    indices = [index for group in for_class for index in group[:count]]
    display_loader = make_device_aware_loader(
        Subset(dataset, indices),
        count,
        device,
        shuffle=False,
    )

    def reconstruct(images: Tensor, labels: Tensor) -> Tensor:
        mu, _ = model.encode(images, labels)
        return model.decode(mu, labels)

    rows = []
    row_labels = []
    for name, (images, labels) in zip(dataset.classes, display_loader, strict=True):
        reconstructions = generate_in_batches(
            (images.to(device), labels.to(device)),
            count,
            reconstruct,
            module=model,
        )
        rows.extend((images.mul(2).sub(1), reconstructions.mul(2).sub(1)))
        row_labels.extend((f"{name}\nOriginal", f"{name}\nRecon."))
    save_image_row_grid(
        rows, row_labels, path, title="Posterior-mean reconstruction", dpi=200
    )


def save_conditional_metric_summary(metrics: dict[str, float], path: Path) -> None:
    """Save the conditional-ELBO metric summary."""
    with plt.ioff():
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(
            ("MSE + KL", "Summed MSE", "Conditional KL"),
            (metrics["loss"], metrics["distortion"], metrics["rate"]),
            color=("#4c78a8", "#f58518", "#54a24b"),
        )
        axis.set_title("Training-set MSE + conditional KL")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)


__all__ = [
    "CVAE_OBJECTIVE",
    "ConditionalVAE",
    "conditional_vae_loss",
    "evaluate_cvae",
    "save_conditional_metric_summary",
    "save_conditional_reconstructions",
    "save_conditional_samples",
]
