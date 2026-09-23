"""Repeatable within-project GAN comparisons using ImageNet features.

These torchvision Inception metrics are not interchangeable with published
TensorFlow FID or KID scores. Real and generated images share preprocessing;
reference images and latent seeds stay fixed across compared checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import Inception_V3_Weights, inception_v3

from dl_utils.data.celeba import CelebAAlignedDataset, aligned_celeba_transform


class TorchvisionInceptionFeatures(nn.Module):
    """ImageNet Inception-v3 pool features for transparent Fréchet proxies.

    This is not the TensorFlow FID implementation.  Results are comparable
    only when every model uses this exact preprocessing and feature extractor.
    Set projection_dim=None to retain all 2048 pool features.
    """

    mean: Tensor
    std: Tensor
    projection: Tensor | None

    def __init__(
        self,
        *,
        projection_dim: int | None = 256,
        projection_seed: int = 2026,
    ) -> None:
        super().__init__()
        model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        model.add_module("fc", nn.Identity())
        self.model = model.eval().requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )
        generator = torch.Generator().manual_seed(projection_seed)
        projection = (
            None
            if projection_dim is None
            else (
                torch.randn(2048, projection_dim, generator=generator)
                / projection_dim**0.5
            )
        )
        self.feature_dim = 2048 if projection_dim is None else projection_dim
        self.register_buffer("projection", projection)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Inception input must have shape [B, 3, H, W]")
        images = images.mul(0.5).add(0.5).clamp(0, 1)
        images = F.interpolate(
            images,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        features = self.model((images - self.mean) / self.std)
        return features if self.projection is None else features @ self.projection


class FeatureMoments:
    """Streaming mean and unbiased covariance without storing all features."""

    def __init__(self, feature_dim: int) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.count = 0
        self.total = torch.zeros(feature_dim, dtype=torch.float64)
        self.outer_total = torch.zeros(feature_dim, feature_dim, dtype=torch.float64)

    def update(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features have the wrong shape")
        features = features.detach().to(device="cpu", dtype=torch.float64)
        self.count += features.shape[0]
        self.total += features.sum(dim=0)
        self.outer_total += features.transpose(0, 1) @ features

    def statistics(self) -> tuple[Tensor, Tensor]:
        if self.count < 2:
            raise ValueError("at least two feature vectors are required")
        mean = self.total / self.count
        covariance = (self.outer_total - self.count * torch.outer(mean, mean)) / (
            self.count - 1
        )
        return mean, covariance


def _symmetric_matrix_square_root(matrix: Tensor) -> Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    return (
        eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)


def frechet_distance(
    first: FeatureMoments,
    second: FeatureMoments,
    *,
    covariance_epsilon: float = 1e-6,
) -> float:
    """Fréchet distance between two empirical Gaussian feature fits."""
    if first.feature_dim != second.feature_dim:
        raise ValueError("feature dimensions must match")
    mean_a, covariance_a = first.statistics()
    mean_b, covariance_b = second.statistics()
    identity = torch.eye(first.feature_dim, dtype=torch.float64)
    covariance_a = covariance_a + covariance_epsilon * identity
    covariance_b = covariance_b + covariance_epsilon * identity
    root_a = _symmetric_matrix_square_root(covariance_a)
    middle = root_a @ covariance_b @ root_a
    trace_root = (
        torch.linalg.eigvalsh(0.5 * (middle + middle.transpose(0, 1)))
        .clamp_min(0)
        .sqrt()
        .sum()
    )
    difference = mean_a - mean_b
    distance = (
        difference.dot(difference)
        + torch.trace(covariance_a)
        + torch.trace(covariance_b)
        - 2.0 * trace_root
    )
    return float(distance.clamp_min(0))


def polynomial_mmd(first: torch.Tensor, second: torch.Tensor) -> float:
    """Unbiased squared MMD with the degree-three KID polynomial kernel."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("Expected feature matrices with matching dimensions.")
    first_count, second_count = len(first), len(second)
    if min(first_count, second_count) < 2:
        raise ValueError("MMD requires at least two samples per distribution.")
    first, second = first.double(), second.double()
    dim = first.shape[1]
    first_kernel = (first @ first.T / dim + 1).pow(3)
    second_kernel = (second @ second.T / dim + 1).pow(3)
    cross_kernel = (first @ second.T / dim + 1).pow(3)
    return float(
        (first_kernel.sum() - first_kernel.diagonal().sum()) / (first_count * (first_count - 1))
        + (second_kernel.sum() - second_kernel.diagonal().sum()) / (second_count * (second_count - 1))
        - 2 * cross_kernel.mean()
    )


class GenerationQualityEvaluator:
    """Evaluate EMA generators against a fixed, unaugmented CelebA split."""

    def __init__(
        self,
        data_dir,
        *,
        device,
        examples=2048,
        batch_size=32,
        seed=20260906,
        split="validation",
        feature_extractor=None,
        generator_kwargs=None,
    ):
        if examples < 2 or batch_size < 1:
            raise ValueError(
                "Evaluation needs at least two examples and a positive batch."
            )
        self.device = device
        self.examples = examples
        self.batch_size = batch_size
        self.seed = seed
        self.split = split
        self.generator_kwargs = dict(generator_kwargs or {})
        if feature_extractor is None:
            # Model construction must not perturb the GAN training RNG stream.
            with torch.random.fork_rng(devices=[device]):
                feature_extractor = TorchvisionInceptionFeatures(projection_dim=None)
        self.features = feature_extractor.to(device).eval()
        rng = torch.Generator().manual_seed(seed)
        self.projection = torch.randn(2048, 256, generator=rng) / 256**0.5
        dataset = CelebAAlignedDataset(
            data_dir,
            split=split,
            transform=aligned_celeba_transform(128),
        )
        if examples > len(dataset):
            raise ValueError("Requested evaluation exceeds the selected CelebA split.")
        indices = torch.randperm(len(dataset), generator=rng)[:examples].tolist()
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        with torch.inference_mode():
            self.real = torch.cat(
                [self.features(images.to(device)).cpu() for images, _ in loader]
            )
        self.real_moments = self.moments(self.real)

    def moments(self, features):
        moments = FeatureMoments(256)
        moments.update(features @ self.projection)
        return moments

    @torch.inference_mode()
    def evaluate(self, generator) -> tuple[dict, torch.Tensor]:
        rng = torch.Generator().manual_seed(self.seed)
        latents = torch.randn(self.examples, generator.z_dim, generator=rng)
        batches, samples = [], []
        was_training = generator.training
        generator.eval()
        try:
            for start in range(0, self.examples, self.batch_size):
                images = generator(
                    latents[start : start + self.batch_size].to(self.device),
                    **self.generator_kwargs,
                )
                if start < 64:
                    samples.append(images[: 64 - start].cpu())
                batches.append(self.features(images).cpu())
        finally:
            generator.train(was_training)
        generated = torch.cat(batches)
        subset_size = min(512, self.examples)
        estimates = []
        for _ in range(20):
            real_indices = torch.randperm(self.examples, generator=rng)[:subset_size]
            fake_indices = torch.randperm(self.examples, generator=rng)[:subset_size]
            estimates.append(
                polynomial_mmd(self.real[real_indices], generated[fake_indices])
            )
        estimates = torch.tensor(estimates, dtype=torch.float64)
        result = {
            "torchvision_inception_kid_mean": estimates.mean().item(),
            "torchvision_inception_kid_subset_std": estimates.std().item(),
            "projected_inception_frechet_256": frechet_distance(
                self.real_moments,
                self.moments(generated),
            ),
            "generated_feature_variance": generated.var(dim=0).mean().item(),
            "examples": self.examples,
            "seed": self.seed,
            "split": self.split,
            "kid_subsets": 20,
            "kid_subset_size": subset_size,
            "feature_extractor": "torchvision Inception_V3_Weights.IMAGENET1K_V1 pool3",
            "preprocessing": "178px center crop to 128px; clamp to [0,1]; bilinear antialiased 299px; ImageNet normalization",
            "scope": "within-project comparison; not published TensorFlow FID/KID",
            "generator_kwargs": self.generator_kwargs,
        }
        if not all(
            torch.isfinite(torch.tensor(result[key]))
            for key in (
                "torchvision_inception_kid_mean",
                "projected_inception_frechet_256",
                "generated_feature_variance",
            )
        ):
            raise FloatingPointError("Non-finite generation quality metrics.")
        return result, torch.cat(samples)
