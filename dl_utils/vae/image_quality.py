"""Small feature-distribution metrics shared by tokenizer evaluations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import Inception_V3_Weights, inception_v3


class ConditionalImageSampler(Protocol):
    """Interface shared by evaluated discrete generative systems."""

    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor,
        temperature: float,
    ) -> Tensor: ...


def structural_similarity_index(
    prediction: Tensor,
    target: Tensor,
    *,
    value_range: float = 2.0,
    window_size: int = 7,
) -> Tensor:
    """Return mean local SSIM for equally shaped image batches.

    The implementation uses a uniform local window so the lesson scripts do
    not need an additional image-metrics dependency.  It is intended for
    comparisons made with this exact protocol, not as a drop-in replacement
    for every SSIM package and preprocessing convention.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching [B,C,H,W] shapes")
    if value_range <= 0:
        raise ValueError("value_range must be positive")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    covariance = (
        F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding)
        - mean_prediction * mean_target
    )
    prediction_variance = (
        F.avg_pool2d(prediction.square(), window_size, stride=1, padding=padding)
        - mean_prediction.square()
    )
    target_variance = (
        F.avg_pool2d(target.square(), window_size, stride=1, padding=padding)
        - mean_target.square()
    )
    prediction_variance = prediction_variance.clamp_min(0.0)
    target_variance = target_variance.clamp_min(0.0)
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    score = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (prediction_variance + target_variance + c2)
        )
    )
    return score.flatten(1).mean(dim=1).mean()


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


@torch.inference_mode()
def collect_reference_feature_moments(
    loader: Iterable[tuple[Tensor, Tensor]],
    feature_extractor: nn.Module,
    *,
    feature_dim: int,
    reconstruction_examples: int,
    generation_examples: int,
    device: torch.device,
) -> tuple[FeatureMoments, FeatureMoments]:
    """Collect the two real-image references shared by tokenizer evaluations."""
    reconstruction_reference = FeatureMoments(feature_dim)
    generation_reference = FeatureMoments(feature_dim)
    examples = 0
    generation_count = 0
    for images, _ in loader:
        remaining = reconstruction_examples - examples
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        features = feature_extractor(images)
        reconstruction_reference.update(features)
        if generation_count < generation_examples:
            count = min(images.shape[0], generation_examples - generation_count)
            generation_reference.update(features[:count])
            generation_count += count
        examples += images.shape[0]
    return reconstruction_reference, generation_reference


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


@torch.inference_mode()
def evaluate_conditional_generation(
    system: ConditionalImageSampler,
    feature_extractor: nn.Module,
    real_moments: FeatureMoments,
    *,
    examples: int,
    batch_size: int,
    num_classes: int,
    temperature: float,
    device: torch.device,
    saved_examples: int = 64,
) -> tuple[dict[str, float], Tensor]:
    """Sample balanced classes and compare their feature distribution."""
    if min(examples, batch_size, num_classes, saved_examples) < 1:
        raise ValueError("sample counts, batch size, and num_classes must be positive")
    generated_moments = FeatureMoments(real_moments.feature_dim)
    sample_batches: list[Tensor] = []
    saved = 0
    generated = 0
    while generated < examples:
        count = min(batch_size, examples - generated)
        labels = torch.arange(
            generated,
            generated + count,
            device=device,
        ).remainder(num_classes)
        images = system.sample(
            count,
            device=device,
            labels=labels,
            temperature=temperature,
        )
        generated_moments.update(feature_extractor(images))
        if saved < saved_examples:
            save_count = min(saved_examples - saved, images.shape[0])
            sample_batches.append(images[:save_count].cpu())
            saved += save_count
        generated += count
    return {
        "projected_inception_frechet": frechet_distance(
            real_moments, generated_moments
        ),
        "temperature": temperature,
    }, torch.cat(sample_batches)


__all__ = [
    "ConditionalImageSampler",
    "FeatureMoments",
    "TorchvisionInceptionFeatures",
    "collect_reference_feature_moments",
    "evaluate_conditional_generation",
    "frechet_distance",
    "structural_similarity_index",
]
