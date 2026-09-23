"""Feature-distribution metrics for comparisons under a fixed feature protocol."""

from __future__ import annotations

import torch
from torch import Tensor


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
        (first_kernel.sum() - first_kernel.diagonal().sum())
        / (first_count * (first_count - 1))
        + (second_kernel.sum() - second_kernel.diagonal().sum())
        / (second_count * (second_count - 1))
        - 2 * cross_kernel.mean()
    )


def feature_precision_recall(real, fake, *, neighbors=3, chunk_size=256):
    """k-NN manifold estimates: precision measures fidelity, recall coverage.

    Each ball uses its own manifold center's k-th *other* neighbor radius.
    Chunking only bounds distance-matrix memory; it does not approximate k-NN.
    """
    if min(len(real), len(fake)) <= neighbors:
        raise ValueError("Precision/recall needs more samples than neighbors.")

    def radii(features):
        distances = []
        for start in range(0, len(features), chunk_size):
            block = features[start : start + chunk_size]
            matrix = torch.cdist(block, features)
            matrix[
                torch.arange(len(block)), torch.arange(start, start + len(block))
            ] = float("inf")
            distances.append(matrix.topk(neighbors, largest=False).values[:, -1])
        return torch.cat(distances)

    def coverage(queries, centers, radius):
        inside = []
        for block in queries.split(chunk_size):
            inside.append((torch.cdist(block, centers) <= radius[None]).any(dim=1))
        return torch.cat(inside).float().mean().item()

    return coverage(fake, real, radii(real)), coverage(real, fake, radii(fake))


def feature_metrics(real, fake, *, seed=20260909, kid_subsets=20, neighbors=3):
    """Compute distribution metrics; unbiased KID estimates may be negative."""
    real, fake = real.cpu().float(), fake.cpu().float()
    moments = []
    for features in (real, fake):
        value = FeatureMoments(features.shape[1])
        value.update(features)
        moments.append(value)
    rng = torch.Generator().manual_seed(seed)
    subset_size = min(512, len(real), len(fake))
    kid = torch.tensor(
        [
            polynomial_mmd(
                real[torch.randperm(len(real), generator=rng)[:subset_size]],
                fake[torch.randperm(len(fake), generator=rng)[:subset_size]],
            )
            for _ in range(kid_subsets)
        ],
        dtype=torch.float64,
    )
    precision, recall = feature_precision_recall(real, fake, neighbors=neighbors)
    return {
        "torchvision_fid": frechet_distance(*moments),
        "torchvision_kid_mean": kid.mean().item(),
        "torchvision_kid_subset_std": kid.std(unbiased=False).item(),
        "feature_precision": precision,
        "feature_recall": recall,
        "feature_dim": real.shape[1],
        "kid_subsets": kid_subsets,
        "kid_subset_size": subset_size,
        "manifold_neighbors": neighbors,
    }


__all__ = [
    "FeatureMoments",
    "feature_metrics",
    "feature_precision_recall",
    "frechet_distance",
    "polynomial_mmd",
]
