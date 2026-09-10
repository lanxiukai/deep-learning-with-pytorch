"""Repeatable within-project GAN comparisons using ImageNet features.

These torchvision Inception metrics are not interchangeable with published
TensorFlow FID or KID scores. Real and generated images share preprocessing;
reference images and latent seeds stay fixed across compared checkpoints.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from dl_utils.data.celeba import CelebAAlignedDataset, aligned_celeba_transform
from dl_utils.vae.image_quality import (
    FeatureMoments,
    TorchvisionInceptionFeatures,
    frechet_distance,
)


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
