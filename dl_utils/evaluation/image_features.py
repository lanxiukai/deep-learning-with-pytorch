"""ImageNet Inception features with explicit optional random projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import Inception_V3_Weights, inception_v3


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


__all__ = ["TorchvisionInceptionFeatures"]
