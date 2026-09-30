"""KL first-stage autoencoder and VGG loss for latent-diffusion lessons.

Shared RGB encoder/decoder blocks are reused from vae.perceptual_autoencoder.
Training objectives and optimizer updates remain in the extension scripts.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from dl_utils.vae.perceptual_autoencoder import PerceptualDecoder, PerceptualEncoder
from dl_utils.vae.vae_common import reparameterize_logvar


class KLPerceptualAutoencoder(nn.Module):
    """Continuous KL-regularized autoencoder used before latent diffusion."""

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        hidden_channels: int = 192,
        downsample_steps: int = 3,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        if image_size % (2**downsample_steps):
            raise ValueError(
                "Image size must be divisible by the encoder scale factor."
            )
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.latent_size = image_size // (2**downsample_steps)
        self.encoder = PerceptualEncoder(
            2 * latent_channels, hidden_channels, downsample_steps
        )
        self.decoder = PerceptualDecoder(
            latent_channels, hidden_channels, downsample_steps
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "Image shape does not match the first-stage configuration."
            )
        mu, logvar = self.encoder(images).chunk(2, dim=1)
        return mu, logvar.clamp(-12.0, 12.0)

    def encode_latent(
        self,
        images: Tensor,
        *,
        sample: bool = True,
        latent_scale: float = 1.0,
    ) -> Tensor:
        """Encode the downstream latent and apply one checkpoint-level scale."""
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        mu, logvar = self.encode(images)
        z = reparameterize_logvar(mu, logvar) if sample else mu
        return z * latent_scale

    def decode_latent(self, z: Tensor, *, latent_scale: float = 1.0) -> Tensor:
        if latent_scale <= 0:
            raise ValueError("latent_scale must be positive")
        return self.decoder(z / latent_scale)

    def reconstruct(self, images: Tensor) -> Tensor:
        return self.decode_latent(
            self.encode_latent(images, sample=False), latent_scale=1.0
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(images)
        z = reparameterize_logvar(mu, logvar)
        return self.decoder(z), mu, logvar, z


class VGGPerceptualLoss(nn.Module):
    """Frozen ImageNet-VGG16 multi-layer feature L1 distance."""

    mean: Tensor
    std: Tensor

    def __init__(self) -> None:
        super().__init__()
        from torchvision import models

        features = cast(
            nn.Sequential, models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        )
        self.blocks = nn.ModuleList([features[:4], features[4:9], features[9:16]])
        self.eval().requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(
        self, prediction: Tensor, target: Tensor, *, reduction: str = "mean"
    ) -> Tensor:
        prediction = (prediction.add(1.0).mul(0.5) - self.mean) / self.std
        target = (target.add(1.0).mul(0.5) - self.mean) / self.std
        per_sample = prediction.new_zeros(prediction.shape[0])
        for block in self.blocks:
            prediction = block(prediction)
            target = block(target)
            per_sample = per_sample + (prediction - target).abs().flatten(1).mean(1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'none' or 'mean'")


__all__ = ["KLPerceptualAutoencoder", "VGGPerceptualLoss"]
