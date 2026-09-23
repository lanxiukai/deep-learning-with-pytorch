"""Shared network blocks for VQGAN and KL-autoencoder lessons.

Only architectures and small loss primitives live here.  Each lesson keeps
its optimizer phases and algorithm-specific reduction visible in the script.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from dl_utils.vae.quantization import VectorQuantizer
from dl_utils.vae.vae_common import reparameterize_logvar


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.net(inputs)


class PerceptualEncoder(nn.Module):
    """Residual image encoder with configurable spatial compression."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 128,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        if hidden_channels < 2 or downsample_steps < 2:
            raise ValueError(
                "hidden_channels and downsample_steps must be at least two"
            )
        groups = min(32, hidden_channels)
        while hidden_channels % groups != 0:
            groups -= 1
        layers: list[nn.Module] = [
            nn.Conv2d(3, hidden_channels // 2, 4, 2, 1),
            nn.SiLU(inplace=True),
        ]
        in_channels = hidden_channels // 2
        for step in range(1, downsample_steps):
            layers.append(nn.Conv2d(in_channels, hidden_channels, 4, 2, 1))
            if step < downsample_steps - 1:
                layers.append(nn.SiLU(inplace=True))
            in_channels = hidden_channels
        layers.extend(
            [
                ResidualBlock(hidden_channels),
                ResidualBlock(hidden_channels),
                nn.GroupNorm(groups, hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        return self.net(images)


class PerceptualDecoder(nn.Module):
    """Mirror a ``PerceptualEncoder`` and reconstruct RGB in [-1, 1]."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        if hidden_channels < 2 or downsample_steps < 2:
            raise ValueError(
                "hidden_channels and downsample_steps must be at least two"
            )
        groups = min(32, hidden_channels)
        while hidden_channels % groups != 0:
            groups -= 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        ]
        for _ in range(downsample_steps - 2):
            layers.extend(
                [
                    nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
                    nn.SiLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels // 2, 3, 4, 2, 1),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    @property
    def last_layer(self) -> nn.Parameter:
        return cast(nn.Parameter, cast(nn.ConvTranspose2d, self.net[-2]).weight)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class VQPerceptualAutoencoder(nn.Module):
    """VQGAN first-stage model without hiding its loss in the module."""

    def __init__(
        self,
        *,
        latent_channels: int = 64,
        codebook_size: int = 512,
        hidden_channels: int = 128,
        commitment: float = 0.25,
        downsample_steps: int = 2,
    ) -> None:
        super().__init__()
        self.downsample_steps = downsample_steps
        self.encoder = PerceptualEncoder(
            latent_channels, hidden_channels, downsample_steps
        )
        self.quantizer = VectorQuantizer(codebook_size, latent_channels, commitment)
        self.decoder = PerceptualDecoder(
            latent_channels, hidden_channels, downsample_steps
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encoder(images))

    def encode_indices(self, images: Tensor) -> Tensor:
        """Encode images and return only their discrete token grid."""
        return self.encode(images)[1]

    def decode_indices(self, indices: Tensor) -> Tensor:
        return self.decoder(self.quantizer.lookup(indices))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        z_st, indices, quantizer_loss, diagnostics = self.encode(images)
        return self.decoder(z_st), indices, quantizer_loss, diagnostics


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
        return self.decode_latent(self.encode_latent(images, sample=False), latent_scale=1.0)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(images)
        z = reparameterize_logvar(mu, logvar)
        return self.decoder(z), mu, logvar, z


class PatchDiscriminator(nn.Module):
    """Small PatchGAN returning a spatial grid of local real/fake logits."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> list[nn.Module]:
            return [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.features = nn.ModuleList(
            [
                nn.Sequential(*block(3, base_channels)),
                nn.Sequential(*block(base_channels, base_channels * 2)),
                nn.Sequential(*block(base_channels * 2, base_channels * 4)),
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 3, 1, 1),
        )

    def forward(self, images: Tensor) -> Tensor:
        for block in self.features:
            images = block(images)
        return self.head(images)


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


class LPIPSPerceptualLoss(nn.Module):
    """Frozen, learned LPIPS v0.1 distance for inputs in [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        import lpips

        self.metric = lpips.LPIPS(
            net="vgg",
            version="0.1",
            lpips=True,
            pretrained=True,
            pnet_rand=False,
            pnet_tune=False,
            eval_mode=True,
            verbose=False,
        )
        self.eval().requires_grad_(False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.metric(prediction, target, normalize=False).mean()


def adaptive_adversarial_weight(
    base_loss: Tensor,
    adversarial_loss: Tensor,
    last_layer: nn.Parameter,
    *,
    scale: float = 1.0,
    maximum: float = 1e4,
) -> Tensor:
    """Match last-decoder-layer gradient norms and stop the ratio's gradient."""
    base_gradient = torch.autograd.grad(base_loss, last_layer, retain_graph=True)[0]
    adversarial_gradient = torch.autograd.grad(
        adversarial_loss, last_layer, retain_graph=True
    )[0]
    ratio = base_gradient.norm() / (adversarial_gradient.norm() + 1e-4)
    return (float(scale) * ratio.clamp(0.0, maximum)).detach()


__all__ = [
    "KLPerceptualAutoencoder",
    "LPIPSPerceptualLoss",
    "PatchDiscriminator",
    "PerceptualDecoder",
    "PerceptualEncoder",
    "ResidualBlock",
    "VGGPerceptualLoss",
    "VQPerceptualAutoencoder",
    "adaptive_adversarial_weight",
]
