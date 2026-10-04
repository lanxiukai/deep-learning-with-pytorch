"""Independent continuous KL codec and perceptual/adversarial building blocks."""

import torch
from torch import nn


def groups(channels):
    return next(g for g in range(min(32, channels), 0, -1) if channels % g == 0)


class ImageResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class KLPerceptualAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        latent_channels=4,
        hidden_channels=128,
        downsample_steps=3,
        image_size=256,
    ):
        super().__init__()
        self.latent_channels, self.image_size = latent_channels, image_size
        self.latent_size = image_size // 2**downsample_steps
        self._config = {
            "latent_channels": latent_channels,
            "hidden_channels": hidden_channels,
            "downsample_steps": downsample_steps,
            "image_size": image_size,
        }
        encoder = [nn.Conv2d(3, hidden_channels, 3, padding=1)]
        for _ in range(downsample_steps):
            encoder.extend(
                [
                    ImageResidual(hidden_channels),
                    nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
                ]
            )
        encoder.extend(
            [
                ImageResidual(hidden_channels),
                nn.GroupNorm(groups(hidden_channels), hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 2 * latent_channels, 3, padding=1),
            ]
        )
        self.encoder = nn.Sequential(*encoder)
        decoder = [nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)]
        for _ in range(downsample_steps):
            decoder.extend(
                [
                    ImageResidual(hidden_channels),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                ]
            )
        decoder.extend(
            [
                ImageResidual(hidden_channels),
                nn.GroupNorm(groups(hidden_channels), hidden_channels),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 3, 3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*decoder)

    def config(self):
        return self._config

    def encode(self, images):
        mu, logvar = self.encoder(images).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 12)

    def encode_latent(self, images, *, sample=True, latent_scale=1.0):
        mu, logvar = self.encode(images)
        return (
            mu + (0.5 * logvar).exp() * torch.randn_like(mu) if sample else mu
        ) * latent_scale

    def decode_latent(self, z, *, latent_scale=1.0):
        return self.decoder(z / latent_scale)

    def reconstruct(self, images):
        return self.decoder(self.encode(images)[0])

    def forward(self, images):
        mu, logvar = self.encode(images)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, logvar, z


class PatchDiscriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(width, 2 * width, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * width, 4 * width, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * width, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class PerceptualLoss(nn.Module):
    """Learned LPIPS; frozen weights, but gradients still reach the reconstruction."""

    def __init__(self):
        super().__init__()
        import lpips

        self.metric = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)

    def forward(self, prediction, target):
        return self.metric(prediction.float(), target.float()).flatten(1).mean(1)
