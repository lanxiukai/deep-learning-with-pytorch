"""64x64 and 128x128 SN-GAN models and spectral-normalization teaching tools.

The network uses a 128-dimensional latent vector, a 64-channel ResNet stem,
categorical conditional BatchNorm in the generator,
and a projection discriminator. The generator is intentionally not spectrally
normalized; that extension is introduced by the following SAGAN lesson.
"""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


def generator_block_resolutions(image_size):
    """Return the upsampling stages for the two teaching resolutions."""
    if image_size not in (64, 128):
        raise ValueError("image_size must be 64 or 128.")
    return tuple(size for size in (8, 16, 32, 64, 128) if size <= image_size)


def discriminator_block_resolutions(image_size):
    """Reverse the generator stages, ending with one 4x4 residual block."""
    return tuple(
        size // 2 for size in reversed(generator_block_resolutions(image_size))
    ) + (4,)


def generator_channels(base_channels, image_size):
    """Scale the channel hierarchy with the requested output resolution."""
    if base_channels < 1:
        raise ValueError("base_channels must be positive.")
    stages = len(generator_block_resolutions(image_size))
    return tuple(
        base_channels * 2 ** min(stages - 1, stages - index)
        for index in range(stages + 1)
    )


def discriminator_channels(base_channels, image_size):
    """Use the generator channel hierarchy in reverse."""
    return tuple(reversed(generator_channels(base_channels, image_size)))


def _maybe_spectral_norm(module, enabled):
    return spectral_norm(module) if enabled else module


def _initialize_affine_layer(module, gain=1.0):
    """Initialize one affine layer, including a spectral-norm wrapped layer."""
    weight = getattr(module, "weight_orig", None)
    if weight is None:
        weight = module.weight
    nn.init.xavier_uniform_(weight, gain=gain)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _initialize_resnet_weights(module):
    """Initialize non-residual maps and normalization affine parameters."""
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        _initialize_affine_layer(module)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _initialize_residual_branches(*blocks):
    """Use the reference implementation's sqrt(2) residual-branch gain."""
    for block in blocks:
        _initialize_affine_layer(block.conv1, gain=math.sqrt(2))
        _initialize_affine_layer(block.conv2, gain=math.sqrt(2))


def validate_class_labels(labels, batch_size):
    """Validate the shape and dtype shared by conditional GAN forwards."""
    if labels.ndim != 1 or labels.shape[0] != batch_size:
        raise ValueError("labels must have shape (batch,).")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long.")


class CategoricalConditionalBatchNorm2d(nn.Module):
    """BatchNorm with one learned gain and bias vector per class."""

    def __init__(self, channels, num_classes):
        super().__init__()
        self.normalization = nn.BatchNorm2d(channels, affine=False)
        self.gain = nn.Embedding(num_classes, channels)
        self.bias = nn.Embedding(num_classes, channels)
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, inputs, labels):
        gain = self.gain(labels)[:, :, None, None]
        bias = self.bias(labels)[:, :, None, None]
        return self.normalization(inputs) * gain + bias


class SNGeneratorResidualBlock(nn.Module):
    """Conditional pre-activation residual block with 2x upsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        *,
        spectral_normalization=False,
    ):
        super().__init__()
        self.norm1 = CategoricalConditionalBatchNorm2d(
            in_channels,
            num_classes,
        )
        self.norm2 = CategoricalConditionalBatchNorm2d(
            out_channels,
            num_classes,
        )
        self.conv1 = _maybe_spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            spectral_normalization,
        )
        self.conv2 = _maybe_spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            spectral_normalization,
        )
        self.skip = _maybe_spectral_norm(
            nn.Conv2d(in_channels, out_channels, 1),
            spectral_normalization,
        )

    def forward(self, inputs, labels):
        residual = F.interpolate(inputs, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        hidden = F.relu(self.norm1(inputs, labels), inplace=True)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(self.norm2(hidden, labels), inplace=True))
        return hidden + residual


class SNGenerator(nn.Module):
    """Projection SN-GAN's class-conditional image generator."""

    def __init__(
        self,
        z_dim=128,
        num_classes=2,
        base_channels=64,
        image_size=64,
    ):
        super().__init__()
        if z_dim < 1 or base_channels < 1 or num_classes < 1:
            raise ValueError("z_dim, base_channels, and num_classes must be positive.")
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.image_size = image_size
        channels = generator_channels(base_channels, image_size)
        self.input = nn.Linear(z_dim, channels[0] * 4 * 4)
        blocks = [
            SNGeneratorResidualBlock(
                in_channels,
                out_channels,
                num_classes,
            )
            for in_channels, out_channels in pairwise(channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.output_norm = nn.BatchNorm2d(channels[-1])
        self.output = nn.Conv2d(channels[-1], 3, 3, padding=1)
        self.apply(_initialize_resnet_weights)
        _initialize_residual_branches(*self.blocks)
        for block in blocks:
            nn.init.ones_(block.norm1.gain.weight)
            nn.init.zeros_(block.norm1.bias.weight)
            nn.init.ones_(block.norm2.gain.weight)
            nn.init.zeros_(block.norm2.bias.weight)

    def forward(self, noise, labels):
        if noise.ndim != 2 or noise.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected noise with shape (batch, {self.z_dim}), "
                f"got {tuple(noise.shape)}."
            )
        validate_class_labels(labels, noise.shape[0])

        hidden = self.input(noise).view(noise.shape[0], -1, 4, 4)
        for block in self.blocks:
            hidden = block(hidden, labels)
        hidden = F.relu(self.output_norm(hidden), inplace=True)
        return torch.tanh(self.output(hidden))


class SNDiscriminatorResidualBlock(nn.Module):
    """Spectral-normalized residual block with optional downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        first=False,
        downsample=True,
        wide=False,
    ):
        super().__init__()
        self.first = first
        self.downsample = downsample
        hidden_channels = out_channels if first or wide else in_channels
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )
        needs_projection = in_channels != out_channels or downsample
        self.skip = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if needs_projection
            else nn.Identity()
        )

    def forward(self, inputs):
        hidden = inputs if self.first else F.relu(inputs, inplace=False)
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.relu(hidden, inplace=True))
        if self.downsample:
            hidden = F.avg_pool2d(hidden, 2)

        residual = self.skip(inputs)
        if self.downsample:
            residual = F.avg_pool2d(residual, 2)
        return hidden + residual


class SNDiscriminator(nn.Module):
    """Spectral-normalized projection discriminator."""

    def __init__(
        self,
        num_classes=2,
        base_channels=64,
        image_size=64,
    ):
        super().__init__()
        if base_channels < 1 or num_classes < 1:
            raise ValueError("base_channels and num_classes must be positive.")
        self.num_classes = num_classes
        self.image_size = image_size
        channels = discriminator_channels(base_channels, image_size)
        self.blocks = nn.ModuleList(
            [SNDiscriminatorResidualBlock(3, channels[0], first=True)]
            + [
                SNDiscriminatorResidualBlock(in_channels, out_channels)
                for in_channels, out_channels in pairwise(channels[:-1])
            ]
            + [
                SNDiscriminatorResidualBlock(
                    channels[-2],
                    channels[-1],
                    downsample=False,
                )
            ]
        )
        self.feature_channels = channels[-1]
        self.output = spectral_norm(nn.Linear(self.feature_channels, 1))
        self.class_embedding = spectral_norm(
            nn.Embedding(num_classes, self.feature_channels)
        )
        self.apply(_initialize_resnet_weights)
        _initialize_residual_branches(*self.blocks)

    def extract_features(self, images):
        if (
            images.ndim != 4
            or images.shape[1] != 3
            or tuple(images.shape[-2:]) != (self.image_size, self.image_size)
        ):
            raise ValueError(
                "Expected images with shape "
                f"(batch, 3, {self.image_size}, {self.image_size}), "
                f"got {tuple(images.shape)}."
            )
        hidden = images
        for block in self.blocks:
            hidden = block(hidden)
        return F.relu(hidden, inplace=True).sum(dim=(2, 3))

    def forward(self, images, labels):
        validate_class_labels(labels, images.shape[0])
        features = self.extract_features(images)
        unconditional = self.output(features).squeeze(1)
        projection = (self.class_embedding(labels) * features).sum(dim=1)
        return unconditional + projection


__all__ = [
    "CategoricalConditionalBatchNorm2d",
    "SNDiscriminator",
    "SNDiscriminatorResidualBlock",
    "SNGenerator",
    "SNGeneratorResidualBlock",
    "discriminator_block_resolutions",
    "discriminator_channels",
    "generator_block_resolutions",
    "generator_channels",
    "validate_class_labels",
]
