"""Compact 4x4-to-128x128 ProGAN models for progressive-growing study.

The module keeps the paper's equalized learning rate, pixel normalization,
minibatch-standard-deviation feature, filtered resolution transitions, and
real/fake fade-in paths without reproducing its large training system.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.gan.stylegan_layers import (
    RESOLUTIONS,
    EqualizedConv2d,
    EqualizedLinear,
    MinibatchStandardDeviation,
    PixelNorm,
    denormalize,
    filtered_downsample2d,
    filtered_upsample2d,
    make_channel_map,
    validate_alpha,
    validate_resolution,
)


class GeneratorInputBlock(nn.Module):
    """Turn a latent vector into the first normalized 4x4 feature map."""

    def __init__(self, z_dim, channels):
        super().__init__()
        self.pixel_norm = PixelNorm()
        self.linear = EqualizedLinear(
            z_dim,
            channels * 4 * 4,
            gain=math.sqrt(2),
        )
        self.convolution = EqualizedConv2d(
            channels,
            channels,
            3,
            padding=1,
        )

    def forward(self, z):
        # z shape: (B, z_dim)
        hidden = self.pixel_norm(z)
        hidden = self.linear(hidden).view(z.shape[0], -1, 4, 4)  # (B, C, 4, 4)
        hidden = self.pixel_norm(F.leaky_relu(hidden, 0.2))
        hidden = self.convolution(hidden)  # (B, C, 4, 4)
        return self.pixel_norm(F.leaky_relu(hidden, 0.2))


class GeneratorBlock(nn.Module):
    """Add one filtered upsampling stage to a progressive generator."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
        )
        self.conv2 = EqualizedConv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
        )
        self.pixel_norm = PixelNorm()

    def forward(self, inputs):
        # inputs shape: (B, C_in, H, W)
        hidden = filtered_upsample2d(inputs)  # (B, C_in, 2H, 2W)
        hidden = self.pixel_norm(
            F.leaky_relu(self.conv1(hidden), 0.2)
        )  # (B, C_out, 2H, 2W)
        hidden = self.pixel_norm(
            F.leaky_relu(self.conv2(hidden), 0.2)
        )  # (B, C_out, 2H, 2W)
        return hidden


class ProGANGenerator(nn.Module):
    """Progressively expose generator blocks from 4x4 through 128x128."""

    def __init__(self, z_dim=128, base_channels=32):
        super().__init__()
        self.z_dim = int(z_dim)
        self.base_channels = int(base_channels)
        self.resolutions = RESOLUTIONS
        self.channels = make_channel_map(self.base_channels, self.resolutions)
        self.input_block = GeneratorInputBlock(
            self.z_dim,
            self.channels[4],
        )  # (B, z_dim) -> (B, channels[4], 4, 4)
        self.blocks = nn.ModuleDict(
            {
                str(resolution): GeneratorBlock(
                    self.channels[resolution // 2],
                    self.channels[resolution],
                )
                for resolution in self.resolutions[1:]
            }
        )  # Each block: (B, channels[R/2], R/2, R/2) -> (B, channels[R], R, R)
        self.to_rgbs = nn.ModuleDict(
            {
                str(resolution): EqualizedConv2d(
                    self.channels[resolution],
                    3,
                    1,
                    gain=1.0,
                )
                for resolution in self.resolutions
            }
        )  # Each block: (B, channels[R], R, R) -> (B, 3, R, R)

    def forward(self, z, resolution=128, alpha=1.0):
        # z shape: (B, z_dim)
        resolution = validate_resolution(resolution, self.resolutions)
        alpha = validate_alpha(alpha)

        hidden = self.input_block(z)  # (B, channels[4], 4, 4)
        if resolution == 4:
            return self.to_rgbs["4"](hidden)  # (B, 3, 4, 4)

        for current_resolution in self.resolutions[1:]:
            previous = hidden
            # (B, channels[R/2], R/2, R/2) -> (B, channels[R], R, R)
            hidden = self.blocks[str(current_resolution)](hidden)
            if current_resolution == resolution:
                break

        # (B, channels[R], R, R) -> (B, 3, R, R)
        new_rgb = self.to_rgbs[str(resolution)](hidden)
        if alpha == 1.0:
            return new_rgb
        # (B, channels[R/2], R/2, R/2) -> (B, 3, R/2, R/2)
        old_rgb = self.to_rgbs[str(resolution // 2)](previous)
        old_rgb = filtered_upsample2d(old_rgb)  # (B, 3, R, R)
        # RGB image-space fade-in blending:
        # (1 - alpha) * old_rgb + alpha * new_rgb
        return torch.lerp(old_rgb, new_rgb, alpha)


class DiscriminatorBlock(nn.Module):
    """Reduce one progressive discriminator stage with filtered sampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = EqualizedConv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
        )
        self.conv2 = EqualizedConv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
        )

    def forward(self, inputs):
        # inputs shape: (B, C_in, H, W)
        hidden = F.leaky_relu(self.conv1(inputs), 0.2)  # (B, C_in, H, W)
        hidden = F.leaky_relu(self.conv2(hidden), 0.2)  # (B, C_out, H, W)
        return filtered_downsample2d(hidden)  # (B, C_out, H/2, W/2)


class ProGANDiscriminator(nn.Module):
    """Mirror the active ProGAN generator resolution and fade-in state."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = int(base_channels)
        self.resolutions = RESOLUTIONS
        self.channels = make_channel_map(self.base_channels, self.resolutions)
        self.from_rgbs = nn.ModuleDict(
            {
                str(resolution): EqualizedConv2d(
                    3,
                    self.channels[resolution],
                    1,
                )
                for resolution in self.resolutions
            }
        )  # Each block: (B, 3, R, R) -> (B, channels[R], R, R)
        self.blocks = nn.ModuleDict(
            {
                str(resolution): DiscriminatorBlock(
                    self.channels[resolution],
                    self.channels[resolution // 2],
                )
                for resolution in self.resolutions[1:]
            }
        )  # Each block: (B, channels[R], R, R) -> (B, channels[R/2], R/2, R/2)
        final_channels = self.channels[4]
        self.minibatch_std = MinibatchStandardDeviation()
        self.final_conv = EqualizedConv2d(
            final_channels + 1,
            final_channels,
            3,
            padding=1,
        )  # (B, final_channels + 1, 4, 4) -> (B, final_channels, 4, 4)
        self.final_linear = EqualizedLinear(
            final_channels * 4 * 4,
            final_channels,
            gain=math.sqrt(2),
        )  # flatten (B, final_channels * 4 * 4) -> (B, final_channels)
        self.output = EqualizedLinear(final_channels, 1)  # (B, 1)

    def forward(self, images, resolution=128, alpha=1.0):
        # images shape: (B, 3, R, R)
        resolution = validate_resolution(resolution, self.resolutions)
        alpha = validate_alpha(alpha)
        expected_shape = (resolution, resolution)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected RGB images in BCHW format.")
        if images.shape[-2:] != expected_shape:
            raise ValueError(
                f"expected {resolution}x{resolution} images, "
                f"got {images.shape[-2]}x{images.shape[-1]}"
            )

        current_index = self.resolutions.index(resolution)
        if resolution == 4:
            # (B, 3, 4, 4) -> (B, channels[4], 4, 4)
            hidden = F.leaky_relu(self.from_rgbs["4"](images), 0.2)
        else:
            new_hidden = F.leaky_relu(
                self.from_rgbs[str(resolution)](images),
                0.2,
            )  # (B, 3, R, R) -> (B, channels[R], R, R)
            # (B, channels[R/2], R/2, R/2)
            new_hidden = self.blocks[str(resolution)](new_hidden)
            # (B, 3, R, R) -> (B, 3, R/2, R/2)
            old_images = filtered_downsample2d(images)
            old_hidden = F.leaky_relu(
                self.from_rgbs[str(resolution // 2)](old_images),
                0.2,
            )  # (B, 3, R/2, R/2) -> (B, channels[R/2], R/2, R/2)
            # feature-space fade-in blending:
            # (1 - alpha) * old_hidden + alpha * new_hidden
            hidden = torch.lerp(old_hidden, new_hidden, alpha)

            for index in range(current_index - 1, 0, -1):
                block_resolution = self.resolutions[index]
                hidden = self.blocks[str(block_resolution)](hidden)  # downsampling

        # (B, channels[4], 4, 4) -> (B, channels[4] + 1, 4, 4)
        hidden = self.minibatch_std(hidden)
        hidden = F.leaky_relu(self.final_conv(hidden), 0.2)  # (B, channels[4], 4, 4)
        hidden = hidden.flatten(1)  # (B, channels[4] * 4 * 4)
        hidden = F.leaky_relu(self.final_linear(hidden), 0.2)  # (B, channels[4])
        return self.output(hidden).squeeze(1)  # (B, 1) -> (B,)


__all__ = [
    "DiscriminatorBlock",
    "GeneratorBlock",
    "GeneratorInputBlock",
    "ProGANDiscriminator",
    "ProGANGenerator",
    "denormalize",
]
