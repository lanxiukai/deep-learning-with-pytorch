"""Compact 128x128 StyleGAN2 without progressive growing or AdaIN.

The generator keeps weight modulation/demodulation, per-layer stochastic
noise, skip-connected RGB outputs, style mixing, truncation, and the original
overlapping W-slot indexing. The discriminator keeps residual filtered
downsampling and a minibatch-standard-deviation feature.
"""

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.gan.stylegan_layers import (
    RESOLUTIONS,
    EqualizedConv2d,
    EqualizedLinear,
    MappingNetwork,
    MinibatchStandardDeviation,
    NoiseInjection,
    denormalize,
    filtered_downsample2d,
    filtered_upsample2d,
    make_channel_map,
    validate_resolution,
)


class ModulatedConv2d(nn.Module):
    """Apply per-sample weight modulation and optional demodulation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        *,
        demodulate=True,
        upsample=False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.style_dim = int(style_dim)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.demodulate = bool(demodulate)
        self.upsample = bool(upsample)
        self.padding = self.kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(
                1,
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )  # (1, Cout, Cin, K, K)
        self.weight_scale = 1 / math.sqrt(
            self.in_channels * self.kernel_size * self.kernel_size
        )  # ()
        self.modulation = EqualizedLinear(
            self.style_dim,
            self.in_channels,
            bias_init=1.0,
        )  # (B, style_dim) → (B, Cin)

    def forward(self, inputs, style):
        batch, channels, height, width = inputs.shape
        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {channels}"
            )
        style_scale = self.modulation(style).view(
            batch,
            1,
            self.in_channels,
            1,
            1,
        )  # (B, Cin) -> (B, 1, Cin, 1, 1)
        weight = self.weight * self.weight_scale * style_scale  # (B, Cout, Cin, K, K)
        if self.demodulate:
            demodulation = torch.rsqrt(
                weight.square().sum(dim=(2, 3, 4)) + 1e-8
            )  # (B, Cout)
            weight = weight * demodulation.view(
                batch,
                self.out_channels,
                1,
                1,
                1,
            )  # (B, Cout, Cin, K, K)
        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )  # (B × Cout, Cin, K, K)

        if self.upsample:
            inputs = filtered_upsample2d(inputs)
            height, width = height * 2, width * 2
        grouped_inputs = inputs.reshape(
            1,
            batch * self.in_channels,
            height,
            width,
        )  # (B, Cin, H', W') -> (1, B × Cin, H', W')
        outputs = F.conv2d(
            grouped_inputs,
            weight,
            padding=self.padding,
            groups=batch,
        )  # (1, B × Cout, H', W')
        return outputs.view(
            batch,
            self.out_channels,
            outputs.shape[-2],
            outputs.shape[-1],
        )  # (B, Cout, H', W')


class StyledConv(nn.Module):
    """StyleGAN2 modulated convolution followed by noise and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        resolution,
        fixed_noise_seed,
        *,
        upsample=False,
    ):
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels,
            out_channels,
            3,
            style_dim,
            demodulate=True,
            upsample=upsample,
        )
        self.noise = NoiseInjection(
            out_channels,
            resolution,
            fixed_noise_seed,
            per_channel=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, inputs, style, noise_mode="random"):
        # inputs: (B, Cin, H, W), style: (B, style_dim)
        hidden = self.convolution(inputs, style)  # (B, Cout, H', W')
        hidden = self.noise(hidden, noise_mode)  # (B, Cout, H', W')
        # (B, Cout, H', W')
        bias = self.bias.to(dtype=hidden.dtype)
        return F.leaky_relu(hidden + bias, 0.2) * math.sqrt(2)


class ToRGB(nn.Module):
    """Convert features to RGB with modulation but no demodulation."""

    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.convolution = ModulatedConv2d(
            in_channels,
            3,
            1,
            style_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, inputs, style):
        # inputs: (B, Cin, H, W), style: (B, style_dim)
        outputs = self.convolution(inputs, style)
        return outputs + self.bias.to(dtype=outputs.dtype)  # (B, 3, H, W)


class SynthesisBlock(nn.Module):
    """One StyleGAN2 resolution block plus a skip-connected RGB output."""

    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        resolution,
        fixed_noise_seed,
        *,
        first=False,
    ):
        super().__init__()
        self.first = bool(first)
        self.num_ws = 2 if self.first else 3
        self.conv1 = StyledConv(
            in_channels,
            out_channels,
            style_dim,
            resolution,
            fixed_noise_seed,
            upsample=not self.first,
        )  # (B, Cin, 4, 4) → (B, Cout, 4, 4) if self.first
        # else (B, Cin, R/2, R/2) → (B, Cout, R, R)
        self.conv2 = (
            None
            if self.first
            else StyledConv(
                out_channels,
                out_channels,
                style_dim,
                resolution,
                fixed_noise_seed + 1,
            )
        )  # (B, Cout, R, R) → (B, Cout, R, R)
        # (B, Cout, R, R) → (B, 3, R, R)
        self.to_rgb = ToRGB(out_channels, style_dim)

    def forward(self, inputs, styles, old_rgb, noise_mode="random"):
        # inputs: (B, Cin, H, W), styles: (B, num_ws, style_dim)
        if styles.ndim != 3 or styles.shape[1] != self.num_ws:
            raise ValueError(
                f"expected {self.num_ws} W inputs for this synthesis block."
            )
        # (B, Cin, 4, 4) -> (B, Cout, 4, 4) if self.first
        # else (B, Cin, R/2, R/2) -> (B, Cout, R, R)
        hidden = self.conv1(inputs, styles[:, 0], noise_mode)
        if self.first:
            rgb_style = styles[:, 1]
        else:
            # (B, Cout, H, W) if self.first else (B, Cout, R, R)
            hidden = self.conv2(hidden, styles[:, 1], noise_mode)
            rgb_style = styles[:, 2]
        new_rgb = self.to_rgb(hidden, rgb_style)  # (B, 3, R, R)
        if old_rgb is not None:
            new_rgb = new_rgb + filtered_upsample2d(old_rgb)
        # hidden: (B, Cout, R, R), new_rgb: (B, 3, R, R)
        return hidden, new_rgb


class StyleGenerator(nn.Module):
    """Full-resolution StyleGAN2 generator with skip RGB connections."""

    def __init__(
        self,
        z_dim=128,
        style_dim=128,
        base_channels=32,
        mapping_layers=8,
        w_avg_beta=0.995,
    ):
        super().__init__()
        if not 0.0 <= w_avg_beta < 1.0:
            raise ValueError("w_avg_beta must be in [0, 1).")
        self.z_dim = int(z_dim)
        self.style_dim = int(style_dim)
        self.base_channels = int(base_channels)
        self.w_avg_beta = float(w_avg_beta)
        self.mapping = MappingNetwork(
            self.z_dim,
            self.style_dim,
            mapping_layers,
        )
        self.register_buffer("w_avg", torch.zeros(self.style_dim))
        self.resolutions = RESOLUTIONS
        # Every new resolution contributes two W slots. ToRGB consumes the
        # next slot, which is also reused by the following block's first conv.
        self.ws_increment_per_block = 2
        self.num_ws = len(self.resolutions) * self.ws_increment_per_block
        channels = make_channel_map(self.base_channels, self.resolutions)
        self.constant = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.blocks = nn.ModuleList(
            [
                SynthesisBlock(
                    channels[resolution // 2] if resolution > 4 else channels[4],
                    channels[resolution],
                    self.style_dim,
                    resolution,
                    fixed_noise_seed=max(0, 2 * index - 1),
                    first=resolution == 4,
                )
                for index, resolution in enumerate(self.resolutions)
            ]
        )

    def num_ws_for_resolution(self, resolution):
        """Return the cumulative W slots through one synthesis resolution."""
        resolution = validate_resolution(resolution, self.resolutions)
        return (self.resolutions.index(resolution) + 1) * self.ws_increment_per_block

    def make_ws(
        self,
        z,
        mixing_z=None,
        mixing_cutoff=None,
        *,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
        first_w = self.mapping(z)  # (B, style_dim)
        if update_w_avg:
            with torch.no_grad():
                batch_average = first_w.detach().float().mean(dim=0)
                # w_avg <- w_avg + (1 - w_avg_beta) * (batch_average - w_avg)
                self.w_avg.lerp_(
                    batch_average.to(self.w_avg),
                    1.0 - self.w_avg_beta,
                )
        ws = first_w[:, None, :].repeat(1, self.num_ws, 1)  # (B, num_ws, style_dim)

        if mixing_z is not None:
            mixing_w = self.mapping(mixing_z)  # (B, style_dim)
            if mixing_cutoff is None:
                mixing_cutoff = random.randint(1, self.num_ws - 1)  # [1, num_ws)
            if not 0 < mixing_cutoff < self.num_ws:
                raise ValueError("mixing_cutoff must be inside the style stack")
            mixing_ws = mixing_w[:, None, :].repeat(1, self.num_ws - mixing_cutoff, 1)
            ws = torch.cat([ws[:, :mixing_cutoff], mixing_ws], dim=1)

        truncation_psi = float(truncation_psi)
        if truncation_psi != 1.0:
            if truncation_cutoff is None:
                truncation_cutoff = self.num_ws
            if not 0 <= truncation_cutoff <= self.num_ws:
                raise ValueError("truncation_cutoff is outside the style stack.")
            if truncation_cutoff > 0:
                # w_avg + truncation_psi * (ws - w_avg)
                truncated = torch.lerp(
                    self.w_avg.view(1, 1, -1),  # (1, 1, style_dim)
                    ws[:, :truncation_cutoff],  # (B, truncation_cutoff, style_dim)
                    truncation_psi,
                )
                ws = torch.cat([truncated, ws[:, truncation_cutoff:]], dim=1)
        return ws  # (B, num_ws, style_dim)

    def synthesize(self, ws, noise_mode="random"):
        hidden = self.constant.expand(ws.shape[0], -1, -1, -1)  # (B, C[4], 4, 4)
        rgb = None
        for index, block in enumerate(self.blocks):
            start = 0 if index == 0 else 2 * index - 1  # overlapped W slot
            block_styles = ws[:, start : start + block.num_ws]
            hidden, rgb = block(hidden, block_styles, rgb, noise_mode)
        return rgb  # (B, 3, R, R)

    def forward(
        self,
        z,
        mixing_z=None,
        mixing_cutoff=None,
        return_ws=False,
        noise_mode="random",
        *,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
        ws = self.make_ws(
            z,
            mixing_z,
            mixing_cutoff,
            update_w_avg=update_w_avg,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        image = self.synthesize(ws, noise_mode)
        return (image, ws) if return_ws else image


class DiscriminatorResidualBlock(nn.Module):
    """StyleGAN2 residual downsampling block with a filtered skip path."""

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
        self.skip = EqualizedConv2d(
            in_channels,
            out_channels,
            1,
            bias=False,
            gain=1.0,
        )
        self.scale = 1 / math.sqrt(2)

    def forward(self, inputs):
        # inputs: (B, Cin, H, W)
        residual = filtered_downsample2d(self.skip(inputs))  # (B, Cout, H/2, W/2)
        hidden = F.leaky_relu(self.conv1(inputs), 0.2)  # (B, Cin, H, W)
        hidden = F.leaky_relu(self.conv2(hidden), 0.2)  # (B, Cout, H, W)
        hidden = filtered_downsample2d(hidden)  # (B, Cout, H/2, W/2)
        return (hidden + residual) * self.scale  # (B, Cout, H/2, W/2)


class StyleDiscriminator(nn.Module):
    """Full 128x128 residual discriminator used by compact StyleGAN2."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = int(base_channels)
        self.resolutions = RESOLUTIONS
        self.channels = make_channel_map(self.base_channels, self.resolutions)
        final_resolution = self.resolutions[-1]
        self.from_rgb = EqualizedConv2d(3, self.channels[final_resolution], 1)
        self.blocks = nn.Sequential(
            *[
                DiscriminatorResidualBlock(
                    self.channels[resolution],
                    self.channels[resolution // 2],
                )
                for resolution in reversed(self.resolutions[1:])
            ]
        )  # (B, C[128], 128, 128) -> (B, C[4], 4, 4)
        self.minibatch_std = MinibatchStandardDeviation()
        self.final_conv = EqualizedConv2d(
            self.channels[4] + 1,
            self.channels[4],
            3,
            padding=1,
        )  # (B, C[4], 4, 4)
        self.final = nn.Sequential(
            nn.Flatten(),
            EqualizedLinear(
                self.channels[4] * 4 * 4,
                self.channels[4],
                gain=math.sqrt(2),
            ),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(self.channels[4], 1),
        )  # (B, C[4] × 4 × 4) -> (B, C[4]) -> (B, 1)

    def forward(self, images):
        # images: (B, 3, 128, 128)
        hidden = F.leaky_relu(self.from_rgb(images), 0.2)  # (B, C[128], 128, 128)
        hidden = self.blocks(hidden)  # (B, C[4], 4, 4)
        hidden = self.minibatch_std(hidden)
        hidden = F.leaky_relu(self.final_conv(hidden), 0.2)  # (B, C[4], 4, 4)
        return self.final(hidden).squeeze(1)  # (B,)


__all__ = [
    "DiscriminatorResidualBlock",
    "MappingNetwork",
    "ModulatedConv2d",
    "StyleDiscriminator",
    "StyleGenerator",
    "StyledConv",
    "SynthesisBlock",
    "ToRGB",
    "denormalize",
]
