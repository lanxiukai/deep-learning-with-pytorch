"""Shared numerical layers for ProGAN, StyleGAN, and StyleGAN2.

The model modules intentionally do not inherit from one another. This module
contains numerical building blocks whose behavior stays
the same across ProGAN, StyleGAN, and StyleGAN2.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

RESOLUTIONS = (4, 8, 16, 32, 64, 128)
CHANNEL_MULTIPLIERS = {
    4: 8,
    8: 8,
    16: 8,
    32: 4,
    64: 2,
    128: 1,
}
NOISE_MODES = frozenset({"random", "fixed", "none"})


def make_channel_map(base_channels, resolutions=RESOLUTIONS):
    """Scale the compact 128x128 feature schedule from one base width."""
    base_channels = int(base_channels)
    resolutions = tuple(int(value) for value in resolutions)
    unknown = sorted(set(resolutions) - set(CHANNEL_MULTIPLIERS))
    if unknown:
        raise ValueError(f"channel multipliers are missing for {unknown}.")
    return {
        resolution: base_channels * CHANNEL_MULTIPLIERS[resolution]
        for resolution in resolutions
    }


def validate_resolution(resolution, resolutions=RESOLUTIONS):
    """Return a supported integer resolution or raise a helpful error."""
    resolution = int(resolution)
    supported = tuple(int(value) for value in resolutions)
    if resolution not in supported:
        raise ValueError(f"resolution must be one of {supported}, got {resolution}")
    return resolution


def validate_alpha(alpha):
    """Return a fade-in coefficient constrained to the closed unit interval."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha


class PixelNorm(nn.Module):
    """Normalize every sample independently across its feature channels."""

    def forward(self, inputs):
        statistics = inputs.float()
        scale = torch.rsqrt(statistics.square().mean(dim=1, keepdim=True) + 1e-8)
        return inputs * scale.to(dtype=inputs.dtype)


class EqualizedLinear(nn.Module):
    """Linear layer with runtime weight scaling and optional LR multiplier."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bias_init=0.0,
        learning_rate_multiplier=1.0,
        gain=1.0,
    ):
        super().__init__()
        if learning_rate_multiplier <= 0:
            raise ValueError("learning_rate_multiplier must be positive.")
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / learning_rate_multiplier
        )
        self.bias = (
            nn.Parameter(torch.full((out_features,), float(bias_init)))
            if bias
            else None
        )
        self.scale = float(gain) * learning_rate_multiplier / math.sqrt(in_features)
        self.learning_rate_multiplier = learning_rate_multiplier

    def forward(self, inputs):
        bias = (
            self.bias * self.learning_rate_multiplier if self.bias is not None else None
        )
        # Store unscaled parameters self.weight and derive the effective weights
        # self.weight * self.scale at each forward pass.
        return F.linear(inputs, self.weight * self.scale, bias)


class MappingNetwork(nn.Module):
    """Map normalized Z vectors into an intermediate W space."""

    def __init__(self, z_dim, style_dim, layers=8):
        super().__init__()
        modules = [PixelNorm()]
        in_features = z_dim
        for _ in range(layers):
            modules.extend(
                [
                    EqualizedLinear(
                        in_features,
                        style_dim,
                        learning_rate_multiplier=0.01,
                        gain=math.sqrt(2),
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_features = style_dim
        self.net = nn.Sequential(*modules)

    def forward(self, z):
        # latent vector (input): (B, z_dim)
        # style vector/W latent vector (output): (B, style_dim)
        return self.net(z)


class EqualizedConv2d(nn.Module):
    """Convolution with ProGAN-style equalized learning-rate scaling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *,
        padding=0,
        bias=True,
        gain=2**0.5,
    ):
        super().__init__()
        self.padding = int(padding)
        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = float(gain) / math.sqrt(fan_in)

    def forward(self, inputs):
        return F.conv2d(
            inputs,
            self.weight * self.scale,
            self.bias,
            padding=self.padding,
        )


class MinibatchStandardDeviation(nn.Module):
    """Append one feature map containing minibatch variation statistics."""

    def __init__(self, maximum_group=4):
        super().__init__()
        self.maximum_group = int(maximum_group)

    def forward(self, inputs):
        # inputs shape: (B, C, H, W)
        batch_size, channels, height, width = inputs.shape
        group = min(batch_size, self.maximum_group)
        while batch_size % group:
            group -= 1
        grouped = inputs.float().reshape(
            group,
            -1,
            channels,
            height,
            width,
        )  # (G, B // G, C, H, W)
        deviation = torch.sqrt(
            grouped.var(dim=0, unbiased=False) + 1e-8
        )  # (B // G, C, H, W)
        feature = deviation.mean(dim=(1, 2, 3), keepdim=True)  # (B // G, 1, 1, 1)
        feature = feature.repeat(group, 1, height, width).to(
            inputs.dtype
        )  # (B, 1, H, W)
        return torch.cat([inputs, feature], dim=1)  # (B, C + 1, H, W)


class NoiseInjection(nn.Module):
    """Add learned-strength random, deterministic, or disabled spatial noise."""

    def __init__(
        self,
        channels,
        resolution,
        fixed_noise_seed,
        *,
        per_channel=True,
    ):
        super().__init__()
        self.resolution = int(resolution)
        weight_shape = (1, channels, 1, 1) if per_channel else ()
        # Learned noise scaling, optionally with one strength per feature channel.
        self.weight = nn.Parameter(torch.zeros(weight_shape))
        random_generator = torch.Generator().manual_seed(int(fixed_noise_seed))
        fixed_noise = torch.randn(
            1,
            1,
            self.resolution,
            self.resolution,
            generator=random_generator,
        )
        # Fixed noise is reproducible from the model configuration and should
        # not make otherwise compatible checkpoints fail to load.
        self.register_buffer("fixed_noise", fixed_noise, persistent=False)

    def forward(self, inputs, noise_mode="random"):
        # inputs shape: (B, C, R, R)
        # noise shape: (B, 1, R, R) if noise_mode != "none" or None
        # weight shape: (1, C, 1, 1) if per_channel else ()
        if noise_mode not in NOISE_MODES:
            modes = ", ".join(sorted(NOISE_MODES))
            raise ValueError(
                f"noise_mode must be one of {{{modes}}}, got {noise_mode!r}"
            )
        if noise_mode == "none":
            return inputs
        if inputs.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError(
                "noise resolution does not match the feature map: "
                f"expected {self.resolution}x{self.resolution}, "
                f"got {inputs.shape[-2]}x{inputs.shape[-1]}"
            )
        if noise_mode == "random":
            noise = torch.randn(
                inputs.shape[0],
                1,
                self.resolution,
                self.resolution,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            noise = self.fixed_noise.to(
                device=inputs.device,
                dtype=inputs.dtype,
            )
        noise_weight = self.weight.to(dtype=inputs.dtype)
        return inputs + noise_weight * noise  # (B, C, R, R)


@lru_cache(maxsize=256)
def _resample_weights(values, device_name, dtype, channels):
    """Cache constant depthwise filter weights without a CUDA sync."""
    if not values:
        raise ValueError("resample kernel must be a non-empty 1D sequence.")
    kernel_sum = sum(values)
    if kernel_sum == 0:
        raise ValueError("resample kernel must have a non-zero sum.")
    # A filter first requested during evaluation is also reused by training.
    # Cached inference tensors cannot be saved by autograd for backward passes.
    with torch.inference_mode(False):
        kernel = torch.tensor(
            values,
            device=torch.device(device_name),
            dtype=dtype,
        )
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel_sum**2
        return kernel.view(1, 1, len(values), len(values)).repeat(
            channels,
            1,
            1,
            1,
        )


def filter2d(inputs, kernel=(1, 3, 3, 1)):
    """Apply a channel-wise FIR low-pass filter without custom kernels."""
    # inputs shape: (B, C, H, W)
    kernel = tuple(float(value) for value in kernel)
    size = len(kernel)
    pad_before = (size - 1) // 2
    pad_after = size // 2
    padded = F.pad(
        inputs,
        (pad_before, pad_after, pad_before, pad_after),
        mode="replicate",
    )  # padded shape: (B, C, H + size - 1, W + size - 1)
    channels = inputs.shape[1]
    weights = _resample_weights(
        kernel,
        str(inputs.device),
        inputs.dtype,
        channels,
    )
    # grouped convolution
    return F.conv2d(padded, weights, groups=channels)  # (B, C, H, W)


def filtered_upsample2d(inputs, kernel=(1, 3, 3, 1)):
    """Upsample by two and suppress nearest-neighbor imaging artifacts."""
    upsampled = F.interpolate(inputs, scale_factor=2, mode="nearest")
    return filter2d(upsampled, kernel)  # (B, C, 2H, 2W)


def filtered_downsample2d(inputs, kernel=(1, 3, 3, 1)):
    """Low-pass filter before reducing spatial resolution by two."""
    if min(inputs.shape[-2:]) < 2:
        raise ValueError("filtered_downsample2d needs spatial size >= 2.")
    return F.avg_pool2d(filter2d(inputs, kernel), kernel_size=2)  # (B, C, H/2, W/2)


def denormalize(images):
    """Map images from [-1, 1] to a display-safe [0, 1] interval."""
    return images.mul(0.5).add(0.5).clamp(0, 1)


__all__ = [
    "CHANNEL_MULTIPLIERS",
    "NOISE_MODES",
    "RESOLUTIONS",
    "EqualizedConv2d",
    "EqualizedLinear",
    "MappingNetwork",
    "MinibatchStandardDeviation",
    "NoiseInjection",
    "PixelNorm",
    "denormalize",
    "filter2d",
    "filtered_downsample2d",
    "filtered_upsample2d",
    "make_channel_map",
    "validate_alpha",
    "validate_resolution",
]
