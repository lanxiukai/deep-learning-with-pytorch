"""StyleGAN models and their explicit continuation configuration."""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.gan.continuation import ContinuationContext, ContinuationPlan
from dl_utils.gan.continuation import (
    add_refinement_arguments as _add_refinement_arguments,
)
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
    validate_alpha,
    validate_resolution,
)
from dl_utils.gan.training import ProgressivePhase


class AdaptiveInstanceNorm(nn.Module):
    """(AdaIN) Normalize features, then apply sample-specific scale and bias from W."""

    def __init__(self, channels, style_dim):
        super().__init__()
        self.channels = int(channels)
        # The first C values are for scaling, and the second C values are for bias.
        # (B, style_dim) → (B, 2C)
        self.affine = EqualizedLinear(style_dim, self.channels * 2)

    def forward(self, inputs, style):
        # inputs: (B, C, H, W), style: (B, style_dim)
        statistics = inputs.float()
        mean = statistics.mean(dim=(2, 3), keepdim=True)
        variance = statistics.var(
            dim=(2, 3),
            unbiased=False,
            keepdim=True,
        )
        normalized = (statistics - mean) * torch.rsqrt(variance + 1e-8)
        normalized = normalized.to(dtype=inputs.dtype)
        scale, bias = self.affine(style).chunk(2, dim=1)  # scale: (B, C), bias: (B, C)
        scale = scale.view(-1, self.channels, 1, 1) + 1.0  # (B, C, 1, 1)
        bias = bias.view(-1, self.channels, 1, 1)  # (B, C, 1, 1)
        return normalized * scale + bias  # (B, C, H, W)


class StyledActivation(nn.Module):
    """Apply StyleGAN noise, activation, and AdaIN in the original order."""

    def __init__(
        self,
        channels,
        style_dim,
        resolution,
        fixed_noise_seed,
    ):
        super().__init__()
        self.noise = NoiseInjection(channels, resolution, fixed_noise_seed)
        self.adain = AdaptiveInstanceNorm(channels, style_dim)

    def forward(self, inputs, style, noise_mode="random"):
        # inputs: (B, C, R, R), style: (B, style_dim)
        hidden = self.noise(inputs, noise_mode)
        hidden = F.leaky_relu(hidden, 0.2)
        return self.adain(hidden, style)  # (B, C, R, R)


class SynthesisBlock(nn.Module):
    """Generate one StyleGAN resolution using two independent W inputs."""

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
        self.conv1 = (
            None
            if self.first
            else EqualizedConv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
            )
        )
        self.activation1 = StyledActivation(
            out_channels,
            style_dim,
            resolution,
            fixed_noise_seed,
        )
        self.conv2 = EqualizedConv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
        )
        self.activation2 = StyledActivation(
            out_channels,
            style_dim,
            resolution,
            fixed_noise_seed + 1,
        )

    def forward(self, inputs, styles, noise_mode="random"):
        # inputs: (B, C_in, R/2, R/2), style: (B, 2, style_dim)
        if self.first:
            hidden = inputs  # (B, C_out, 4, 4)
        else:
            hidden = filtered_upsample2d(inputs)  # (B, C_in, R, R)
            hidden = self.conv1(hidden)  # (B, C_out, R, R)
        hidden = self.activation1(hidden, styles[:, 0], noise_mode)
        hidden = self.conv2(hidden)  # (B, C_out, R, R)
        return self.activation2(hidden, styles[:, 1], noise_mode)


class StyleGANGenerator(nn.Module):
    """Progressive AdaIN generator with style mixing and truncation."""

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
        self.resolutions = RESOLUTIONS
        self.styles_per_block = 2
        # Style vectors, ws.shape: (B, 12, style_dim)
        self.num_ws = len(self.resolutions) * self.styles_per_block
        self.channels = make_channel_map(self.base_channels, self.resolutions)
        self.mapping = MappingNetwork(
            self.z_dim,
            self.style_dim,
            mapping_layers,
        )
        # self.w_avg.shape: (style_dim,)
        self.register_buffer("w_avg", torch.zeros(self.style_dim))
        self.constant = nn.Parameter(
            torch.randn(1, self.channels[4], 4, 4)
        )  # (1, C[4], 4, 4)
        self.blocks = nn.ModuleList(
            [
                SynthesisBlock(
                    self.channels[resolution // 2]
                    if resolution > 4
                    else self.channels[4],
                    self.channels[resolution],
                    self.style_dim,
                    resolution,
                    fixed_noise_seed=2 * index,
                    first=resolution == 4,
                )
                for index, resolution in enumerate(self.resolutions)
            ]
        )
        self.to_rgbs = nn.ModuleList(
            [
                EqualizedConv2d(
                    self.channels[resolution],
                    3,
                    1,
                    gain=1.0,
                )
                for resolution in self.resolutions
            ]
        )

    def num_ws_for_resolution(self, resolution):
        resolution = validate_resolution(resolution, self.resolutions)
        return (self.resolutions.index(resolution) + 1) * self.styles_per_block

    # z (B, z_dim) -> style tensor (B, num_ws, style_dim)
    def make_ws(
        self,
        z,
        *,
        resolution=128,
        mixing_z=None,
        mixing_cutoff=None,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
    ):
        resolution = validate_resolution(resolution, self.resolutions)
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
        active_ws = self.num_ws_for_resolution(resolution)

        if mixing_z is not None:
            # mixing_z: (B, z_dim)
            mixing_w = self.mapping(mixing_z)  # (B, style_dim)
            if mixing_cutoff is None:
                mixing_cutoff = random.randint(1, active_ws - 1)  # [1, active_ws)
            if not 0 < mixing_cutoff < active_ws:
                raise ValueError("mixing_cutoff must be inside the active style stack")
            mixing_ws = mixing_w[:, None, :].repeat(1, self.num_ws - mixing_cutoff, 1)
            ws = torch.cat([ws[:, :mixing_cutoff], mixing_ws], dim=1)

        truncation_psi = float(truncation_psi)
        if truncation_psi != 1.0:
            if truncation_cutoff is None:
                truncation_cutoff = active_ws
            if not 0 <= truncation_cutoff <= active_ws:
                raise ValueError("truncation_cutoff is outside the active style stack.")
            if truncation_cutoff > 0:
                # w_avg + truncation_psi * (ws - w_avg)
                truncated = torch.lerp(
                    self.w_avg.view(1, 1, -1),  # (1, 1, style_dim)
                    ws[:, :truncation_cutoff],  # (B, truncation_cutoff, style_dim)
                    truncation_psi,
                )
                ws = torch.cat([truncated, ws[:, truncation_cutoff:]], dim=1)
        return ws  # (B, num_ws, style_dim)

    def synthesize(
        self,
        ws,
        *,
        resolution=128,
        alpha=1.0,
        noise_mode="random",
    ):
        resolution = validate_resolution(resolution, self.resolutions)
        alpha = validate_alpha(alpha)

        hidden = self.constant.expand(ws.shape[0], -1, -1, -1)  # (B, C[4], 4, 4)
        hidden = self.blocks[0](
            hidden,
            ws[:, : self.styles_per_block],
            noise_mode,
        )  # (B, C[4], 4, 4)
        if resolution == 4:
            return self.to_rgbs[0](hidden)  # (B, 3, 4, 4)

        target_index = self.resolutions.index(resolution)
        for index in range(1, target_index + 1):
            previous = hidden  # (B, C[R/2], R/2, R/2)
            start = index * self.styles_per_block
            hidden = self.blocks[index](
                hidden,
                ws[:, start : start + self.styles_per_block],
                noise_mode,
            )  # (B, C[R], R, R)
            if index == target_index:
                break
        new_rgb = self.to_rgbs[index](hidden)  # (B, 3, R, R)
        if alpha == 1.0:
            return new_rgb
        old_rgb = filtered_upsample2d(self.to_rgbs[index - 1](previous))
        # old_rgb + alpha * (new_rgb - old_rgb)
        return torch.lerp(old_rgb, new_rgb, alpha)  # (B, 3, R, R)

    def forward(
        self,
        z,
        *,
        resolution=128,
        alpha=1.0,
        mixing_z=None,
        mixing_cutoff=None,
        update_w_avg=False,
        truncation_psi=1.0,
        truncation_cutoff=None,
        return_ws=False,
        noise_mode="random",
    ):
        ws = self.make_ws(
            z,
            resolution=resolution,
            mixing_z=mixing_z,
            mixing_cutoff=mixing_cutoff,
            update_w_avg=update_w_avg,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        image = self.synthesize(
            ws,
            resolution=resolution,
            alpha=alpha,
            noise_mode=noise_mode,
        )
        return (image, ws) if return_ws else image


class DiscriminatorBlock(nn.Module):
    """Reduce one StyleGAN discriminator resolution."""

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
        # (B, C_in, R, R)
        hidden = F.leaky_relu(self.conv1(inputs), 0.2)
        hidden = F.leaky_relu(self.conv2(hidden), 0.2)
        return filtered_downsample2d(hidden)  # (B, C_out, R/2, R/2)


class StyleGANDiscriminator(nn.Module):
    """Progressive discriminator paired with the AdaIN generator."""

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
        )
        self.blocks = nn.ModuleDict(
            {
                str(resolution): DiscriminatorBlock(
                    self.channels[resolution],
                    self.channels[resolution // 2],
                )
                for resolution in self.resolutions[1:]
            }
        )
        final_channels = self.channels[4]
        self.minibatch_std = MinibatchStandardDeviation()
        self.final_conv = EqualizedConv2d(
            final_channels + 1,
            final_channels,
            3,
            padding=1,
        )
        self.final_linear = EqualizedLinear(
            final_channels * 4 * 4,
            final_channels,
            gain=math.sqrt(2),
        )
        self.output = EqualizedLinear(final_channels, 1)

    def forward(self, images, resolution=128, alpha=1.0):
        # images: (B, 3, R, R)
        resolution = validate_resolution(resolution, self.resolutions)
        alpha = validate_alpha(alpha)
        if images.shape[-2:] != (resolution, resolution):
            raise ValueError(
                f"expected {resolution}x{resolution} images, "
                f"got {images.shape[-2]}x{images.shape[-1]}"
            )

        current_index = self.resolutions.index(resolution)
        if resolution == 4:
            hidden = F.leaky_relu(self.from_rgbs["4"](images), 0.2)  # (B, C[4], 4, 4)
        else:
            new_hidden = F.leaky_relu(
                self.from_rgbs[str(resolution)](images),
                0.2,
            )  # (B, C[R], R, R)
            new_hidden = self.blocks[str(resolution)](
                new_hidden
            )  # (B, C[R/2], R/2, R/2)
            old_images = filtered_downsample2d(images)  # (B, 3, R/2, R/2)
            old_hidden = F.leaky_relu(
                self.from_rgbs[str(resolution // 2)](old_images),
                0.2,
            )  # (B, C[R/2], R/2, R/2)
            # old_hidden + alpha * (new_hidden - old_hidden)
            hidden = torch.lerp(old_hidden, new_hidden, alpha)
            for index in range(current_index - 1, 0, -1):
                block_resolution = self.resolutions[index]
                hidden = self.blocks[str(block_resolution)](hidden)  # (B, C[4], 4, 4)

        hidden = self.minibatch_std(hidden)  # (B, C[4] + 1, 4, 4)
        hidden = F.leaky_relu(self.final_conv(hidden), 0.2)  # (B, C[4], 4, 4)
        hidden = hidden.flatten(1)  # (B, C[4] * 4 * 4)
        hidden = F.leaky_relu(self.final_linear(hidden), 0.2)  # (B, C[4])
        return self.output(hidden).squeeze(1)  # (B, 1) -> (B,)


def add_refinement_arguments(parser):
    """Add continuation flags with the model's established defaults."""
    _add_refinement_arguments(
        parser,
        batch_size=32,
        learning_rate=1e-3,
        regularization_help="R1 gamma for StyleGAN.",
    )


def make_continuation_plan(
    args, *, model_config, discriminator_config, train_phase, d_reg_every
) -> ContinuationPlan:
    """Adapt the lesson's phase trainer without importing the lesson itself."""
    d_reg_every = d_reg_every if args.d_reg_every is None else args.d_reg_every
    reg_batch_shrink = 2 if args.reg_batch_shrink is None else args.reg_batch_shrink

    def train_chunk(context: ContinuationContext, count, state):
        metrics, state["global_step"] = train_phase(
            context.generator,
            context.discriminator,
            context.run.data,
            context.optimizer_g,
            context.optimizer_d,
            context.ema,
            ProgressivePhase(128, "stabilization", context.batch_size, count),
            context.run.precision,
            state["global_step"],
            d_reg_every,
            reg_batch_shrink,
            r1_gamma=args.refine_reg_weight,
        )
        return metrics

    return ContinuationPlan(
        model_name="stylegan",
        generator_class=StyleGANGenerator,
        discriminator_class=StyleGANDiscriminator,
        model_config=dict(model_config),
        discriminator_config=dict(discriminator_config),
        initial_unit="progressive-phase-main-metrics-v2",
        initial_completed_units=11,
        legacy_units=(),
        generator_kwargs={"noise_mode": "fixed", "truncation_psi": 1.0},
        d_reg_every=d_reg_every,
        reg_batch_shrink=reg_batch_shrink,
        path_batch_shrink=4,
        optimizer_ratios={"generator": 1.0, "discriminator": 1.0},
        train_chunk=train_chunk,
    )


__all__ = [
    "AdaptiveInstanceNorm",
    "DiscriminatorBlock",
    "MappingNetwork",
    "StyleGANDiscriminator",
    "StyleGANGenerator",
    "StyledActivation",
    "SynthesisBlock",
    "add_refinement_arguments",
    "denormalize",
    "make_continuation_plan",
]
