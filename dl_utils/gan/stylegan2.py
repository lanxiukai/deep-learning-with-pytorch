"""StyleGAN2 models, training configuration, regularization, and continuation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.data.loading import resolve_num_workers
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


@dataclass(frozen=True)
class FixedResolutionGANOptions:
    """Resolved runtime options for the fixed-resolution GAN lesson."""

    total_kimg: int
    batch_size: int
    r1_batch_shrink: int
    path_batch_shrink: int
    num_workers: int
    prefetch_factor: int


def resolve_fixed_resolution_gan_options(
    *,
    total_kimg: int | None,
    batch_scale: int | None,
    r1_batch_shrink: int | None,
    path_batch_shrink: int | None,
    num_workers: int | None,
    prefetch_factor: int,
    base_batch_size: int,
    default_total_kimg: int,
    default_r1_batch_shrink: int,
    default_path_batch_shrink: int,
    default_num_workers: int,
) -> FixedResolutionGANOptions:
    """Validate and resolve the shared StyleGAN2 runtime options."""
    resolved_total_kimg = default_total_kimg if total_kimg is None else total_kimg
    resolved_batch_scale = 1 if batch_scale is None else batch_scale
    resolved_r1_shrink = (
        default_r1_batch_shrink if r1_batch_shrink is None else r1_batch_shrink
    )
    resolved_path_shrink = (
        default_path_batch_shrink if path_batch_shrink is None else path_batch_shrink
    )
    if (
        min(
            resolved_total_kimg,
            resolved_batch_scale,
            resolved_r1_shrink,
            resolved_path_shrink,
            base_batch_size,
            prefetch_factor,
        )
        < 1
    ):
        raise ValueError("training counts and scales must be positive.")
    return FixedResolutionGANOptions(
        total_kimg=resolved_total_kimg,
        batch_size=base_batch_size * resolved_batch_scale,
        r1_batch_shrink=resolved_r1_shrink,
        path_batch_shrink=resolved_path_shrink,
        num_workers=resolve_num_workers(num_workers, default_num_workers),
        prefetch_factor=prefetch_factor,
    )


@dataclass(frozen=True)
class TrainingEpoch:
    """One data epoch in a fixed-resolution image budget."""

    num_batches: int
    num_images: int
    final_batch_size: int

    def batch_size_at(self, batch_index, batch_size):
        if not 0 <= batch_index < self.num_batches:
            raise ValueError("batch_index is outside this training epoch.")
        if batch_index == self.num_batches - 1:
            return self.final_batch_size
        return batch_size


def build_training_schedule(total_kimg, batch_size, dataset_size):
    """Build data epochs with an exact final image and batch count."""
    if min(total_kimg, batch_size, dataset_size) < 1:
        raise ValueError("total_kimg, batch_size, and dataset_size must be positive.")
    if dataset_size < batch_size:
        raise ValueError("dataset_size must contain at least one complete batch.")
    total_images = total_kimg * 1_000
    total_batches = math.ceil(total_images / batch_size)
    final_training_batch = total_images - batch_size * (total_batches - 1)
    batches_per_epoch = dataset_size // batch_size
    schedule = []
    for batch_start in range(0, total_batches, batches_per_epoch):
        num_batches = min(batches_per_epoch, total_batches - batch_start)
        is_last = batch_start + num_batches == total_batches
        final_batch_size = final_training_batch if is_last else batch_size
        num_images = num_batches * batch_size
        if is_last:
            num_images -= batch_size - final_training_batch
        schedule.append(
            TrainingEpoch(
                num_batches=num_batches,
                num_images=num_images,
                final_batch_size=final_batch_size,
            )
        )
    return tuple(schedule)


def path_length_penalty(
    images: torch.Tensor,
    ws: torch.Tensor,
    running_mean: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularize image-space change per unit movement in W."""
    # images: (Bpath, 3, H, W), ws: (Bpath, num_ws, style_dim)
    if running_mean.ndim != 0:
        raise ValueError("running_mean must be a scalar tensor.")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be within [0, 1].")

    noise = torch.randn_like(images) / math.sqrt(images.shape[2] * images.shape[3])
    gradients = torch.autograd.grad(  # ∂output / ∂input
        (images * noise).sum(),  # output
        ws,  # input
        create_graph=True,
    )[0]  # (Bpath, num_ws, style_dim)
    lengths = torch.sqrt(gradients.square().sum(dim=2).mean(dim=1) + 1e-8)  # (B,)
    # running_mean + decay * (lengths.mean() - running_mean)
    updated_mean = running_mean.lerp(lengths.mean().detach(), decay)
    return (lengths - updated_mean).square().mean(), updated_mean


def add_refinement_arguments(parser):
    """Add continuation flags with StyleGAN2's established defaults."""
    _add_refinement_arguments(
        parser,
        batch_size=16,
        learning_rate=1e-3,
        regularization_help="R1 gamma for StyleGAN2.",
    )


def make_continuation_plan(
    args, *, model_config, discriminator_config, train_epoch, d_reg_every, g_reg_every
) -> ContinuationPlan:
    """Adapt lazy penalties and the running path mean to a bounded chunk."""
    reg_batch_shrink = 2 if args.r1_batch_shrink is None else args.r1_batch_shrink
    path_batch_shrink = 4 if args.path_batch_shrink is None else args.path_batch_shrink

    def train_chunk(context: ContinuationContext, count, state):
        metrics, path_mean, state["global_step"] = train_epoch(
            context.generator,
            context.discriminator,
            context.run.data,
            context.optimizer_g,
            context.optimizer_d,
            context.ema,
            torch.tensor(state["path_mean"], device=context.run.device),
            state["global_step"],
            TrainingEpoch(count, count * context.batch_size, context.batch_size),
            context.batch_size,
            context.run.precision,
            reg_batch_shrink,
            path_batch_shrink,
            r1_gamma=args.refine_reg_weight,
            d_reg_every=d_reg_every,
            g_reg_every=g_reg_every,
        )
        state["path_mean"] = path_mean.item()
        return metrics

    return ContinuationPlan(
        model_name="stylegan2",
        generator_class=StyleGenerator,
        discriminator_class=StyleDiscriminator,
        model_config=dict(model_config),
        discriminator_config=dict(discriminator_config),
        initial_unit="fixed-resolution-epoch-main-metrics-v2",
        initial_completed_units=None,
        legacy_units=(),
        generator_kwargs={"noise_mode": "fixed", "truncation_psi": 1.0},
        d_reg_every=d_reg_every,
        reg_batch_shrink=reg_batch_shrink,
        path_batch_shrink=path_batch_shrink,
        optimizer_ratios={
            "generator": g_reg_every / (g_reg_every + 1),
            "discriminator": d_reg_every / (d_reg_every + 1),
        },
        train_chunk=train_chunk,
    )


__all__ = [
    "DiscriminatorResidualBlock",
    "FixedResolutionGANOptions",
    "MappingNetwork",
    "ModulatedConv2d",
    "StyleDiscriminator",
    "StyleGenerator",
    "StyledConv",
    "SynthesisBlock",
    "ToRGB",
    "TrainingEpoch",
    "add_refinement_arguments",
    "build_training_schedule",
    "denormalize",
    "make_continuation_plan",
    "path_length_penalty",
    "resolve_fixed_resolution_gan_options",
]
