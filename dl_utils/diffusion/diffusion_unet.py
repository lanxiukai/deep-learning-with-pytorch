"""Compact time-conditioned U-Net shared by the diffusion lessons.

The architecture descends from the DGAI chapter helper originally bundled with
this repository.  It keeps residual time conditioning and spatial attention,
and places attention at declared low-resolution levels. The default 256x256
model uses attention at the 32x32 bottleneck.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["DiffusionUNet", "UNet"]


def _group_count(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    """Fourier features for either integer diffusion steps or continuous time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("time embedding dimension must be at least 4")
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        time = time.reshape(-1).float()
        half = self.dim // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / denominator
        )
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class ResidualBlock(nn.Module):
    """A residual convolutional block with additive time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, out_channels)
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, inputs: Tensor, time_embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(inputs)


class SelfAttention2d(nn.Module):
    """Spatial self-attention used only at explicitly selected resolutions."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        qkv = self.to_qkv(self.norm(inputs)).reshape(
            batch, 3, self.num_heads, self.head_dim, height * width
        )
        query, key, value = qkv.unbind(dim=1)
        attention = torch.einsum("bhdn,bhdm->bhnm", query, key) * self.scale
        attention = attention.softmax(dim=-1)
        attended = torch.einsum("bhnm,bhdm->bhdn", attention, value)
        attended = attended.reshape(batch, channels, height, width)
        return inputs + self.projection(attended)


class ClassCrossAttention(nn.Module):
    """One semantic token: softmax is 1; the class value is broadcast spatially."""

    def __init__(self, channels, context_dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context = nn.Linear(context_dim, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, images, context):
        tokens = images.flatten(2).transpose(1, 2)
        context = self.context(context)[:, None]
        update, _ = self.attention(
            self.norm(tokens), context, context, need_weights=False
        )
        return images + update.transpose(1, 2).reshape_as(images)


class ResampleResidual(nn.Module):
    """SR3-style time-conditioned residual resampling on both branches."""

    def __init__(self, source, target, time_dim, *, up):
        super().__init__()
        self.up = up
        self.norm = nn.GroupNorm(_group_count(source), source)
        self.main = nn.Conv2d(source, target, 3, padding=1)
        self.time = nn.Linear(time_dim, target)
        self.last = nn.Conv2d(target, target, 3, padding=1)
        self.skip = nn.Conv2d(source, target, 1)

    def forward(self, x, time):
        def resize(y):
            return (
                F.interpolate(y, scale_factor=2, mode="nearest")
                if self.up
                else F.avg_pool2d(y, 2)
            )

        h = self.main(resize(F.silu(self.norm(x))))
        h = self.last(F.silu(h + self.time(time)[:, :, None, None]))
        return (h + self.skip(resize(x))) / math.sqrt(2)


class DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        next_channels: int,
        time_dim: int,
        use_attention: bool,
        dropout: float,
        num_heads: int,
        downsample: bool,
        residual_resampling: bool = False,
    ) -> None:
        super().__init__()
        self.block1 = ResidualBlock(in_channels, channels, time_dim, dropout)
        self.block2 = ResidualBlock(channels, channels, time_dim, dropout)
        self.attention = (
            SelfAttention2d(channels, num_heads) if use_attention else nn.Identity()
        )
        self.residual_resampling = residual_resampling and downsample
        self.downsample = (
            ResampleResidual(channels, next_channels, time_dim, up=False)
            if self.residual_resampling
            else (
                nn.Conv2d(channels, next_channels, 4, stride=2, padding=1)
                if downsample
                else nn.Identity()
            )
        )

    def forward(
        self, hidden_states: Tensor, time_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        hidden_states = self.block1(hidden_states, time_embedding)
        hidden_states = self.block2(hidden_states, time_embedding)
        hidden_states = self.attention(hidden_states)
        down = (
            self.downsample(hidden_states, time_embedding)
            if self.residual_resampling
            else self.downsample(hidden_states)
        )
        return down, hidden_states


class UpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        channels: int,
        next_channels: int,
        time_dim: int,
        use_attention: bool,
        dropout: float,
        num_heads: int,
        upsample: bool,
        residual_resampling: bool = False,
    ) -> None:
        super().__init__()
        self.block1 = ResidualBlock(
            in_channels + skip_channels, channels, time_dim, dropout
        )
        self.block2 = ResidualBlock(channels, channels, time_dim, dropout)
        self.attention = (
            SelfAttention2d(channels, num_heads) if use_attention else nn.Identity()
        )
        self.residual_resampling = residual_resampling and upsample
        self.upsample = (
            ResampleResidual(channels, next_channels, time_dim, up=True)
            if self.residual_resampling
            else (
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv2d(channels, next_channels, 3, padding=1),
                )
                if upsample
                else nn.Identity()
            )
        )

    def forward(
        self, hidden_states: Tensor, skip: Tensor, time_embedding: Tensor
    ) -> Tensor:
        hidden_states = torch.cat((hidden_states, skip), dim=1)
        hidden_states = self.block1(hidden_states, time_embedding)
        hidden_states = self.block2(hidden_states, time_embedding)
        hidden_states = self.attention(hidden_states)
        return (
            self.upsample(hidden_states, time_embedding)
            if self.residual_resampling
            else self.upsample(hidden_states)
        )


class DiffusionUNet(nn.Module):
    """Compact U-Net for discrete diffusion steps or continuous noise time.

    If ``num_classes`` is given, one extra embedding represents the null class.
    Supplying ``labels=None`` selects it, which exposes classifier-free guidance
    without introducing a second classifier network.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_dims: Sequence[int] = (64, 128, 256, 384),
        attention_levels: Sequence[int] | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: int | None = None,
        out_channels: int | None = None,
        class_conditioning: str = "additive",
        residual_resampling: bool = False,
        extra_noise_condition: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if len(hidden_dims) < 2 or any(channels < 8 for channels in hidden_dims):
            raise ValueError("hidden_dims must contain at least two values >= 8")
        downsample_factor = 2 ** (len(hidden_dims) - 1)
        if image_size % downsample_factor != 0:
            raise ValueError("image_size must be divisible by the U-Net scale factor")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        attention_levels = (
            (len(hidden_dims) - 1,)
            if attention_levels is None
            else tuple(attention_levels)
        )
        if any(level < 0 or level >= len(hidden_dims) for level in attention_levels):
            raise ValueError("attention level is outside hidden_dims")

        self.sample_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_dims = hidden_dims
        self.attention_levels = tuple(attention_levels)
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.null_class = num_classes if num_classes is not None else 0

        if class_conditioning not in ("additive", "cross_attention"):
            raise ValueError("Unknown class conditioning interface.")
        self.class_conditioning = class_conditioning
        self.residual_resampling = residual_resampling
        self.extra_noise_condition = extra_noise_condition
        base_channels = hidden_dims[0]
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embedding = (
            nn.Embedding(num_classes + 1, time_dim) if num_classes is not None else None
        )
        self.aux_time_embedding = (
            nn.Sequential(
                SinusoidalTimeEmbedding(base_channels),
                nn.Linear(base_channels, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
            if extra_noise_condition
            else None
        )
        self.class_cross_attention = (
            ClassCrossAttention(hidden_dims[-1], time_dim, num_heads)
            if class_conditioning == "cross_attention" and num_classes is not None
            else None
        )
        self.input_projection = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        down_stages: list[nn.Module] = []
        current_channels = base_channels
        for level, channels in enumerate(hidden_dims):
            has_next = level + 1 < len(hidden_dims)
            next_channels = hidden_dims[level + 1] if has_next else channels
            down_stages.append(
                DownStage(
                    current_channels,
                    channels,
                    next_channels,
                    time_dim,
                    level in attention_levels,
                    dropout,
                    num_heads,
                    has_next,
                    residual_resampling,
                )
            )
            current_channels = next_channels
        self.down_stages = nn.ModuleList(down_stages)

        bottleneck_channels = hidden_dims[-1]
        self.middle_block1 = ResidualBlock(
            bottleneck_channels, bottleneck_channels, time_dim, dropout
        )
        self.middle_attention = SelfAttention2d(bottleneck_channels, num_heads)
        self.middle_block2 = ResidualBlock(
            bottleneck_channels, bottleneck_channels, time_dim, dropout
        )

        up_stages: list[nn.Module] = []
        current_channels = bottleneck_channels
        for level in reversed(range(len(hidden_dims))):
            channels = hidden_dims[level]
            has_previous = level > 0
            next_channels = hidden_dims[level - 1] if has_previous else channels
            up_stages.append(
                UpStage(
                    current_channels,
                    channels,
                    channels,
                    next_channels,
                    time_dim,
                    level in attention_levels,
                    dropout,
                    num_heads,
                    has_previous,
                    residual_resampling,
                )
            )
            current_channels = next_channels
        self.up_stages = nn.ModuleList(up_stages)

        self.output_norm = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.output_projection = nn.Conv2d(
            base_channels, self.out_channels, 3, padding=1
        )

    def config(self) -> dict[str, object]:
        return {
            "image_size": self.sample_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_dims": list(self.hidden_dims),
            "attention_levels": list(self.attention_levels),
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "class_conditioning": self.class_conditioning,
            "residual_resampling": self.residual_resampling,
            "extra_noise_condition": self.extra_noise_condition,
        }

    def forward(
        self,
        noisy_images: Tensor,
        timesteps: Tensor,
        labels: Tensor | None = None,
        *,
        extra_time=None,
        return_features=False,
    ) -> Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.in_channels:
            raise ValueError(
                "noisy_images must have shape [batch, in_channels, height, width]"
            )
        if noisy_images.shape[-2:] != (self.sample_size, self.sample_size):
            raise ValueError("input spatial size does not match model configuration")
        if timesteps.reshape(-1).shape[0] != noisy_images.shape[0]:
            raise ValueError("timesteps must contain one value per image")

        time_embedding = self.time_embedding(timesteps)
        if self.class_embedding is None:
            if labels is not None:
                raise ValueError("labels require a class-conditional model")
        else:
            if labels is None:
                labels = torch.full(
                    (noisy_images.shape[0],),
                    self.null_class,
                    device=noisy_images.device,
                    dtype=torch.long,
                )
            else:
                labels = labels.to(device=noisy_images.device, dtype=torch.long)
                if labels.shape != (noisy_images.shape[0],):
                    raise ValueError("labels must have shape [batch]")
                if labels.min() < 0 or labels.max() > self.null_class:
                    raise ValueError("class label is outside the configured range")
            if self.class_conditioning == "additive":
                time_embedding = time_embedding + self.class_embedding(labels)
        if self.aux_time_embedding is not None:
            if extra_time is None:
                raise ValueError(
                    "This model requires the condition augmentation level."
                )
            time_embedding = time_embedding + self.aux_time_embedding(extra_time)

        hidden = self.input_projection(noisy_images)
        skips: list[Tensor] = []
        for stage in self.down_stages:
            hidden, skip = stage(hidden, time_embedding)
            skips.append(skip)

        hidden = self.middle_block1(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle_block2(hidden, time_embedding)
        if self.class_cross_attention is not None:
            hidden = self.class_cross_attention(hidden, self.class_embedding(labels))
        features = hidden.mean(dim=(-2, -1))

        for stage in self.up_stages:
            hidden = stage(hidden, skips.pop(), time_embedding)
        output = self.output_projection(F.silu(self.output_norm(hidden)))
        return (output, features) if return_features else output


# Compatibility with notebooks that imported the original DGAI class name.
UNet = DiffusionUNet
