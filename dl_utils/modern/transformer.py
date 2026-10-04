"""DiT/SiT patch backbone and explicit RoPE/QK-normalization teaching variants."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.diffusion.diffusion_unet import SinusoidalTimeEmbedding


def position_embedding(height, width, dim, device):
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(dim // 4, device=device) / (dim // 4)
    )
    angles = [coordinate.flatten()[:, None] * frequencies for coordinate in (y, x)]
    return torch.cat([a.sin() for a in angles] + [a.cos() for a in angles], dim=-1)


def rope_2d(x, height, width, prefix=0):
    """Rotate image Q/K pairs by row/column; condition prefix tokens stay unrotated."""
    context, image = x[:, :, :prefix], x[:, :, prefix:]
    y, z = torch.meshgrid(
        torch.arange(height, device=x.device),
        torch.arange(width, device=x.device),
        indexing="ij",
    )
    quarter = x.shape[-1] // 4
    frequency = torch.exp(
        -math.log(10000) * torch.arange(quarter, device=x.device) / quarter
    )
    angles = torch.cat(
        [y.flatten()[:, None] * frequency, z.flatten()[:, None] * frequency], dim=-1
    )
    pairs = image.reshape(*image.shape[:-1], -1, 2)
    a, b = pairs.unbind(-1)
    rotated = torch.stack(
        (a * angles.cos() - b * angles.sin(), a * angles.sin() + b * angles.cos()),
        dim=-1,
    )
    return torch.cat((context, rotated.flatten(-2)), dim=2)


class Attention(nn.Module):
    def __init__(self, dim, heads, rope=False, qk_norm=False):
        super().__init__()
        if dim % heads or (dim // heads) % 4:
            raise ValueError("Head dimension must be a multiple of four.")
        self.heads, self.rope = heads, rope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(dim // heads, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(dim // heads, eps=1e-6) if qk_norm else nn.Identity()
        self.last_max_logit = None
        self.collect_diagnostics = False
        self.last_entropy = None

    def forward(self, tokens, height, width, prefix=0):
        batch, length, dim = tokens.shape
        q, k, v = (
            self.qkv(tokens)
            .reshape(batch, length, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            q, k = rope_2d(q, height, width, prefix), rope_2d(k, height, width, prefix)
        # Explicit math also supports forward-mode JVP in iMF.
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(dim // self.heads)
        weights = logits.float().softmax(-1).to(v.dtype)
        if self.collect_diagnostics:
            self.last_max_logit = logits.detach().abs().max().item()
            probabilities = weights.detach().float()
            self.last_entropy = (
                -(probabilities * probabilities.clamp_min(1e-12).log())
                .sum(-1)
                .mean()
                .item()
            )
        value = (weights @ v).transpose(1, 2).reshape(batch, length, dim)
        return self.proj(value)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, rope, qk_norm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = Attention(dim, heads, rope, qk_norm)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition, h, w):
        b1, s1, g1, b2, s2, g2 = self.modulation(condition)[:, None].chunk(6, dim=-1)
        x = x + g1 * self.attention(self.norm1(x) * (1 + s1) + b1, h, w)
        return x + g2 * self.mlp(self.norm2(x) * (1 + s2) + b2)


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        in_channels=4,
        out_channels=4,
        dim=384,
        depth=12,
        heads=6,
        patch_size=2,
        num_classes=101,
        rope=False,
        qk_norm=False,
    ):
        super().__init__()
        self._config = {
            "image_size": image_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "patch_size": patch_size,
            "num_classes": num_classes,
            "rope": rope,
            "qk_norm": qk_norm,
        }
        self.sample_size, self.in_channels = image_size, in_channels
        self.out_channels, self.patch_size, self.num_classes = (
            out_channels,
            patch_size,
            num_classes,
        )
        self.null_class, self.dim, self.rope = num_classes, dim, rope
        self.patch = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.classes = nn.Embedding(num_classes + 1, dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, rope, qk_norm) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        self.output = nn.Linear(dim, patch_size**2 * out_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def config(self):
        return self._config

    def unpatchify(self, tokens, h, w):
        p, c = self.patch_size, self.out_channels
        return (
            tokens.reshape(len(tokens), h, w, p, p, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(len(tokens), c, h * p, w * p)
        )

    def forward(self, x, time, labels=None, *, return_layer=None):
        if x.shape[-1] % self.patch_size or x.shape[-2] % self.patch_size:
            raise ValueError("Spatial size must be divisible by patch size.")
        patches = self.patch(x)
        h, w = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        if not self.rope:
            tokens = tokens + position_embedding(h, w, self.dim, x.device)
        if labels is None:
            labels = torch.full(
                (len(x),), self.null_class, device=x.device, dtype=torch.long
            )
        condition = self.time(time) + self.classes(labels)
        intermediate = None
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, condition, h, w)
            if i == return_layer:
                intermediate = tokens
        bias, scale = self.final_modulation(condition)[:, None].chunk(2, dim=-1)
        output = self.unpatchify(
            self.output(self.norm(tokens) * (1 + scale) + bias), h, w
        )
        return (output, intermediate) if return_layer is not None else output


class RepresentationAlignment(nn.Module):
    """Frozen DINOv2 patch targets; the codec and semantic teacher remain distinct."""

    def __init__(self, dim, teacher=None):
        super().__init__()
        self.teacher = (
            teacher
            if teacher is not None
            else torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14")
        )
        self.teacher.eval().requires_grad_(False)
        self.projector = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 384)
        )

    def forward(self, tokens, images):
        with torch.no_grad():
            rgb = F.interpolate(
                images.add(1).div(2),
                (224, 224),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            target = self.teacher.forward_features((rgb - mean) / std)[
                "x_norm_patchtokens"
            ]
            side = math.isqrt(tokens.shape[1])
            grid = target.transpose(1, 2).reshape(len(images), 384, 16, 16)
            target = (
                F.interpolate(grid, (side, side), mode="bilinear", align_corners=False)
                .flatten(2)
                .transpose(1, 2)
            )
        return (
            -(
                F.normalize(self.projector(tokens).float(), dim=-1)
                * F.normalize(target.float(), dim=-1)
            )
            .sum(-1)
            .mean(-1)
        )
