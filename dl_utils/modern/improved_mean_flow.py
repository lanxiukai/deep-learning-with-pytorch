"""iMF: induced instantaneous velocity, stopped JVP and adjustable guidance.

Reference: https://github.com/Lyy-iiis/imeanflow/tree/torch
The lesson uses coordinate-mean errors; its weighting epsilon is in those units.
"""

import itertools

import torch
from torch import nn

from dl_utils.modern.transformer import (
    Attention,
    DiffusionTransformer,
    position_embedding,
)


class ContextBlock(nn.Module):
    def __init__(self, dim, heads, rope=False, qk_norm=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, rope, qk_norm)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.gate1 = nn.Parameter(torch.zeros(dim))
        self.gate2 = nn.Parameter(torch.zeros(dim))

    def forward(self, x, h, w, prefix):
        x = x + self.gate1 * self.attention(self.norm1(x), h, w, prefix)
        return x + self.gate2 * self.mlp(self.norm2(x))


class ImprovedMeanFlow(DiffusionTransformer):
    def __init__(self, *, auxiliary_head=True, conditioning="tokens", **kwargs):
        super().__init__(**kwargs)
        self.auxiliary_head, self.conditioning = auxiliary_head, conditioning
        self._config.update(auxiliary_head=auxiliary_head, conditioning=conditioning)
        dim = self.dim
        self.slots = nn.Parameter(torch.randn(6, dim) * 0.02)
        if conditioning == "tokens":
            self.blocks = nn.ModuleList(
                [
                    ContextBlock(
                        dim,
                        kwargs.get("heads", 6),
                        kwargs.get("rope", False),
                        kwargs.get("qk_norm", False),
                    )
                    for _ in range(kwargs.get("depth", 12))
                ]
            )
            del self.final_modulation
        elif conditioning != "adaln":
            raise ValueError("Choose tokens or adaln conditioning.")
        self.velocity_head = (
            nn.Linear(dim, self.patch_size**2 * self.out_channels)
            if auxiliary_head
            else None
        )
        if self.velocity_head is not None:
            nn.init.zeros_(self.velocity_head.weight)
            nn.init.zeros_(self.velocity_head.bias)

    def forward(
        self,
        x,
        r,
        t,
        labels=None,
        omega=None,
        lower=None,
        upper=None,
        *,
        return_velocity=False,
    ):
        if labels is None:
            labels = torch.full(
                (len(x),), self.null_class, device=x.device, dtype=torch.long
            )
        omega = torch.ones_like(t) if omega is None else omega
        lower = torch.zeros_like(t) if lower is None else lower
        upper = torch.ones_like(t) if upper is None else upper
        conditions = (
            torch.stack(
                [
                    self.time(r * 1000),
                    self.time(t * 1000),
                    self.classes(labels),
                    self.time(omega.log() * 1000),
                    self.time(lower * 1000),
                    self.time(upper * 1000),
                ],
                dim=1,
            )
            + self.slots
        )
        patches = self.patch(x)
        h, w = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        if not self.rope:
            tokens = tokens + position_embedding(h, w, self.dim, x.device)
        if self.conditioning == "tokens":
            tokens = torch.cat((conditions, tokens), dim=1)
            for block in self.blocks:
                tokens = block(tokens, h, w, 6)
            tokens = self.norm(tokens[:, 6:])
        else:
            condition = conditions.sum(1)
            for block in self.blocks:
                tokens = block(tokens, condition, h, w)
            bias, scale = self.final_modulation(condition)[:, None].chunk(2, -1)
            tokens = self.norm(tokens) * (1 + scale) + bias
        u = self.unpatchify(self.output(tokens), h, w)
        if return_velocity:
            return u, self.unpatchify(
                self.velocity_head(tokens), h, w
            ) if self.velocity_head is not None else u
        return u


def imf_loss(
    model,
    clean,
    labels,
    *,
    equal_fraction=0.5,
    guidance_max=5.0,
    dropout=0.1,
    adaptive_power=0.5,
    auxiliary_weight=1.0,
    variable_interval=True,
):
    batch = len(clean)
    times = torch.randn(2, batch, device=clean.device).sigmoid().sort(dim=0).values
    r, t = times.unbind(0)
    r = torch.where(torch.rand_like(t) < equal_fraction, t, r)
    noise = torch.randn_like(clean)
    z = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    target = noise - clean
    dropped = torch.rand_like(t) < dropout
    labels = torch.where(dropped, model.null_class, labels)
    omega = 1 + (guidance_max - 1) * torch.rand_like(t)
    omega = torch.where(dropped, 1, omega)
    lower = torch.rand_like(t) * 0.3 if variable_interval else torch.zeros_like(t)
    upper = 0.7 + torch.rand_like(t) * 0.3 if variable_interval else torch.ones_like(t)
    full_interval = torch.rand_like(t) < 0.5
    lower = torch.where(full_interval, 0, lower)
    upper = torch.where(full_interval, 1, upper)
    effective = torch.where((t >= lower) & (t <= upper), omega, 1)
    _, velocity = model(z, t, t, labels, effective, lower, upper, return_velocity=True)
    with torch.no_grad():
        _, unconditional = model(
            z, t, t, None, torch.ones_like(t), lower, upper, return_velocity=True
        )
        guided_target = target + (1 - 1 / effective)[:, None, None, None] * (
            velocity.detach() - unconditional
        )
        guided_target = torch.where(dropped[:, None, None, None], target, guided_target)
    u, derivative = torch.func.jvp(
        lambda state, start, end: model(state, start, end, labels, omega, lower, upper),
        (z, r, t),
        (velocity.detach(), torch.zeros_like(r), torch.ones_like(t)),
    )
    induced = u + (t - r)[:, None, None, None] * derivative.detach()
    main_error = (induced - guided_target).square().flatten(1).mean(1)
    aux_error = (velocity - guided_target).square().flatten(1).mean(1)
    loss = main_error / (main_error.detach() + 1e-3).pow(adaptive_power)
    if model.auxiliary_head:
        loss = loss + auxiliary_weight * aux_error / (aux_error.detach() + 1e-3).pow(
            adaptive_power
        )
    return (
        loss,
        t,
        {
            "induced_mse": main_error.detach().mean(),
            "boundary_mse": (main_error.detach() * (r == t)).sum()
            / (r == t).sum().clamp_min(1),
            "interval_mse": (main_error.detach() * (r < t)).sum()
            / (r < t).sum().clamp_min(1),
            "boundary_examples": (r == t).sum(),
            "interval_examples": (r < t).sum(),
            "velocity_mse": aux_error.detach().mean(),
            "jvp_rms": derivative.detach().square().mean().sqrt(),
        },
    )


@torch.no_grad()
def sample_imf(model, noise, labels, *, steps=1, omega=1.0, lower=0.0, upper=1.0):
    state = noise.clone()
    times = torch.linspace(1, 0, steps + 1, device=noise.device)
    for t, r in itertools.pairwise(times):
        batch_t, batch_r = t.expand(len(noise)), r.expand(len(noise))
        velocity = model(
            state,
            batch_r,
            batch_t,
            labels,
            torch.full_like(batch_t, omega),
            torch.full_like(batch_t, lower),
            torch.full_like(batch_t, upper),
        )
        state = state - (t - r) * velocity
    return state
