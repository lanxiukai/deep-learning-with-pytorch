"""Class-conditional consistency distillation from a frozen EDM teacher."""

import torch
from torch import nn

from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.edm import edm_noise_grid


class ConsistencyModel(nn.Module):
    def __init__(
        self,
        network_config,
        sigma_min=0.002,
        sigma_data=0.5,
        noise_embedding_scale=1000.0,
    ):
        super().__init__()
        self.network = DiffusionUNet(**network_config)
        self.sigma_min, self.sigma_data = sigma_min, sigma_data
        self.noise_embedding_scale = noise_embedding_scale

    def config(self):
        return {
            "network_config": self.network.config(),
            "sigma_min": self.sigma_min,
            "sigma_data": self.sigma_data,
            "noise_embedding_scale": self.noise_embedding_scale,
        }

    def forward(self, x, sigma, labels=None):
        s = sigma[:, None, None, None]
        skip = self.sigma_data**2 / ((s - self.sigma_min).square() + self.sigma_data**2)
        out = (
            self.sigma_data
            * (s - self.sigma_min)
            / (s.square() + self.sigma_data**2).sqrt()
        )
        residual = self.network(
            x / (s.square() + self.sigma_data**2).sqrt(),
            sigma.log() * 0.25 * self.noise_embedding_scale,
            labels,
        )
        return skip * x + out * residual


def consistency_loss(
    student,
    target,
    teacher,
    clean,
    labels,
    *,
    sigma_max=80.0,
    intervals=80,
    solver="heun",
):
    grid = edm_noise_grid(intervals + 1, student.sigma_min, sigma_max, 7, clean.device)[
        :-1
    ].flip(0)
    index = torch.randint(intervals, (len(clean),), device=clean.device)
    low, high = grid[index], grid[index + 1]
    noisy = clean + high[:, None, None, None] * torch.randn_like(clean)
    with torch.no_grad():
        derivative = (noisy - teacher(noisy, high, labels)) / high[:, None, None, None]
        h = (low - high)[:, None, None, None]
        paired = noisy + h * derivative
        if solver == "heun":
            next_derivative = (paired - teacher(paired, low, labels)) / low[
                :, None, None, None
            ]
            paired = noisy + h * (derivative + next_derivative) / 2
        expected = target(paired, low, labels)
    prediction = student(noisy, high, labels)
    return (prediction - expected).square().flatten(1).mean(
        1
    ), index.float() / intervals


@torch.no_grad()
def sample_consistency(model, noise, labels, *, steps=1, sigma_max=80.0):
    if steps < 1:
        raise ValueError("Need at least one consistency mapping.")
    levels = edm_noise_grid(steps + 1, model.sigma_min, sigma_max, 7, noise.device)[:-2]
    state = sigma_max * noise
    for i, sigma in enumerate(levels):
        if i:
            state = state + (sigma.square() - model.sigma_min**2).clamp_min(
                0
            ).sqrt() * torch.randn_like(state)
        state = model(state, sigma.expand(len(noise)), labels)
    return state
