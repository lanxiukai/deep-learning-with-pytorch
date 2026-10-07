"""DMD2 helpers: shared fake-score critic, online rollout and stopped DM gradient."""

import copy

import torch
from torch import nn


class FakeScoreCritic(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.denoiser = copy.deepcopy(teacher).requires_grad_(True)
        self.discriminator = nn.Linear(teacher.network.hidden_dims[-1], 1)

    def forward(self, x, sigma, labels):
        denoised, features = self.denoiser(x, sigma, labels, return_features=True)
        return denoised, self.discriminator(features).squeeze(-1)


def student_rollout(student, noise, labels, sigmas, *, stop_at=None, track_last=False):
    """Earlier stages are stopped; only the selected stage carries gradients."""
    state = noise * sigmas[0]
    final = len(sigmas) - 1 if stop_at is None else stop_at
    for i in range(final + 1):
        with torch.set_grad_enabled(track_last and i == final):
            clean = student(state, sigmas[i].expand(len(noise)), labels)
        if i < final:
            state = clean.detach() + sigmas[i + 1] * torch.randn_like(clean)
    return clean


def dm_surrogate(student_images, real_denoised, fake_denoised):
    """Normalized denoiser difference = a reweighted score-difference direction.

    This scalar is a gradient carrier, not a KL estimate. Sigma-squared weight
    cancels the additive-path score denominator, before per-image normalization.
    """
    normalizer = (
        (student_images.detach() - real_denoised)
        .abs()
        .flatten(1)
        .mean(1)
        .clamp_min(1e-5)
    )
    direction = (fake_denoised - real_denoised) / normalizer[:, None, None, None]
    target = (student_images - direction).detach()
    loss = 0.5 * (student_images - target).square().flatten(1).mean(1)
    return loss, direction.detach().square().mean().sqrt()
