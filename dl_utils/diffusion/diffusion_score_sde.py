"""Continuous VP score and classifier guidance; time runs from data to noise."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch


def _expand(value, sample):
    return value.reshape(-1, *((1,) * (sample.ndim - 1)))


@dataclass(frozen=True)
class VPSDE:
    """Linear-beta VP: drift=-beta*x/2, diffusion variance rate=beta."""

    beta_min: float = 0.1
    beta_max: float = 20.0
    prior_std: float = 1.0

    def __post_init__(self):
        if not 0 < self.beta_min < self.beta_max:
            raise ValueError("Expected 0 < beta_min < beta_max.")

    def beta(self, time):
        return self.beta_min + time * (self.beta_max - self.beta_min)

    def integrated_beta(self, time):
        return (
            self.beta_min * time + 0.5 * (self.beta_max - self.beta_min) * time.square()
        )

    def marginal_coefficients(self, time, sample):
        integral = self.integrated_beta(time)
        alpha = (-0.5 * integral).exp()
        sigma = (-torch.expm1(-integral)).sqrt()
        return _expand(alpha, sample), _expand(sigma, sample)

    def marginal_sample(self, clean, time, noise=None):
        noise = torch.randn_like(clean) if noise is None else noise
        alpha, sigma = self.marginal_coefficients(time, clean)
        return alpha * clean + sigma * noise, alpha, sigma

    def score_target(self, noise, sigma):
        return -noise / sigma

    def drift_diffusion(self, sample, time):
        beta = _expand(self.beta(time), sample)
        return -0.5 * beta * sample, beta


@torch.no_grad()
def sample_score_model(
    model,
    sde,
    noise,
    *,
    steps=50,
    solver="heun",
    epsilon=1e-3,
    classifier=None,
    labels=None,
    guidance=0.0,
):
    if (
        steps < 1
        or not 0 < epsilon < 1
        or solver not in ("euler", "heun", "reverse_sde")
    ):
        raise ValueError("Invalid VP sampler configuration.")
    if guidance and (classifier is None or labels is None):
        raise ValueError(
            "Classifier guidance needs a trained noisy classifier and labels."
        )
    state = noise.clone()
    times = torch.linspace(1, epsilon, steps + 1, device=state.device)

    def score(x, t):
        value = model(x, t * 1000)
        if guidance:
            with torch.enable_grad():
                leaf = x.detach().requires_grad_(True)
                logits = classifier(leaf, t * 1000)
                selected = logits.log_softmax(-1).gather(1, labels[:, None]).sum()
                gradient = torch.autograd.grad(selected, leaf)[0]
            value = value + guidance * gradient
        return value

    def field(x, t):
        drift, g2 = sde.drift_diffusion(x, t)
        return drift - (1 if solver == "reverse_sde" else 0.5) * g2 * score(x, t)

    for t, next_t in itertools.pairwise(times):
        h = next_t - t
        batch_t = t.expand(len(state))
        velocity = field(state, batch_t)
        proposal = state + h * velocity
        if solver == "reverse_sde":
            _, g2 = sde.drift_diffusion(state, batch_t)
            state = proposal + (-h * g2).sqrt() * torch.randn_like(state)
        elif solver == "heun":
            state = (
                state + h * (velocity + field(proposal, next_t.expand(len(state)))) / 2
            )
        else:
            state = proposal
    t = times[-1].expand(len(state))
    alpha, sigma = sde.marginal_coefficients(t, state)
    return (state + sigma.square() * score(state, t)) / alpha
