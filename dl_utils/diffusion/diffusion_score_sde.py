"""VP, VE, and sub-VP perturbations with explicit score-based samplers.

Forward time runs from data to noise. Reverse integration uses negative dt;
Brownian variance uses |dt|. Optional Tweedie denoising follows a finite endpoint.
NCSN-style annealed Langevin is restricted to the VE noise path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ScoreSampler = Literal["reverse_sde", "probability_flow", "pc", "annealed_langevin"]


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


@dataclass(frozen=True)
class SubVPSDE(VPSDE):
    """Same drift, reduced diffusion; marginal std is 1-exp(-integral(beta))."""

    def marginal_coefficients(self, time, sample):
        integral = self.integrated_beta(time)
        return _expand((-0.5 * integral).exp(), sample), _expand(
            -torch.expm1(-integral), sample
        )

    def drift_diffusion(self, sample, time):
        beta = _expand(self.beta(time), sample)
        discount = _expand(-torch.expm1(-2 * self.integrated_beta(time)), sample)
        return -0.5 * beta * sample, beta * discount


@dataclass(frozen=True)
class VESDE(VPSDE):
    """Additive geometric noise; the finite sigma_min endpoint is approximate."""

    sigma_min: float = 0.01
    sigma_max: float = 50.0

    def __post_init__(self):
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("Expected 0 < sigma_min < sigma_max.")
        object.__setattr__(self, "prior_std", self.sigma_max)

    def marginal_coefficients(self, time, sample):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** time
        return torch.ones_like(_expand(sigma, sample)), _expand(sigma, sample)

    def drift_diffusion(self, sample, time):
        import math

        _, sigma = self.marginal_coefficients(time, sample)
        return torch.zeros_like(sample), 2 * math.log(
            self.sigma_max / self.sigma_min
        ) * sigma.square()


def make_sde(name, **config):
    return {"vp": VPSDE, "ve": VESDE, "subvp": SubVPSDE}[name](**config)


@torch.no_grad()
def sample_score_model(
    model,
    sde,
    shape,
    *,
    sampler="probability_flow",
    num_steps=250,
    time_epsilon=1e-3,
    ode_solver="heun",
    time_embedding_scale=1000.0,
    final_denoise=True,
    corrector_steps=1,
    langevin_step_size=0.01,
    initial_noise=None,
    generator=None,
    return_trajectory=False,
    trajectory_frames=8,
):
    """Euler-Maruyama, Langevin PC, Euler/Heun PF-ODE, or VE annealed Langevin.

    Langevin uses eta(t) = langevin_step_size * sigma(t)^2. This explicit
    fixed rule is a teaching choice, not the adaptive-SNR corrector recipe.
    At every reverse-SDE interval Brownian noise is retained; the separately
    counted final network evaluation performs the optional finite-endpoint
    denoise. Disabling it returns the state at time_epsilon (then pixel-clipped).
    """
    if num_steps < 2 or not 0 < time_epsilon < 1:
        raise ValueError("Need >=2 steps and 0 < time_epsilon < 1.")
    if sampler not in ("reverse_sde", "pc", "probability_flow", "annealed_langevin"):
        raise ValueError("Unknown score sampler.")
    if ode_solver not in ("euler", "heun"):
        raise ValueError("Unknown ODE solver.")
    if sampler == "annealed_langevin" and not isinstance(sde, VESDE):
        raise ValueError("Annealed Langevin uses the VE/NCSN noise path.")
    if corrector_steps < 1 or langevin_step_size <= 0:
        raise ValueError("Langevin count and step size must be positive.")
    parameter = next(model.parameters())
    device = parameter.device
    if initial_noise is None:
        initial_noise = torch.randn(shape, device=device, generator=generator)
    state = initial_noise.to(device).clone() * sde.prior_std
    annealed = sampler == "annealed_langevin"
    times = torch.linspace(
        1.0, time_epsilon, num_steps if annealed else num_steps + 1, device=device
    )
    path = [state.cpu()] if return_trajectory else []
    capture = set(
        torch.linspace(0, num_steps - 1, min(trajectory_frames - 1, num_steps))
        .round()
        .long()
        .tolist()
    )

    def score(sample, time):
        return model(sample, time * time_embedding_scale)

    def field(sample, time, factor):
        drift, diffusion_squared = sde.drift_diffusion(sample, time)
        return drift - factor * diffusion_squared * score(sample, time)

    was_training = model.training
    model.eval()
    try:
        for index in range(num_steps):
            time = times[index].expand(shape[0])
            next_time = times[min(index + 1, len(times) - 1)].expand(shape[0])
            step_size = next_time[0] - time[0]
            if sampler == "probability_flow":
                velocity = field(state, time, 0.5)
                proposal = state + step_size * velocity
                state = (
                    proposal
                    if ode_solver == "euler"
                    else state + 0.5 * step_size * (velocity + field(proposal, next_time, 0.5))
                )
            elif sampler != "annealed_langevin":
                _, diffusion_squared = sde.drift_diffusion(state, time)
                noise = torch.randn(state.shape, device=device, generator=generator)
                state = state + step_size * field(state, time, 1.0) + (-step_size * diffusion_squared).sqrt() * noise
            if sampler in ("pc", "annealed_langevin"):
                correction_time = next_time if sampler == "pc" else time
                _, sigma = sde.marginal_coefficients(correction_time, state)
                correction_step_size = langevin_step_size * sigma.square()
                for _ in range(corrector_steps):
                    noise = torch.randn(state.shape, device=device, generator=generator)
                    state = (
                        state
                        + correction_step_size * score(state, correction_time)
                        + (2 * correction_step_size).sqrt() * noise
                    )
            if return_trajectory and index in capture:
                path.append(state.cpu())
        if final_denoise:
            final_time = times[-1].expand(shape[0])
            alpha, sigma = sde.marginal_coefficients(final_time, state)
            state = (state + sigma.square() * score(state, final_time)) / alpha
        if path:
            path[-1] = state.cpu()
    finally:
        model.train(was_training)
    return state.clamp(-1, 1), path


# Retain the original VP entry point for importing notebooks.
sample_vp_sde = sample_score_model
__all__ = [
    "VESDE",
    "VPSDE",
    "ScoreSampler",
    "SubVPSDE",
    "make_sde",
    "sample_score_model",
    "sample_vp_sde",
]
