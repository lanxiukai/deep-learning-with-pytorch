"""Conditional Gaussian paths and direct velocity ODE generation.

Time runs from standard Gaussian noise at 0 to data at 1. The network
predicts dx/dt, which is distinct from the DDPM diffusion-v target.
Linear independent-endpoint CFM is also the basic rectified-flow objective;
it does not solve a global optimal-transport coupling or perform reflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class GaussianConditionalPath:
    """x_t = alpha(t) * data + sigma(t) * noise, with analytic derivatives."""

    schedule: Literal["linear", "trigonometric"] = "linear"

    def __post_init__(self):
        if self.schedule not in ("linear", "trigonometric"):
            raise ValueError("Choose a linear or trigonometric conditional path.")

    def config(self):
        return {"schedule": self.schedule}

    def coefficients(self, time):
        """Return alpha, sigma, d(alpha)/dt, d(sigma)/dt in physical time."""
        if self.schedule == "linear":
            unit = torch.ones_like(time)
            return time, 1 - time, unit, -unit
        angle = torch.pi * time / 2
        alpha, sigma = angle.sin(), angle.cos()
        return alpha, sigma, (torch.pi / 2) * sigma, -(torch.pi / 2) * alpha

    @torch.no_grad()
    def sample(self, noise, data, time):
        """Construct detached state and conditional velocity for endpoint pairs.

        The lesson pairs independent endpoints. Neither endpoint nor the
        conditional target is a network input; only (state, time) is observed.
        Squared regression learns the posterior mean of conditional velocities.
        """
        if noise.shape != data.shape or time.shape != (len(data),):
            raise ValueError("Endpoints must match and time must have shape (batch,).")
        alpha, sigma, alpha_dot, sigma_dot = (
            value.reshape(-1, *((1,) * (data.ndim - 1)))
            for value in self.coefficients(time)
        )
        state = alpha * data + sigma * noise
        target = alpha_dot * data + sigma_dot * noise
        return state, target

    def score_to_velocity(self, score, state, time):
        """Bridge to the marginal score of this SAME path, for 0 < t < 1.

        This identity is not a conversion from an unrelated DDPM schedule.
        Generation predicts velocity directly and needs no endpoint division.
        """
        if not bool(((time > 0) & (time < 1)).all()):
            raise ValueError("Score conversion requires interior times 0 < t < 1.")
        alpha, sigma, alpha_dot, sigma_dot = (
            value.reshape(-1, *((1,) * (state.ndim - 1)))
            for value in self.coefficients(time)
        )
        rate = alpha_dot / alpha
        return rate * state + (rate * sigma.square() - sigma * sigma_dot) * score


@torch.no_grad()
def sample_flow_model(
    model,
    shape,
    *,
    num_steps=50,
    solver="heun",
    time_embedding_scale=1000.0,
    initial_noise=None,
    generator=None,
    return_trajectory=False,
    trajectory_frames=8,
):
    """Integrate a learned velocity from 0 to 1 on a uniform increasing grid.

    Euler uses K network calls; midpoint and Heun use 2K, including their last
    interval. The time-embedding scale changes the network's input only; it
    does not multiply the velocity or dt. No extra endpoint denoising follows.
    Return raw states: pixel clipping belongs to display/quality postprocessing,
    never to the ODE update. The saved trajectory also remains unclipped.
    """
    if num_steps < 1 or solver not in ("euler", "midpoint", "heun"):
        raise ValueError("Need positive steps and an Euler, midpoint, or Heun solver.")
    parameter = next(model.parameters())
    if initial_noise is None:
        initial_noise = torch.randn(
            shape, device=parameter.device, dtype=parameter.dtype, generator=generator
        )
    if tuple(initial_noise.shape) != tuple(shape):
        raise ValueError("Initial noise must match the requested sample shape.")
    state = initial_noise.to(parameter).clone()
    times = torch.linspace(0, 1, num_steps + 1, device=state.device, dtype=state.dtype)
    path = [state.cpu()] if return_trajectory else []
    capture = set(
        torch.linspace(0, num_steps - 1, min(trajectory_frames - 1, num_steps))
        .round()
        .long()
        .tolist()
    )
    was_training = model.training
    model.eval()
    try:
        for index in range(num_steps):
            time = times[index].expand(shape[0])
            next_time = times[index + 1].expand(shape[0])
            dt = next_time[0] - time[0]
            velocity = model(state, time * time_embedding_scale)
            if solver == "euler":
                state = state + dt * velocity
            elif solver == "midpoint":
                midpoint = state + (dt / 2) * velocity
                middle_time = (time + next_time) / 2
                state = state + dt * model(midpoint, middle_time * time_embedding_scale)
            else:
                proposal = state + dt * velocity
                endpoint_velocity = model(proposal, next_time * time_embedding_scale)
                state = state + (dt / 2) * (velocity + endpoint_velocity)
            if return_trajectory and index in capture:
                path.append(state.cpu())
    finally:
        model.train(was_training)
    if not torch.isfinite(state).all():
        raise FloatingPointError(
            "Flow integration produced non-finite terminal states."
        )
    return state, path
