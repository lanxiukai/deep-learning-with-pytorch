r"""Frozen VP diffusion ODE: Euler, Heun, DPM-Solver-1/2M, DPM-Solver++-2M.

Use lambda=log(alpha/sigma), increasing toward data. DPM methods integrate
the linear factor analytically. Second order reuses the preceding prediction;
it costs one new network evaluation per interval after first-order startup.
Reference equations: https://github.com/LuChengTHU/dpm-solver
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F


def vp_coefficients(log_snr_half):
    alpha = torch.exp(-0.5 * F.softplus(-2 * log_snr_half))
    sigma = torch.exp(-0.5 * F.softplus(2 * log_snr_half))
    return alpha, sigma


class DiscreteVPPath:
    """Piecewise-linear log(alpha) extension of a checkpoint's discrete path.

    The trained U-Net sees fractional *index* time, not a normalized time or
    lambda. No extrapolation beyond the first/last training indices is used.
    """

    def __init__(self, diffusion):
        self.log_alpha = diffusion.alpha_bars.double().log() * 0.5
        sigma = (-torch.expm1(2 * self.log_alpha)).sqrt()
        self.lambdas = self.log_alpha - sigma.log()
        self.start, self.end = self.lambdas[-1], self.lambdas[0]

    def model_time(self, log_snr_half):
        target = -0.5 * F.softplus(-2 * log_snr_half.double())
        right = torch.searchsorted(-self.log_alpha, -target).clamp(
            1, len(self.log_alpha) - 1
        )
        left = right - 1
        fraction = (target - self.log_alpha[left]) / (
            self.log_alpha[right] - self.log_alpha[left]
        )
        return (left + fraction).clamp(0, len(self.log_alpha) - 1).float()


class ContinuousVPPath:
    def __init__(self, sde, *, device, time_epsilon=1e-3, embedding_scale=1000.0):
        self.sde, self.embedding_scale = sde, embedding_scale
        times = torch.tensor([1.0, time_epsilon], device=device, dtype=torch.float64)
        alpha, sigma = sde.marginal_coefficients(
            times, torch.empty(2, 1, device=device)
        )
        self.start, self.end = (alpha.log() - sigma.log()).flatten()

    def model_time(self, log_snr_half):
        integral = F.softplus(-2 * log_snr_half)
        low, delta = self.sde.beta_min, self.sde.beta_max - self.sde.beta_min
        # Rationalized quadratic root avoids cancellation at the clean end.
        time = 2 * integral / (low + torch.sqrt(low**2 + 2 * delta * integral))
        return time.float() * self.embedding_scale


def dpm_update(
    state,
    prediction,
    alpha_s,
    sigma_s,
    alpha_t,
    sigma_t,
    log_snr_step,
    *,
    previous_prediction=None,
    previous_h=None,
    data_prediction=False,
):
    """First-order startup or midpoint-type second-order multistep update."""
    estimate = prediction
    if previous_prediction is not None:
        # D1 = (prediction_s - prediction_previous) / (previous_h / h).
        estimate = prediction + 0.5 * log_snr_step / previous_h * (
            prediction - previous_prediction
        )
    if data_prediction:
        return sigma_t / sigma_s * state - alpha_t * torch.expm1(-log_snr_step) * estimate
    return alpha_t / alpha_s * state - sigma_t * torch.expm1(log_snr_step) * estimate


@torch.no_grad()
def sample_diffusion_ode(
    model,
    path,
    initial_noise,
    *,
    solver="dpmpp_2m",
    num_steps=25,
    prediction_type="epsilon",
    clip_x0=False,
):
    """Uniform lambda grid, positive-noise endpoint, then one final x0 estimate.

    With a fixed checkpoint and initial noise all methods are deterministic.
    The last denoising call is part of NFE, unlike the update interval count.
    """
    if solver not in ("euler", "heun", "dpm_1", "dpm_2m", "dpmpp_2m") or num_steps < 2:
        raise ValueError("Choose an implemented solver and at least two intervals.")
    state = initial_noise.clone()
    grid = torch.linspace(
        path.start, path.end, num_steps + 1, device=state.device, dtype=torch.float64
    )

    def predictions(sample, coordinate):
        alpha, sigma = (v.float() for v in vp_coefficients(coordinate))
        output = model(sample, path.model_time(coordinate).expand(len(sample)))
        if output.shape[1] == 2 * sample.shape[1]:
            output = output.chunk(2, dim=1)[0]  # Variance head is unused by ODEs.
        if prediction_type == "epsilon":
            epsilon = output
        elif prediction_type == "x0":
            epsilon = (sample - alpha * output) / sigma
        elif prediction_type == "v":
            epsilon = sigma * sample + alpha * output
        elif prediction_type == "score":
            epsilon = -sigma * output
        else:
            raise ValueError("Unknown prediction type.")
        clean = (sample - sigma * epsilon) / alpha
        if clip_x0:
            clean = clean.clamp(-1, 1)
            epsilon = (sample - alpha * clean) / sigma
        return epsilon, clean

    previous_prediction = previous_h = None
    was_training = model.training
    model.eval()
    try:
        for source, target in pairwise(grid):
            alpha_s, sigma_s = (v.float() for v in vp_coefficients(source))
            alpha_t, sigma_t = (v.float() for v in vp_coefficients(target))
            log_snr_step = (target - source).float()
            epsilon, clean = predictions(state, source)
            if solver in ("euler", "heun"):
                # VP ODE in lambda: dx/dlambda = sigma^2*x - sigma*epsilon.
                velocity = sigma_s.square() * state - sigma_s * epsilon
                proposal = state + log_snr_step * velocity
                if solver == "heun":
                    next_epsilon, _ = predictions(proposal, target)
                    next_velocity = sigma_t.square() * proposal - sigma_t * next_epsilon
                    state = state + 0.5 * log_snr_step * (velocity + next_velocity)
                else:
                    state = proposal
            else:
                value = clean if solver == "dpmpp_2m" else epsilon
                state = dpm_update(
                    state,
                    value,
                    alpha_s,
                    sigma_s,
                    alpha_t,
                    sigma_t,
                    log_snr_step,
                    previous_prediction=previous_prediction
                    if solver != "dpm_1"
                    else None,
                    previous_h=previous_h,
                    data_prediction=solver == "dpmpp_2m",
                )
                previous_prediction, previous_h = value, log_snr_step
        _, state = predictions(state, grid[-1])
    finally:
        model.train(was_training)
    return state.clamp(-1, 1)
