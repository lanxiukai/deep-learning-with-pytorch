"""DPM-Solver++-2M in the correct VP or additive-EDM coordinates."""

import itertools

import torch
import torch.nn.functional as F


def guided_prediction(model, x, time, labels, guidance):
    value = model(x, time, labels)
    if labels is not None and guidance != 1:
        empty = model(x, time, None)
        value = empty + guidance * (value - empty)
    return value


@torch.no_grad()
def sample_dpmpp(
    model,
    noise,
    *,
    diffusion=None,
    labels=None,
    guidance=1.0,
    steps=25,
    sigma_min=0.002,
    sigma_max=80.0,
):
    if steps < 1:
        raise ValueError("Need at least one integration interval.")
    if diffusion is None:
        endpoints = noise.new_tensor([-sigma_max, -sigma_min]).abs().log().neg()
        state = noise * sigma_max
    else:
        log_alpha = diffusion.alpha_bars.double().log() / 2
        lambdas = log_alpha - (-torch.expm1(2 * log_alpha)).sqrt().log()
        endpoints = torch.stack((lambdas[-1], lambdas[0])).to(noise.device)
        state = noise.clone()
    grid = torch.linspace(endpoints[0], endpoints[1], steps + 1, device=noise.device)

    def coefficients(lam):
        if diffusion is None:
            return torch.ones_like(lam), (-lam).exp()
        return (-0.5 * F.softplus(-2 * lam)).exp(), (-0.5 * F.softplus(2 * lam)).exp()

    def denoise(x, lam):
        a, b = coefficients(lam)
        if diffusion is None:
            return model(x, b.expand(len(x)), labels)
        target = -0.5 * F.softplus(-2 * lam.double())
        right = torch.searchsorted(-log_alpha, -target).clamp(1, len(log_alpha) - 1)
        left = right - 1
        fraction = (target - log_alpha[left]) / (log_alpha[right] - log_alpha[left])
        time = (left + fraction).float().expand(len(x))
        epsilon = guided_prediction(model, x, time, labels, guidance)
        if epsilon.shape[1] == 2 * x.shape[1]:
            epsilon = epsilon.chunk(2, 1)[0]
        return (x - b * epsilon) / a

    previous = previous_h = None
    for source, target in itertools.pairwise(grid):
        a_t, b_t = coefficients(target)
        _, b_s = coefficients(source)
        h = target - source
        clean = denoise(state, source)
        estimate = (
            clean
            if previous is None
            else clean + 0.5 * h / previous_h * (clean - previous)
        )
        state = b_t / b_s * state - a_t * torch.expm1(-h) * estimate
        previous, previous_h = clean, h
    # A finite-noise terminal prediction is an additional network evaluation.
    return denoise(state, grid[-1])


@torch.no_grad()
def sample_velocity(
    model,
    noise,
    *,
    labels=None,
    guidance=1.0,
    steps=50,
    solver="heun",
    reverse=False,
    sde_strength=0.2,
    epsilon=1e-3,
):
    if steps < 1 or solver not in ("euler", "heun", "sde"):
        raise ValueError("Invalid velocity sampler.")
    if solver == "sde" and not reverse:
        raise ValueError(
            "The SDE comparison is defined for the SiT data-to-noise path."
        )
    end = epsilon if solver == "sde" else (0 if reverse else 1)
    times = torch.linspace(1 if reverse else 0, end, steps + 1, device=noise.device)
    state = noise.clone()
    for t, next_t in itertools.pairwise(times):
        velocity = guided_prediction(
            model, state, t.expand(len(state)) * 1000, labels, guidance
        )
        h = next_t - t
        if solver == "sde":
            score = -(state + (1 - t) * velocity) / t
            state = (
                state
                + h * (velocity - 0.5 * sde_strength**2 * score)
                + (-h).sqrt() * sde_strength * torch.randn_like(state)
            )
        else:
            proposal = state + h * velocity
            if solver == "heun":
                last = guided_prediction(
                    model, proposal, next_t.expand(len(state)) * 1000, labels, guidance
                )
                state = state + h * (velocity + last) / 2
            else:
                state = proposal
    if solver == "sde":
        velocity = guided_prediction(
            model, state, times[-1].expand(len(state)) * 1000, labels, guidance
        )
        state = state - times[-1] * velocity
    return state
