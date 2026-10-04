"""Guided predictions and ordinary velocity integration shared by both series."""

import itertools

import torch


def guided_prediction(model, x, time, labels, guidance):
    value = model(x, time, labels)
    if labels is not None and guidance != 1:
        empty = model(x, time, None)
        value = empty + guidance * (value - empty)
    return value


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
