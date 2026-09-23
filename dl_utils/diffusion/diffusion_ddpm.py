r"""Discrete variance-preserving diffusion equations and DDPM/DDIM sampling.

This module contains no network architecture and no training loop.  It keeps
the fixed forward process, equivalent prediction parameterizations, analytic
posterior, and two reverse transitions together because they share one indexing
and alpha-bar convention.

Python index ``t=0`` denotes the paper's first noising step.  Consequently
``alpha_bars[t]`` is the one-based :math:`\bar\alpha_{t+1}`, while the synthetic
previous cumulative signal at the clean endpoint is exactly one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PredictionType = Literal["epsilon", "x0", "v", "score"]
DiscreteSampler = Literal["ddpm", "ddim"]

__all__ = [
    "DiffusionPrediction",
    "DiscreteSampler",
    "GaussianDiffusion",
    "PredictionType",
    "cosine_beta_schedule",
    "linear_beta_schedule",
]


def linear_beta_schedule(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> Tensor:
    """The linear variance schedule used by the original DDPM experiments."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("expected 0 < beta_start < beta_end < 1")
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def cosine_beta_schedule(
    num_steps: int,
    offset: float = 0.008,
    max_beta: float = 0.999,
) -> Tensor:
    """Discretize the Improved-DDPM cosine cumulative signal schedule."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if offset < 0.0 or not 0.0 < max_beta < 1.0:
        raise ValueError("offset must be non-negative and max_beta in (0, 1)")

    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    angles = ((steps / num_steps + offset) / (1.0 + offset)) * math.pi / 2.0
    alpha_bars = torch.cos(angles).square()
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(max=max_beta).float()


def _extract(values: Tensor, timesteps: Tensor, sample: Tensor) -> Tensor:
    """Gather one scalar per batch item and add broadcast dimensions."""
    if timesteps.ndim != 1 or timesteps.shape[0] != sample.shape[0]:
        raise ValueError("timesteps must have shape [batch]")
    gathered = values.gather(0, timesteps.long())
    return gathered.reshape(sample.shape[0], *((1,) * (sample.ndim - 1)))


def _randn_like(sample: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        sample.shape,
        device=sample.device,
        dtype=sample.dtype,
        generator=generator,
    )


@dataclass(frozen=True)
class DiffusionPrediction:
    """Equivalent views of a model prediction on a VP diffusion path."""

    epsilon: Tensor
    x0: Tensor
    score: Tensor


class GaussianDiffusion(nn.Module):
    """Fixed-variance VP diffusion with exact DDPM and generalized DDIM steps.

    The class owns only analytic buffers; it has no trainable parameters.  A
    network may predict noise, the clean sample, diffusion velocity, or score.
    Every reverse update first converts that output to a consistent
    ``(epsilon, x0, score)`` triple.
    """

    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor
    alpha_bars_prev: Tensor
    sqrt_alpha_bars: Tensor
    sqrt_one_minus_alpha_bars: Tensor
    posterior_variance: Tensor
    posterior_log_variance: Tensor
    posterior_mean_coef_x0: Tensor
    posterior_mean_coef_xt: Tensor

    def __init__(
        self,
        num_steps: int = 1000,
        beta_schedule: Literal["linear", "cosine"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_offset: float = 0.008,
    ) -> None:
        super().__init__()
        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_steps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_steps, cosine_offset)
        else:
            raise ValueError(f"unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef_x0 = betas * alpha_bars_prev.sqrt() / (1.0 - alpha_bars)
        posterior_mean_coef_xt = (
            alphas.sqrt() * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

        self.num_steps = num_steps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cosine_offset = cosine_offset

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register_buffer("posterior_mean_coef_x0", posterior_mean_coef_x0)
        self.register_buffer("posterior_mean_coef_xt", posterior_mean_coef_xt)

    def config(self) -> dict[str, float | int | str]:
        """Return the minimal constructor configuration for a checkpoint."""
        return {
            "num_steps": self.num_steps,
            "beta_schedule": self.beta_schedule,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "cosine_offset": self.cosine_offset,
        }

    def signal_scale(self, timesteps: Tensor, sample: Tensor) -> Tensor:
        r"""Return :math:`\sqrt{\bar\alpha_t}` with image broadcast shape."""
        return _extract(self.sqrt_alpha_bars, timesteps, sample)

    def noise_scale(self, timesteps: Tensor, sample: Tensor) -> Tensor:
        r"""Return :math:`\sqrt{1-\bar\alpha_t}` with image broadcast shape."""
        return _extract(self.sqrt_one_minus_alpha_bars, timesteps, sample)

    def q_sample(
        self,
        x0: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample ``q(x_t | x_0)`` in one reparameterized operation."""
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape:
            raise ValueError("noise and x0 must have identical shapes")
        alpha = self.signal_scale(timesteps, x0)
        sigma = self.noise_scale(timesteps, x0)
        return alpha * x0 + sigma * noise

    def training_target(
        self,
        x0: Tensor,
        noise: Tensor,
        timesteps: Tensor,
        prediction_type: PredictionType,
    ) -> Tensor:
        """Construct the selected, algebraically equivalent model target."""
        alpha = self.signal_scale(timesteps, x0)
        sigma = self.noise_scale(timesteps, x0)
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x0":
            return x0
        if prediction_type == "v":
            return alpha * noise - sigma * x0
        if prediction_type == "score":
            return -noise / sigma
        raise ValueError(f"unknown prediction type: {prediction_type}")

    def convert_model_output(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        model_output: Tensor,
        prediction_type: PredictionType = "epsilon",
        clip_x0: tuple[float, float] | None = (-1.0, 1.0),
    ) -> DiffusionPrediction:
        """Convert epsilon/x0/v/score output into one consistent prediction."""
        alpha = self.signal_scale(timesteps, x_t)
        sigma = self.noise_scale(timesteps, x_t)

        if prediction_type == "epsilon":
            epsilon = model_output
            x0 = (x_t - sigma * epsilon) / alpha
        elif prediction_type == "x0":
            x0 = model_output
            epsilon = (x_t - alpha * x0) / sigma
        elif prediction_type == "v":
            x0 = alpha * x_t - sigma * model_output
            epsilon = sigma * x_t + alpha * model_output
        elif prediction_type == "score":
            epsilon = -sigma * model_output
            x0 = (x_t - sigma * epsilon) / alpha
        else:
            raise ValueError(f"unknown prediction type: {prediction_type}")

        if clip_x0 is not None:
            x0 = x0.clamp(*clip_x0)
            # Clipping changes x0; recomputing epsilon keeps the DDIM direction
            # and every returned parameterization mutually consistent.
            epsilon = (x_t - alpha * x0) / sigma
        score = -epsilon / sigma
        return DiffusionPrediction(epsilon=epsilon, x0=x0, score=score)

    def _guided_output(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        labels: Tensor | None,
        guidance_scale: float,
    ) -> Tensor:
        if labels is None:
            return model(x_t, timesteps, None)
        conditional = model(x_t, timesteps, labels)
        if guidance_scale == 1.0:
            return conditional
        unconditional = model(x_t, timesteps, None)
        return unconditional + guidance_scale * (conditional - unconditional)

    def model_predictions(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        prediction_type: PredictionType = "epsilon",
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: tuple[float, float] | None = (-1.0, 1.0),
    ) -> DiffusionPrediction:
        output = self._guided_output(model, x_t, timesteps, labels, guidance_scale)
        return self.convert_model_output(
            x_t, timesteps, output, prediction_type, clip_x0
        )

    def q_posterior(
        self,
        x0: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean, variance, and log variance of ``q(x_{t-1}|x_t,x0)``."""
        mean = (
            _extract(self.posterior_mean_coef_x0, timesteps, x_t) * x0
            + _extract(self.posterior_mean_coef_xt, timesteps, x_t) * x_t
        )
        variance = _extract(self.posterior_variance, timesteps, x_t)
        log_variance = _extract(self.posterior_log_variance, timesteps, x_t)
        return mean, variance, log_variance

    def ddpm_step(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        prediction_type: PredictionType = "epsilon",
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: tuple[float, float] | None = (-1.0, 1.0),
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, DiffusionPrediction]:
        """One ancestral ``p_theta(x_{t-1}|x_t)`` update."""
        prediction = self.model_predictions(
            model,
            x_t,
            timesteps,
            prediction_type,
            labels,
            guidance_scale,
            clip_x0,
        )
        mean, variance, _ = self.q_posterior(prediction.x0, x_t, timesteps)
        noise = _randn_like(x_t, generator)
        has_previous = (timesteps > 0).reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        x_previous = mean + has_previous * variance.sqrt() * noise
        return x_previous, prediction

    def ddim_step(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        previous_timesteps: Tensor,
        *,
        prediction_type: PredictionType = "epsilon",
        eta: float = 0.0,
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: tuple[float, float] | None = (-1.0, 1.0),
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, DiffusionPrediction]:
        """One generalized DDIM update, including skipped timesteps."""
        if eta < 0.0:
            raise ValueError("eta must be non-negative")
        prediction = self.model_predictions(
            model,
            x_t,
            timesteps,
            prediction_type,
            labels,
            guidance_scale,
            clip_x0,
        )

        alpha_bar_t = _extract(self.alpha_bars, timesteps, x_t)
        safe_previous = previous_timesteps.clamp(min=0)
        alpha_bar_previous = _extract(self.alpha_bars, safe_previous, x_t)
        clean_endpoint = (previous_timesteps < 0).reshape(
            x_t.shape[0], *((1,) * (x_t.ndim - 1))
        )
        alpha_bar_previous = torch.where(
            clean_endpoint, torch.ones_like(alpha_bar_previous), alpha_bar_previous
        )

        posterior_ratio = (
            (1.0 - alpha_bar_previous)
            / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_previous)
        )
        stochastic_std = eta * posterior_ratio.clamp(min=0.0).sqrt()
        direction_scale = (
            (1.0 - alpha_bar_previous - stochastic_std.square()).clamp(min=0.0).sqrt()
        )
        x_previous = (
            alpha_bar_previous.sqrt() * prediction.x0
            + direction_scale * prediction.epsilon
        )
        if eta > 0.0:
            x_previous = x_previous + stochastic_std * _randn_like(x_t, generator)
        return x_previous, prediction

    def inference_timesteps(self, num_inference_steps: int) -> Tensor:
        """Return a descending subsequence containing both noisy endpoints."""
        if not 2 <= num_inference_steps <= self.num_steps:
            raise ValueError("num_inference_steps must be in [2, num_train_steps]")
        timesteps = (
            torch.linspace(
                self.num_steps - 1,
                0,
                num_inference_steps,
                device=self.betas.device,
            )
            .round()
            .long()
        )
        if torch.unique_consecutive(timesteps).numel() != num_inference_steps:
            raise RuntimeError("rounded inference schedule contains duplicates")
        return timesteps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Sequence[int],
        *,
        sampler: DiscreteSampler = "ddim",
        prediction_type: PredictionType = "epsilon",
        num_inference_steps: int | None = None,
        eta: float = 0.0,
        labels: Tensor | None = None,
        guidance_scale: float = 1.0,
        clip_x0: tuple[float, float] | None = (-1.0, 1.0),
        initial_noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
        trajectory_frames: int = 8,
    ) -> tuple[Tensor, list[Tensor]]:
        """Generate samples with ancestral DDPM or a shortened DDIM path.

        DDPM is kept on all adjacent training steps because skipping changes its
        posterior.  DDIM explicitly defines the non-adjacent transition and may
        therefore use a shorter inference schedule.
        """
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("model must contain parameters") from error
        if self.betas.device != device:
            raise ValueError("move diffusion and model to the same device")

        shape = tuple(shape)
        if len(shape) != 4 or shape[0] < 1:
            raise ValueError("shape must be [batch, channels, height, width]")
        if initial_noise is None:
            x_t = torch.randn(
                shape,
                device=device,
                dtype=self.betas.dtype,
                generator=generator,
            )
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError("initial_noise does not match requested shape")
            x_t = initial_noise.to(device=device, dtype=self.betas.dtype).clone()
        if labels is not None:
            labels = labels.to(device=device, dtype=torch.long)
            if labels.shape != (shape[0],):
                raise ValueError("labels must have shape [batch]")

        if sampler == "ddpm":
            if num_inference_steps not in (None, self.num_steps):
                raise ValueError("DDPM sampling uses every adjacent training step")
            steps = list(range(self.num_steps - 1, -1, -1))
        elif sampler == "ddim":
            count = num_inference_steps or min(50, self.num_steps)
            steps = self.inference_timesteps(count).tolist()
        else:
            raise ValueError(f"unknown sampler: {sampler}")

        capture_positions: set[int] = set()
        trajectory: list[Tensor] = []
        if return_trajectory:
            if trajectory_frames < 2:
                raise ValueError("trajectory_frames must be at least 2")
            trajectory.append(x_t.detach().cpu())
            capture_count = min(trajectory_frames - 1, len(steps))
            capture_positions = set(
                torch.linspace(0, len(steps) - 1, capture_count).round().long().tolist()
            )

        was_training = model.training
        model.eval()
        try:
            for position, step in enumerate(steps):
                timesteps = torch.full(
                    (shape[0],), step, device=device, dtype=torch.long
                )
                if sampler == "ddpm":
                    x_t, _ = self.ddpm_step(
                        model,
                        x_t,
                        timesteps,
                        prediction_type=prediction_type,
                        labels=labels,
                        guidance_scale=guidance_scale,
                        clip_x0=clip_x0,
                        generator=generator,
                    )
                else:
                    previous = steps[position + 1] if position + 1 < len(steps) else -1
                    previous_timesteps = torch.full_like(timesteps, previous)
                    x_t, _ = self.ddim_step(
                        model,
                        x_t,
                        timesteps,
                        previous_timesteps,
                        prediction_type=prediction_type,
                        eta=eta,
                        labels=labels,
                        guidance_scale=guidance_scale,
                        clip_x0=clip_x0,
                        generator=generator,
                    )
                if return_trajectory and position in capture_positions:
                    trajectory.append(x_t.detach().cpu())
        finally:
            model.train(was_training)

        if clip_x0 is not None:
            x_t = x_t.clamp(*clip_x0)
        return x_t, trajectory
