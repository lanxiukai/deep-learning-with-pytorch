"""Learned reverse variance and hybrid loss, separate from fixed-variance DDPM.

The raw second head defines r=(v+1)/2 without a sigmoid or clamp, following
Improved DDPM. At index 0, use the index-1 posterior variance as the finite
lower log-variance endpoint and an 8-bit discretized Gaussian likelihood.
"""

from __future__ import annotations

import math

import torch

from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion, _extract, _randn_like


def discretized_gaussian_nll(values, mean, log_variance):
    """Per-image bits/dimension for 256 bins over [-1, 1], with edge tails."""
    inverse_std = torch.exp(-0.5 * log_variance)
    plus = (values - mean + 1.0 / 255.0) * inverse_std
    minus = (values - mean - 1.0 / 255.0) * inverse_std
    cdf_plus = torch.special.ndtr(plus)
    cdf_minus = torch.special.ndtr(minus)
    # Survival functions avoid cancellation in the upper Gaussian tail.
    middle = torch.where(
        minus > 0,
        torch.special.ndtr(-minus) - torch.special.ndtr(-plus),
        cdf_plus - cdf_minus,
    )
    mass = torch.where(
        values < -0.999, cdf_plus, torch.where(values > 0.999, torch.special.ndtr(-minus), middle)
    )
    return -mass.clamp_min(1e-12).log().flatten(1).mean(1) / math.log(2)


class ImprovedDDPM(GaussianDiffusion):
    """Same forward path; the U-Net predicts C epsilon and C variance channels."""

    def log_variance(self, raw_variance, timesteps, sample):
        lower = self.posterior_variance.clone()
        lower[0] = lower[1]
        minimum = _extract(lower.log(), timesteps, sample)
        maximum = _extract(self.betas.log(), timesteps, sample)
        fraction = (raw_variance + 1.0) / 2.0
        return minimum + fraction * (maximum - minimum)

    def _guided_output(self, model, x_t, timesteps, labels, guidance_scale):
        # DDIM uses only the prediction head; its eta defines its own variance.
        if labels is not None or guidance_scale != 1.0:
            raise ValueError("The Improved DDPM lesson is unconditional.")
        return model(x_t, timesteps).chunk(2, dim=1)[0]

    def training_losses(self, model, x0, timesteps, noise, *, vlb_weight=0.001):
        noisy = self.q_sample(x0, timesteps, noise)
        epsilon, variance_head = model(noisy, timesteps).chunk(2, dim=1)
        simple = (epsilon - noise).square().flatten(1).mean(1)
        # Stop gradient through the mean, including its epsilon-to-x0 conversion.
        predicted = self.convert_model_output(
            noisy, timesteps, epsilon.detach(), clip_x0=None
        )
        model_mean, _, _ = self.q_posterior(predicted.x0, noisy, timesteps)
        true_mean, _, _ = self.q_posterior(x0, noisy, timesteps)
        model_logvar = self.log_variance(variance_head, timesteps, noisy)
        true_logvar = self.log_variance(
            -torch.ones_like(variance_head), timesteps, noisy
        )
        kl = 0.5 * (
            model_logvar
            - true_logvar
            - 1.0
            + torch.exp(true_logvar - model_logvar)
            + (true_mean - model_mean).square() * torch.exp(-model_logvar)
        )
        kl = kl.flatten(1).mean(1) / math.log(2)
        nll = discretized_gaussian_nll(x0, model_mean, model_logvar)
        vb_term = torch.where(timesteps == 0, nll, kl)
        # Uniform t estimates the sum of T trainable VLB terms. The constant
        # prior KL is omitted here; this diagnostic is not a full likelihood.
        vb_sum = self.num_steps * vb_term
        return simple + vlb_weight * vb_sum, simple, vb_sum

    def ddpm_step(
        self,
        model,
        x_t,
        timesteps,
        *,
        prediction_type="epsilon",
        labels=None,
        guidance_scale=1.0,
        clip_x0=(-1.0, 1.0),
        generator=None,
    ):
        if prediction_type != "epsilon" or labels is not None or guidance_scale != 1.0:
            raise ValueError("Improved DDPM uses unconditional epsilon prediction.")
        epsilon, variance_head = model(x_t, timesteps).chunk(2, dim=1)
        prediction = self.convert_model_output(x_t, timesteps, epsilon, clip_x0=clip_x0)
        mean, _, _ = self.q_posterior(prediction.x0, x_t, timesteps)
        logvar = self.log_variance(variance_head, timesteps, x_t)
        active = (timesteps > 0).view(-1, 1, 1, 1)
        return mean + active * torch.exp(0.5 * logvar) * _randn_like(
            x_t, generator
        ), prediction
