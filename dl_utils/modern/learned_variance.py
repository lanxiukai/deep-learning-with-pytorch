"""DiT's learned-range variance with a continuous-latent Gaussian endpoint."""

import math

import torch

from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion, _extract, _randn_like


class LearnedVarianceDiffusion(GaussianDiffusion):
    def log_variance(self, head, time, sample):
        lower = self.posterior_variance.clone()
        lower[0] = lower[1]
        minimum = _extract(lower.log(), time, sample)
        maximum = _extract(self.betas.log(), time, sample)
        return minimum + (head + 1) / 2 * (maximum - minimum)

    def _heads(self, model, x, t, labels, guidance):
        epsilon, variance = model(x, t, labels).chunk(2, 1)
        if labels is not None and guidance != 1:
            unconditional = model(x, t, None).chunk(2, 1)[0]
            epsilon = unconditional + guidance * (epsilon - unconditional)
        return epsilon, variance

    def _guided_output(self, model, x_t, timesteps, labels, guidance_scale):
        return self._heads(model, x_t, timesteps, labels, guidance_scale)[0]

    def training_losses(self, model, x0, time, noise, labels=None, vlb_weight=0.001):
        noisy = self.q_sample(x0, time, noise)
        epsilon, head = model(noisy, time, labels).float().chunk(2, 1)
        fraction = (head.detach() + 1) / 2
        self.last_variance_fraction = (fraction.min().item(), fraction.max().item())
        simple = (epsilon - noise).square().flatten(1).mean(1)
        clean = self.convert_model_output(
            noisy, time, epsilon.detach(), clip_x0=None
        ).x0
        mean = self.q_posterior(clean, noisy, time)[0]
        true_mean = self.q_posterior(x0, noisy, time)[0]
        logvar = self.log_variance(head, time, noisy)
        true_logvar = self.log_variance(-torch.ones_like(head), time, noisy)
        kl = 0.5 * (
            logvar
            - true_logvar
            - 1
            + (true_logvar - logvar).exp()
            + (true_mean - mean).square() * (-logvar).exp()
        )
        nll = 0.5 * (
            math.log(2 * math.pi) + logvar + (x0 - mean).square() * (-logvar).exp()
        )
        vb = (
            torch.where(time == 0, nll.flatten(1).mean(1), kl.flatten(1).mean(1))
            * self.num_steps
        )
        return simple + vlb_weight * vb, simple, vb

    def ddpm_step(
        self,
        model,
        x_t,
        timesteps,
        *,
        prediction_type="epsilon",
        labels=None,
        guidance_scale=1.0,
        clip_x0=None,
        generator=None,
    ):
        if prediction_type != "epsilon":
            raise ValueError("DiT predicts epsilon and variance.")
        epsilon, head = self._heads(model, x_t, timesteps, labels, guidance_scale)
        prediction = self.convert_model_output(x_t, timesteps, epsilon, clip_x0=clip_x0)
        mean = self.q_posterior(prediction.x0, x_t, timesteps)[0]
        active = (timesteps > 0)[:, None, None, None]
        return mean + active * (
            0.5 * self.log_variance(head, timesteps, x_t)
        ).exp() * _randn_like(x_t, generator), prediction
