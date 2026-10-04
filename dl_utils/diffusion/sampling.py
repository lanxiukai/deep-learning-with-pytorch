"""DDPM, VP-score and linear-CFM sampling for the foundation series."""

import torch

from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_score_sde import VPSDE, sample_score_model
from dl_utils.diffusion.solvers import sample_velocity


def make_sampler(model, metadata, *, codec=None, latent_scale=1.0, classifier=None):
    algorithm = metadata["algorithm"]
    task = algorithm["task"]
    if task not in ("ddpm", "vp", "cfm") or codec is not None:
        raise ValueError("Use the modern visual generation sampler for this model.")
    device = next(model.parameters()).device
    size = metadata["data_config"]["image_size"]
    diffusion = GaussianDiffusion(**algorithm.get("diffusion", {})).to(device)

    @torch.no_grad()
    def sample(
        count,
        *,
        labels=None,
        steps=50,
        solver=None,
        guidance=1.0,
        low=None,
        augmentation_level=0.0,
        noise=None,
        lower=0.0,
        upper=1.0,
    ):
        if noise is None:
            noise = torch.randn(count, 3, size, size, device=device)
        if not algorithm.get("conditional", False) and task != "vp":
            labels = None
        if labels is not None:
            labels = labels.to(device)
        if task == "ddpm":
            solver = solver or "ddim"
            images, _ = diffusion.sample(
                model,
                noise.shape,
                sampler=solver,
                num_inference_steps=diffusion.num_steps if solver == "ddpm" else steps,
                initial_noise=noise,
                labels=labels,
                guidance_scale=guidance,
                clip_x0=(-1, 1),
            )
            return images
        if task == "vp":
            return sample_score_model(
                model,
                VPSDE(**algorithm["sde"]),
                noise,
                steps=steps,
                solver=solver or "heun",
                epsilon=algorithm["time_epsilon"],
                classifier=classifier,
                labels=labels,
                guidance=guidance if labels is not None else 0.0,
            )
        return sample_velocity(
            model,
            noise,
            labels=labels,
            guidance=guidance,
            steps=steps,
            solver=solver or "heun",
            reverse=False,
        )

    return sample
