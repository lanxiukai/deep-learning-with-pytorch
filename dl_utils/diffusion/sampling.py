"""Assemble trained lesson samplers without optimizing any model parameters."""

import torch

from dl_utils.diffusion.consistency import sample_consistency
from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_score_sde import VPSDE, sample_score_model
from dl_utils.diffusion.dmd2 import student_rollout
from dl_utils.diffusion.edm import sample_edm
from dl_utils.diffusion.improved_mean_flow import sample_imf
from dl_utils.diffusion.learned_variance import LearnedVarianceDiffusion
from dl_utils.diffusion.solvers import sample_dpmpp, sample_velocity
from dl_utils.diffusion.sr3 import sample_sr3, upsample


def make_sampler(model, metadata, *, codec=None, latent_scale=1.0, classifier=None):
    algorithm = metadata["algorithm"]
    task = algorithm["task"]
    device = next(model.parameters()).device
    latent = algorithm.get("latent", False)
    size = metadata["data_config"]["image_size"]
    channels, spatial = (
        (codec.latent_channels, codec.latent_size) if latent else (3, size)
    )
    diffusion_type = LearnedVarianceDiffusion if task == "dit" else GaussianDiffusion
    diffusion = diffusion_type(**algorithm.get("diffusion", {})).to(device)

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
            noise = torch.randn(count, channels, spatial, spatial, device=device)
        conditional = algorithm.get("conditional", False)
        if not conditional and task != "vp":
            labels = None
        if labels is not None:
            labels = labels.to(device)
        if task in ("consistency", "dmd2", "sr3", "sr_regression") and guidance != 1:
            raise ValueError(
                "This lesson does not train a CFG null branch or a guidance input."
            )
        if (
            task == "sr3"
            and not 0 <= augmentation_level <= algorithm["augmentation_max"]
        ):
            raise ValueError("Condition augmentation is outside the training range.")
        if task in ("ddpm", "ldm", "dit"):
            solver = solver or "ddim"
            if solver == "dpmpp_2m":
                images = sample_dpmpp(
                    model,
                    noise,
                    diffusion=diffusion,
                    labels=labels,
                    guidance=guidance,
                    steps=steps,
                )
            else:
                images, _ = diffusion.sample(
                    model,
                    noise.shape,
                    sampler=solver,
                    num_inference_steps=diffusion.num_steps
                    if solver == "ddpm"
                    else steps,
                    initial_noise=noise,
                    labels=labels,
                    guidance_scale=guidance,
                    clip_x0=None if latent else (-1, 1),
                )
        elif task == "vp":
            images = sample_score_model(
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
        elif task in ("cfm", "sit"):
            images = sample_velocity(
                model,
                noise,
                labels=labels,
                guidance=guidance,
                steps=steps,
                solver=solver or "heun",
                reverse=task == "sit",
            )
        elif task == "edm":
            if guidance != 1:
                raise ValueError(
                    "The EDM baseline has labels but no trained CFG null branch."
                )
            if solver == "dpmpp_2m":
                images = sample_dpmpp(
                    model,
                    noise,
                    labels=labels,
                    steps=steps,
                    sigma_min=algorithm["sigma_min"],
                    sigma_max=algorithm["sigma_max"],
                )
            else:
                images, _, _ = sample_edm(
                    model,
                    noise.shape,
                    initial_noise=noise,
                    labels=labels,
                    num_steps=steps,
                    sigma_min=algorithm["sigma_min"],
                    sigma_max=algorithm["sigma_max"],
                    rho=7,
                    solver=solver or "heun",
                )
        elif task == "consistency":
            images = sample_consistency(
                model, noise, labels, steps=steps, sigma_max=algorithm["sigma_max"]
            )
        elif task == "dmd2":
            if steps != len(algorithm["student_sigmas"]):
                raise ValueError(
                    "DMD2 uses the student grid it was trained on; load a separately trained grid."
                )
            images = student_rollout(
                model, noise, labels, noise.new_tensor(algorithm["student_sigmas"])
            )
        elif task == "imf":
            if not 1 <= guidance <= algorithm["guidance_max"]:
                raise ValueError("Guidance outside the training range.")
            images = sample_imf(
                model,
                noise,
                labels,
                steps=steps,
                omega=guidance,
                lower=lower,
                upper=upper,
            )
        elif task == "sr3":
            if low is None:
                raise ValueError("SR3 requires a low-resolution observation.")
            images = sample_sr3(
                model,
                diffusion,
                noise,
                low,
                labels,
                steps=steps,
                augmentation_level=augmentation_level,
            )
        elif task == "sr_regression":
            if low is None:
                raise ValueError("Regression requires a low-resolution observation.")
            images = model(
                upsample(low, (size, size)), torch.zeros(count, device=device)
            )
        else:
            raise ValueError(f"No free-generation sampler for {task}.")
        return (
            codec.decode_latent(images, latent_scale=latent_scale) if latent else images
        )

    return sample
