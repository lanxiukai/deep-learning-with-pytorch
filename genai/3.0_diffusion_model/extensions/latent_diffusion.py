r"""Latent diffusion extension: freeze an f=8 KL stage, learn DDPM in its latents.

128px images become 4x16x16 latents and decode back to 128px. The latent
scale and checkpoint identity belong to the frozen first stage. Never clip
latents to the pixel range. Optional Smiling conditioning teaches CFG here,
after the unconditional pixel diffusion main line. Quality uses decoded RGB.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import cast

import torch
from tqdm import tqdm

from dl_utils.data.datasets.celeba import CelebAAlignedDataset
from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.lesson_utils import (
    OUTPUT_ROOT,
    BinnedLoss,
    add_training_arguments,
    append_record,
    make_image_loader,
    prepare_output,
    restore_checkpoint,
    save_checkpoint,
    training_metadata,
)
from dl_utils.diffusion.quality import DiffusionQualityMonitor
from dl_utils.filesystem.directories import reset_dir
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.ema import update_ema
from dl_utils.vae.perceptual_autoencoder import KLPerceptualAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "latent_diffusion")
    parser.set_defaults(batch_size=16, hidden_dims=(128, 256, 384))
    parser.add_argument("--mode", choices=("train", "sample"), default="train")
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        default=OUTPUT_ROOT / "kl_autoencoder" / "kl_autoencoder.pth",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_ROOT / "latent_diffusion" / "latest.pth",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--beta-schedule", choices=("linear", "cosine"), default="linear"
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    return parser.parse_args()


def load_autoencoder(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("model_name") != "kl_perceptual_autoencoder"
        or checkpoint.get("dataset") != "CelebA"
    ):
        raise ValueError(
            "Use the CelebA first-stage checkpoint from kl_autoencoder.py."
        )
    config = checkpoint["model_config"]
    model = KLPerceptualAutoencoder(**config).to(device)
    if model.image_size < 128:
        raise ValueError("The first stage must decode images at least 128px wide.")
    model.load_state_dict(checkpoint["state_dict"])
    scale = float(checkpoint["latent_interface"]["latent_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid first-stage latent scale.")
    interface = {
        "autoencoder_config": config,
        "latent_scale": scale,
        "autoencoder_interface_id": checkpoint["interface_id"],
    }
    return model.eval().requires_grad_(False), interface


def latent_loss(model, diffusion, latent, class_labels, condition_dropout):
    time = torch.randint(diffusion.num_steps, (len(latent),), device=latent.device)
    noise = torch.randn_like(latent)
    labels = None
    if model.num_classes is not None:
        labels = class_labels.clone()
        labels[torch.rand(len(latent), device=latent.device) < condition_dropout] = (
            model.null_class
        )
    prediction = model(diffusion.q_sample(latent, time, noise), time, labels)
    return (prediction - noise).square().flatten(1).mean(1), time


def main():
    args = parse_args()
    if not 0 <= args.condition_dropout < 1:
        raise ValueError("condition_dropout must lie in [0,1).")
    set_seed(args.seed)
    device = try_gpu()
    first_stage, interface = load_autoencoder(args.autoencoder_checkpoint, device)
    args.image_size = first_stage.image_size
    scale = interface["latent_scale"]
    if args.mode == "sample":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for key, value in {"algorithm": "latent_vp_ddpm", **interface}.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Latent checkpoint has a different {key}.")
        averaged = DiffusionUNet(**checkpoint["model_config"]).to(device)
        averaged.load_state_dict(checkpoint["ema_state"])
        averaged.eval().requires_grad_(False)
        diffusion = GaussianDiffusion(**checkpoint["diffusion_config"]).to(device)
        # The training directory can hold the checkpoint; reset only its sample child.
        args.output_dir = args.output_dir / "samples"
        if any(
            path.resolve().is_relative_to(args.output_dir.resolve())
            for path in (args.checkpoint, args.autoencoder_checkpoint)
        ):
            raise ValueError("Sample output must not contain a source checkpoint.")
        reset_dir(str(args.output_dir))
    else:
        loader = make_image_loader(args, device)
        model = DiffusionUNet(
            image_size=first_stage.latent_size,
            in_channels=first_stage.latent_channels,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            num_classes=2 if args.conditional else None,
        ).to(device)
        averaged = copy.deepcopy(model).eval().requires_grad_(False)
        diffusion = GaussianDiffusion(
            num_steps=args.num_steps, beta_schedule=args.beta_schedule
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        metadata: dict = dict(
            algorithm="latent_vp_ddpm",
            prediction_type="epsilon",
            **interface,
            diffusion_config=diffusion.config(),
            training=training_metadata(args),
            condition="Smiling" if args.conditional else None,
            condition_dropout=args.condition_dropout,
        )
        start = restore_checkpoint(
            args.resume_from, model, averaged, optimizer, **metadata
        )
        if args.autoencoder_checkpoint.resolve().is_relative_to(
            args.output_dir.resolve()
        ):
            raise ValueError(
                "Training output must not contain the frozen first-stage checkpoint."
            )
        prepare_output(args)

    monitor = (
        DiffusionQualityMonitor(args, device)
        if args.eval_every or args.mode == "sample"
        else None
    )
    if averaged.num_classes is None and args.guidance_scale != 1.0:
        raise ValueError("Guidance requires a conditional checkpoint.")

    def sample_batch(count, generator):
        # Draw the condition from the empirical training prior; balanced labels
        # would change the distribution compared with an unconditional reference.
        labels = None
        if averaged.num_classes is not None:
            probability = cast(float, label_probability)
            labels = (
                torch.rand(count, device=device, generator=generator) < probability
            ).long()
        latent, _ = diffusion.sample(
            averaged,
            (
                count,
                first_stage.latent_channels,
                first_stage.latent_size,
                first_stage.latent_size,
            ),
            sampler=args.sampler,
            num_inference_steps=args.ddim_steps if args.sampler == "ddim" else None,
            labels=labels,
            guidance_scale=args.guidance_scale,
            clip_x0=None,
            generator=generator,
        )
        return first_stage.decode_latent(latent, latent_scale=scale)

    if args.mode == "sample":
        label_probability = checkpoint["label_probability"]
    else:
        label_probability = (
            torch.tensor(cast(CelebAAlignedDataset, loader.dataset).targets)
            .float()
            .mean()
            .item()
            if args.conditional
            else None
        )
        metadata["label_probability"] = label_probability

    def evaluate(epoch=None):
        if monitor is not None:
            monitor.evaluate(
                averaged,
                sample_batch,
                name=args.sampler,
                epoch=epoch,
                model_state="ema",
                sampler=args.sampler,
                sampling_steps=args.ddim_steps
                if args.sampler == "ddim"
                else diffusion.num_steps,
                first_stage_id=interface["autoencoder_interface_id"],
                latent_scale=scale,
                guidance_scale=args.guidance_scale,
                label_probability=label_probability,
                latency_scope="denoiser plus RGB decoder",
            )

    if args.mode == "sample":
        evaluate()
        return
    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        for images, labels in tqdm(loader, desc=f"Latent DDPM {epoch}/{args.epochs}"):
            with torch.no_grad():
                latent = first_stage.encode_latent(
                    images.to(device), sample=True, latent_scale=scale
                )
            per_image, time = latent_loss(
                model, diffusion, latent, labels.to(device), args.condition_dropout
            )
            optimizer.zero_grad(set_to_none=True)
            per_image.mean().backward()
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)
            meter.update(per_image, time / (diffusion.num_steps - 1))
        append_record(
            args.output_dir / "training.jsonl",
            {"epoch": epoch, "noise_coordinate": "t/(T-1)", **meter.result()},
        )
        save_checkpoint(
            args.output_dir / "latest.pth",
            model,
            averaged,
            optimizer,
            epoch,
            **metadata,
        )
        if monitor and (epoch % args.eval_every == 0 or epoch == args.epochs):
            evaluate(epoch)


if __name__ == "__main__":
    main()
