r"""EDM: preconditioning, log-normal training noise, weighted denoising, EMA.

The training noise distribution is independent of the rho-shaped sampling
grid. 2.2 compares Euler/Heun using this same frozen EMA denoiser.
"""

from __future__ import annotations

import argparse
import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.edm import EDMPreconditioner, sample_edm
from dl_utils.diffusion.lesson_utils import (
    NoiseLossBins,
    add_training_arguments,
    append_record,
    make_image_loader,
    prepare_output,
    restore_checkpoint,
    save_checkpoint,
    training_metadata,
)
from dl_utils.diffusion.quality import DiffusionQualityMonitor
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.optimization import update_ema


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "edm")
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--log-sigma-mean", type=float, default=-1.2)
    parser.add_argument("--log-sigma-std", type=float, default=1.2)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    return parser.parse_args()


def edm_loss(model, clean, log_sigma_mean, log_sigma_std):
    log_sigma = log_sigma_mean + log_sigma_std * torch.randn(
        len(clean), device=clean.device
    )
    sigma = log_sigma.exp()
    noisy = clean + sigma[:, None, None, None] * torch.randn_like(clean)
    denoised = model(noisy, sigma)
    weight = (sigma.square() + model.sigma_data**2) / (
        sigma * model.sigma_data
    ).square()
    per_image = weight * (denoised - clean).square().flatten(1).mean(1)
    return per_image, log_sigma


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_image_loader(args, device)
    network = DiffusionUNet(
        image_size=args.image_size, hidden_dims=args.hidden_dims, dropout=args.dropout
    )
    model = EDMPreconditioner(network, sigma_data=args.sigma_data).to(device)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metadata = {
        "algorithm": "edm",
        "training_noise": {
            "distribution": "log_normal",
            "log_sigma_mean": args.log_sigma_mean,
            "log_sigma_std": args.log_sigma_std,
        },
        "training": training_metadata(args),
    }
    start = restore_checkpoint(args.resume_from, model, averaged, optimizer, **metadata)
    prepare_output(args)
    monitor = DiffusionQualityMonitor(args, device) if args.eval_every else None

    def sample_batch(count, generator):
        return sample_edm(
            averaged,
            (count, 3, args.image_size, args.image_size),
            num_steps=args.sampling_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            solver=args.solver,
            generator=generator,
        )[0]

    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = NoiseLossBins()
        for clean, _ in tqdm(loader, desc=f"EDM {epoch}/{args.epochs}"):
            clean = clean.to(device, non_blocking=True)
            per_image, log_sigma = edm_loss(
                model, clean, args.log_sigma_mean, args.log_sigma_std
            )
            optimizer.zero_grad(set_to_none=True)
            per_image.mean().backward()
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)
            # Three bins: log(sigma)<-2, [-2,0), >=0. No sampling-grid weighting.
            meter.update(per_image, (log_sigma + 4) / 6)
        append_record(
            args.output_dir / "training.jsonl",
            {
                "epoch": epoch,
                "noise_coordinate": "log_sigma bins (-inf,-2),[-2,0),[0,inf)",
                **meter.result(),
            },
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
            monitor.evaluate(
                averaged,
                sample_batch,
                name=args.solver,
                epoch=epoch,
                model_state="ema",
                sampler=args.solver,
                sampling_steps=args.sampling_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                endpoint="Euler to zero; no vector field at sigma=0",
            )


if __name__ == "__main__":
    train(parse_args())
