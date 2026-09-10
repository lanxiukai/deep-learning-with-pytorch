r"""DDPM: q(x_t|x_0), epsilon regression, EMA, and free-generation monitoring.

Default: full CelebA train split, 128x128 RGB, fixed posterior variance,
linear beta schedule. Optional x0/v/score targets expose VP conversions.
Follow with 1.1 DDIM, which changes generation while keeping this model fixed.
"""

from __future__ import annotations

import argparse
import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.lesson_utils import (
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
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.optimization import update_ema


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "ddpm")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--beta-schedule", choices=("linear", "cosine"), default="linear"
    )
    parser.add_argument(
        "--prediction-type", choices=("epsilon", "x0", "v", "score"), default="epsilon"
    )
    parser.add_argument("--eval-sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--ddim-steps", type=int, default=50)
    return parser.parse_args()


def denoising_loss(model, diffusion, clean, prediction_type):
    # Python index 0 represents the paper's first NOISY state, not clean x_0.
    with torch.no_grad():
        time = torch.randint(diffusion.num_steps, (len(clean),), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = diffusion.q_sample(clean, time, noise)
        target = diffusion.training_target(clean, noise, time, prediction_type)
    residual = model(noisy, time) - target
    if prediction_type == "score":
        residual = diffusion.noise_scale(time, clean) * residual
    per_image = residual.square().flatten(1).mean(1)
    return per_image, time


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_image_loader(args, device)
    model = DiffusionUNet(
        image_size=args.image_size, hidden_dims=args.hidden_dims, dropout=args.dropout
    ).to(device)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(
        num_steps=args.num_steps, beta_schedule=args.beta_schedule
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metadata = {
        "algorithm": "vp_ddpm",
        "prediction_type": args.prediction_type,
        "variance_type": "fixed_posterior",
        "diffusion_config": diffusion.config(),
        "training": training_metadata(args),
    }
    start = restore_checkpoint(args.resume_from, model, averaged, optimizer, **metadata)
    prepare_output(args)
    monitor = DiffusionQualityMonitor(args, device) if args.eval_every else None
    print(
        f"CelebA training_batches={len(loader)}; RGB={args.image_size}; parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    def sample_batch(count, generator):
        shape = (count, 3, args.image_size, args.image_size)
        return diffusion.sample(
            averaged,
            shape,
            sampler=args.eval_sampler,
            prediction_type=args.prediction_type,
            num_inference_steps=args.ddim_steps
            if args.eval_sampler == "ddim"
            else None,
            generator=generator,
        )[0]

    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        gradient_sum = 0.0
        for clean, _ in tqdm(loader, desc=f"DDPM {epoch}/{args.epochs}"):
            clean = clean.to(device, non_blocking=True)
            per_image, time = denoising_loss(
                model, diffusion, clean, args.prediction_type
            )
            loss = per_image.mean()  # L_simple, mean over images and pixels.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_sum += float(
                torch.nn.utils.get_total_norm(
                    [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ],
                    error_if_nonfinite=True,
                )
            )
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)
            meter.update(per_image, time / (diffusion.num_steps - 1))
        append_record(
            args.output_dir / "training.jsonl",
            {
                "epoch": epoch,
                "noise_coordinate": "t/(T-1)",
                "gradient_norm": gradient_sum / len(loader),
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
                name=args.eval_sampler,
                epoch=epoch,
                model_state="ema",
                sampler=args.eval_sampler,
                sampling_steps=args.ddim_steps
                if args.eval_sampler == "ddim"
                else args.num_steps,
                eta=0.0,
                prediction_type=args.prediction_type,
                variance_type="fixed_posterior",
            )


if __name__ == "__main__":
    train(parse_args())
