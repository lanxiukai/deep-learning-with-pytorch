r"""Score-SDE: continuous VP/VE/sub-VP denoising score matching at 128px.

Learn s_theta with sigma^2 ||s_theta + epsilon/sigma||^2. Data is at time 0,
noise at time 1. Freeze that score and compare reverse SDE with Euler/Heun
probability-flow ODE in 2.1, using the same finite endpoint and prior.
"""

from __future__ import annotations

import argparse
import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.diffusion_score_sde import make_sde, sample_score_model
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

TIME_EMBEDDING_SCALE = 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "score_sde")
    parser.add_argument("--sde", choices=("vp", "ve", "subvp"), default="vp")
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=50.0)
    parser.add_argument("--time-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--eval-sampler",
        choices=("reverse_sde", "probability_flow"),
        default="probability_flow",
    )
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--ode-solver", choices=("euler", "heun"), default="heun")
    parser.add_argument(
        "--final-denoise", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def score_matching_loss(model, sde, clean, time_epsilon):
    with torch.no_grad():
        time = time_epsilon + (1 - time_epsilon) * torch.rand(
            len(clean), device=clean.device
        )
        epsilon = torch.randn_like(clean)
        noisy, _, sigma = sde.marginal_sample(clean, time, epsilon)
    score = model(noisy, time * TIME_EMBEDDING_SCALE)
    # sigma^2 weighting cancels the singular scale of -epsilon/sigma.
    per_image = (sigma * score + epsilon).square().flatten(1).mean(1)
    return per_image, time


def train(args):
    if not 0 < args.time_epsilon < 1:
        raise ValueError("Continuous score training needs 0 < time_epsilon < 1.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_image_loader(args, device)
    config = (
        {"sigma_min": args.sigma_min, "sigma_max": args.sigma_max}
        if args.sde == "ve"
        else {"beta_min": args.beta_min, "beta_max": args.beta_max}
    )
    sde = make_sde(args.sde, **config)
    model = DiffusionUNet(
        image_size=args.image_size, hidden_dims=args.hidden_dims, dropout=args.dropout
    ).to(device)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metadata = {
        "algorithm": "score_sde",
        "sde_name": args.sde,
        "sde_config": config,
        "time_epsilon": args.time_epsilon,
        "time_embedding_scale": TIME_EMBEDDING_SCALE,
        "prediction_type": "score",
        "time_direction": "data_to_noise",
        "time_sampling": "uniform[time_epsilon,1)",
        "loss_weight": "sigma(t)^2",
        "sampling_config": {
            "sampler": args.eval_sampler,
            "steps": args.sampling_steps,
            "ode_solver": args.ode_solver
            if args.eval_sampler == "probability_flow"
            else None,
            "final_denoise": args.final_denoise,
        },
        "training": training_metadata(args),
    }
    start = restore_checkpoint(args.resume_from, model, averaged, optimizer, **metadata)
    prepare_output(args)
    monitor = DiffusionQualityMonitor(args, device) if args.eval_every else None

    def sample_batch(count, generator):
        return sample_score_model(
            averaged,
            sde,
            (count, 3, args.image_size, args.image_size),
            sampler=args.eval_sampler,
            num_steps=args.sampling_steps,
            time_epsilon=args.time_epsilon,
            ode_solver=args.ode_solver,
            time_embedding_scale=TIME_EMBEDDING_SCALE,
            final_denoise=args.final_denoise,
            generator=generator,
        )[0]

    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        gradient_sum = 0.0
        for clean, _ in tqdm(loader, desc=f"{args.sde} score {epoch}/{args.epochs}"):
            clean = clean.to(device, non_blocking=True)
            per_image, time = score_matching_loss(model, sde, clean, args.time_epsilon)
            optimizer.zero_grad(set_to_none=True)
            per_image.mean().backward()
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
            meter.update(per_image, time)
        append_record(
            args.output_dir / "training.jsonl",
            {
                "epoch": epoch,
                "noise_coordinate": "continuous forward time in [0,1]",
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
                sde=args.sde,
                sampler=args.eval_sampler,
                sampling_steps=args.sampling_steps,
                ode_solver=args.ode_solver
                if args.eval_sampler == "probability_flow"
                else None,
                time_epsilon=args.time_epsilon,
                time_embedding_scale=TIME_EMBEDDING_SCALE,
                final_denoise=args.final_denoise,
                grid="uniform decreasing time",
            )


if __name__ == "__main__":
    train(parse_args())
