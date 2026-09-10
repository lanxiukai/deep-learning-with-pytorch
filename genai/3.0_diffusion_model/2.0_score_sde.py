r"""Score-SDE: continuous VP/VE/sub-VP denoising score matching at 128px.

Learn s_theta with sigma^2 ||s_theta + epsilon/sigma||^2. Freeze that score
and compare reverse SDE, Langevin predictor-corrector, and probability-flow
ODE in 2.2. --sde ve --noise-levels 64 instead trains discrete noise levels,
providing the NCSN increment for annealed Langevin sampling.
"""

from __future__ import annotations

import argparse
import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.diffusion_score_sde import make_sde, sample_score_model
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
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
    parser.add_argument("--noise-levels", type=int, default=0)
    parser.add_argument(
        "--eval-sampler",
        choices=("reverse_sde", "pc", "probability_flow", "annealed_langevin"),
        default="probability_flow",
    )
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--ode-solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--corrector-steps", type=int, default=1)
    parser.add_argument("--langevin-step-size", type=float, default=0.01)
    return parser.parse_args()


def score_matching_loss(model, sde, clean, time_epsilon, noise_levels=0):
    time = torch.rand(len(clean), device=clean.device)
    if noise_levels:
        time = torch.randint(noise_levels, (len(clean),), device=clean.device) / (
            noise_levels - 1
        )
    time = time_epsilon + (1 - time_epsilon) * time
    epsilon = torch.randn_like(clean)
    noisy, _, sigma = sde.marginal_sample(clean, time, epsilon)
    score = model(noisy, time * TIME_EMBEDDING_SCALE)
    per_image = (sigma * score + epsilon).square().flatten(1).mean(1)
    return per_image, time


def train(args):
    if args.noise_levels and (args.noise_levels < 2 or args.sde != "ve"):
        raise ValueError("Discrete NCSN training requires VE and >=2 noise levels.")
    if args.eval_sampler == "annealed_langevin" and args.sde != "ve":
        raise ValueError("Annealed Langevin requires VE.")
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
        "noise_levels": args.noise_levels,
        "training": training_metadata(args),
    }
    start = restore_checkpoint(args.resume_from, model, averaged, optimizer, **metadata)
    prepare_output(args)
    monitor = DiffusionQualityMonitor(args, device) if args.eval_every else None
    sampling_steps = (
        args.noise_levels
        if args.eval_sampler == "annealed_langevin" and args.noise_levels
        else args.sampling_steps
    )

    def sample_batch(count, generator):
        return sample_score_model(
            averaged,
            sde,
            (count, 3, args.image_size, args.image_size),
            sampler=args.eval_sampler,
            num_steps=sampling_steps,
            time_epsilon=args.time_epsilon,
            ode_solver=args.ode_solver,
            corrector_steps=args.corrector_steps,
            langevin_step_size=args.langevin_step_size,
            generator=generator,
        )[0]

    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = NoiseLossBins()
        for clean, _ in tqdm(loader, desc=f"{args.sde} score {epoch}/{args.epochs}"):
            clean = clean.to(device, non_blocking=True)
            per_image, time = score_matching_loss(
                model, sde, clean, args.time_epsilon, args.noise_levels
            )
            optimizer.zero_grad(set_to_none=True)
            per_image.mean().backward()
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)
            meter.update(per_image, time)
        append_record(
            args.output_dir / "training.jsonl",
            {"epoch": epoch, "noise_coordinate": "continuous time", **meter.result()},
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
                sampling_steps=sampling_steps,
                ode_solver=args.ode_solver,
                time_epsilon=args.time_epsilon,
                corrector_steps=args.corrector_steps,
                langevin_step_size=args.langevin_step_size,
            )


if __name__ == "__main__":
    train(parse_args())
