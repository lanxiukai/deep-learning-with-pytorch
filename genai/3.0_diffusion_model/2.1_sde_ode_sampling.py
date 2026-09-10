r"""Freeze one continuous score and compare reverse SDE with probability-flow ODE.

All methods integrate the same decreasing time grid from 1 to time_epsilon.
Reverse SDE uses Euler-Maruyama with negative dt and Brownian variance |dt|;
PF-ODE uses Euler or Heun and half the score drift correction. The exact
fields share marginals, not sample paths; learned finite-step outputs differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from dl_utils.diffusion.checkpoints import load_pixel_checkpoint
from dl_utils.diffusion.diffusion_score_sde import VPSDE, sample_score_model
from dl_utils.diffusion.lesson_utils import (
    OUTPUT_ROOT,
    add_data_arguments,
    add_quality_arguments,
)
from dl_utils.diffusion.quality import DiffusionQualityMonitor
from dl_utils.filesystem.directories import reset_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_quality_arguments(parser)
    parser.add_argument(
        "--checkpoint", type=Path, default=OUTPUT_ROOT / "score_sde" / "latest.pth"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "sde_ode_comparison"
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        choices=("reverse_sde", "euler", "heun"),
        default=["reverse_sde", "euler", "heun"],
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 250])
    parser.add_argument(
        "--final-denoise", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sde, checkpoint, args.image_size = load_pixel_checkpoint(
        args.checkpoint, device
    )
    if checkpoint["algorithm"] != "score_sde":
        raise ValueError("Use a continuous score checkpoint from 2.0_score_sde.py.")
    sde = cast(VPSDE, sde)
    if args.checkpoint.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError("Comparison output must not contain the source checkpoint.")
    reset_dir(str(args.output_dir))
    monitor = DiffusionQualityMonitor(args, device)
    for sampler in args.samplers:
        for steps in args.steps:
            # Brownian draws consume a separate stream, keeping x_T identical.
            initial_rng = torch.Generator(device=device).manual_seed(args.eval_seed + 1)

            def sample_batch(
                count,
                generator,
                *,
                sampler=sampler,
                steps=steps,
                initial_rng=initial_rng,
            ):
                noise = torch.randn(
                    count,
                    3,
                    args.image_size,
                    args.image_size,
                    device=device,
                    generator=initial_rng,
                )
                return sample_score_model(
                    model,
                    sde,
                    noise.shape,
                    sampler="reverse_sde"
                    if sampler == "reverse_sde"
                    else "probability_flow",
                    ode_solver=sampler if sampler != "reverse_sde" else "euler",
                    num_steps=steps,
                    time_epsilon=checkpoint["time_epsilon"],
                    time_embedding_scale=checkpoint["time_embedding_scale"],
                    final_denoise=args.final_denoise,
                    initial_noise=noise,
                    generator=generator,
                )[0]

            monitor.evaluate(
                model,
                sample_batch,
                name=f"{sampler}_{steps}",
                checkpoint=args.checkpoint.name,
                checkpoint_epoch=checkpoint["epoch"],
                algorithm="score_sde",
                model_state="ema",
                sde=checkpoint["sde_name"],
                sde_config=checkpoint["sde_config"],
                sampler="reverse_sde"
                if sampler == "reverse_sde"
                else "probability_flow",
                ode_solver=sampler if sampler != "reverse_sde" else None,
                sampling_steps=steps,
                grid="uniform decreasing time",
                time_epsilon=checkpoint["time_epsilon"],
                time_embedding_scale=checkpoint["time_embedding_scale"],
                final_denoise=args.final_denoise,
                prior_std=sde.prior_std,
                initial_seed=args.eval_seed + 1,
            )


if __name__ == "__main__":
    main()
