r"""Freeze a CFM velocity and compare Euler, midpoint, and Heun generation.

The checkpoint fixes the probability path and time scale. Each solver starts
from identical standard Gaussian noise and integrates from 0 to 1. Compare
both equal interval counts and equal NFE budgets; second order costs two
network calls per interval. There is no DDPM-style endpoint denoising.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dl_utils.diffusion.checkpoints import load_pixel_checkpoint
from dl_utils.diffusion.flow_matching import sample_flow_model
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
        "--checkpoint", type=Path, default=OUTPUT_ROOT / "flow_matching" / "latest.pth"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "flow_comparison"
    )
    parser.add_argument(
        "--solvers",
        choices=("euler", "midpoint", "heun"),
        nargs="+",
        default=["euler", "midpoint", "heun"],
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50, 100])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, checkpoint, args.image_size = load_pixel_checkpoint(
        args.checkpoint, device
    )
    if checkpoint["algorithm"] != "flow_matching":
        raise ValueError("Use a direct velocity checkpoint from 3.0_flow_matching.py.")
    if args.checkpoint.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError("Comparison output must not contain the source checkpoint.")
    reset_dir(str(args.output_dir))
    monitor = DiffusionQualityMonitor(args, device)
    for solver in args.solvers:
        for steps in args.steps:

            def sample_batch(count, generator, *, solver=solver, steps=steps):
                raw, _ = sample_flow_model(
                    model,
                    (count, 3, args.image_size, args.image_size),
                    solver=solver,
                    num_steps=steps,
                    time_embedding_scale=checkpoint["time_embedding_scale"],
                    generator=generator,
                )
                return raw.clamp(-1, 1)

            monitor.evaluate(
                model,
                sample_batch,
                name=f"{solver}_{steps}",
                checkpoint=args.checkpoint.name,
                checkpoint_epoch=checkpoint["epoch"],
                algorithm="flow_matching",
                model_state="ema",
                sampler=solver,
                sampling_steps=steps,
                path=checkpoint["flow_config"],
                prediction_type=checkpoint["prediction_type"],
                coupling=checkpoint["coupling"],
                time_direction=checkpoint["time_direction"],
                time_range=[0.0, 1.0],
                time_embedding_scale=checkpoint["time_embedding_scale"],
                grid="uniform increasing time",
                final_denoise=False,
                postprocessing="terminal RGB clamp to [-1,1]",
            )


if __name__ == "__main__":
    main()
