r"""Freeze the 1.0 DDPM denoiser and compare its ancestral chain with DDIM.

Both use identical EMA weights and initial noise. DDPM uses every adjacent
transition and its checkpoint's variance. DDIM uses a subsequence and eta;
quality, actual NFE, and latency are saved alongside images. Changing eta
does not train a new model. An Improved DDPM extension checkpoint also works;
its learned variance is used only for its ancestral chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

import torch

from dl_utils.diffusion.checkpoints import load_pixel_checkpoint
from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.lesson_utils import (
    OUTPUT_ROOT,
    add_data_arguments,
    add_quality_arguments,
)
from dl_utils.diffusion.quality import DiffusionQualityMonitor
from dl_utils.filesystem.directories import reset_dir
from dl_utils.runtime.devices import try_gpu


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_quality_arguments(parser)
    parser.add_argument(
        "--checkpoint", type=Path, default=OUTPUT_ROOT / "ddpm" / "latest.pth"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "ddpm_ddim_comparison"
    )
    parser.add_argument("--sampler", choices=("ddpm", "ddim", "both"), default="both")
    parser.add_argument("--ddim-steps", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--eta", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.eta <= 1:
        raise ValueError("eta must lie in [0, 1].")
    device = try_gpu()
    model, diffusion, checkpoint, args.image_size = load_pixel_checkpoint(
        args.checkpoint, device
    )
    if checkpoint["algorithm"] not in ("vp_ddpm", "improved_ddpm"):
        raise ValueError("DDIM requires a discrete diffusion checkpoint.")
    diffusion = cast(GaussianDiffusion, diffusion)
    if args.checkpoint.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError(
            "Choose a comparison output directory that does not contain the checkpoint."
        )
    reset_dir(str(args.output_dir))
    monitor = DiffusionQualityMonitor(args, device)
    requested = []
    if args.sampler in ("ddpm", "both"):
        requested.append(("ddpm", diffusion.num_steps))
    if args.sampler in ("ddim", "both"):
        requested.extend(("ddim", steps) for steps in args.ddim_steps)
    for sampler, steps in requested:
        # Separate streams keep x_T fixed even when one sampler consumes
        # additional transition noise. The monitor resets both between methods.
        initial_rng = torch.Generator(device=device).manual_seed(args.eval_seed + 1)

        def sample_batch(
            count, generator, *, sampler=sampler, steps=steps, initial_rng=initial_rng
        ):
            noise = torch.randn(
                count,
                3,
                args.image_size,
                args.image_size,
                device=device,
                generator=initial_rng,
            )
            return diffusion.sample(
                model,
                noise.shape,
                sampler=cast(Literal["ddpm", "ddim"], sampler),
                prediction_type=checkpoint["prediction_type"],
                num_inference_steps=steps,
                eta=args.eta if sampler == "ddim" else 0,
                initial_noise=noise,
                generator=generator,
            )[0]

        monitor.evaluate(
            model,
            sample_batch,
            name=f"{sampler}_{steps}",
            checkpoint=args.checkpoint.name,
            checkpoint_epoch=checkpoint["epoch"],
            model_state="ema",
            sampler=sampler,
            sampling_steps=steps,
            initial_seed=args.eval_seed + 1,
            eta=args.eta if sampler == "ddim" else None,
            variance_type=checkpoint["variance_type"]
            if sampler == "ddpm"
            else "ddim_eta",
            grid="all training indices"
            if sampler == "ddpm"
            else "uniform rounded index subsequence",
        )


if __name__ == "__main__":
    main()
