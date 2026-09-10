r"""Compare frozen diffusion solvers: Euler, Heun, DPM-Solver and DPM-Solver++.

One checkpoint, fixed initial noises, fixed real reference, and several step
budgets. Discrete/continuous VP ODE solvers use a common lambda grid; EDM uses
its sigma grid. Reverse SDE/PC and VE/sub-VP PF-ODE use forward-time grids.
No solver here trains or changes a network. NFE includes endpoint denoising.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

import torch

from dl_utils.diffusion.checkpoints import load_pixel_checkpoint
from dl_utils.diffusion.diffusion_score_sde import sample_score_model
from dl_utils.diffusion.edm import EDMPreconditioner, sample_edm
from dl_utils.diffusion.lesson_utils import (
    OUTPUT_ROOT,
    add_data_arguments,
    add_quality_arguments,
)
from dl_utils.diffusion.quality import DiffusionQualityMonitor
from dl_utils.diffusion.solvers import (
    ContinuousVPPath,
    DiscreteVPPath,
    sample_diffusion_ode,
)
from dl_utils.filesystem.directories import reset_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_quality_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "solver_comparison"
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        help="Default: all ODE solvers supported by this checkpoint.",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[25, 50])
    parser.add_argument("--clip-x0", action="store_true")
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--corrector-steps", type=int, default=1)
    parser.add_argument("--langevin-step-size", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, process, checkpoint, args.image_size = load_pixel_checkpoint(
        args.checkpoint, device
    )
    kind = checkpoint["algorithm"]
    path = None
    if kind in ("vp_ddpm", "improved_ddpm"):
        path = DiscreteVPPath(process)
    elif kind == "score_sde" and checkpoint["sde_name"] == "vp":
        path = ContinuousVPPath(
            process,
            device=device,
            time_epsilon=checkpoint["time_epsilon"],
            embedding_scale=checkpoint["time_embedding_scale"],
        )
    allowed = ["euler", "heun"]
    if path is not None:
        allowed += ["dpm_1", "dpm_2m", "dpmpp_2m"]
    default_solvers = list(allowed)
    if kind == "score_sde":
        allowed += ["reverse_sde", "pc"]
        if checkpoint["sde_name"] == "ve":
            allowed += ["annealed_langevin"]
    solvers = args.solvers or default_solvers
    if any(solver not in allowed for solver in solvers):
        raise ValueError(f"This checkpoint supports: {', '.join(allowed)}")
    if args.checkpoint.resolve().is_relative_to(args.output_dir.resolve()):
        raise ValueError("Comparison output must not contain the source checkpoint.")
    reset_dir(str(args.output_dir))
    monitor = DiffusionQualityMonitor(args, device)
    for solver in solvers:
        for steps in args.steps:
            initial_rng = torch.Generator(device=device).manual_seed(args.eval_seed + 1)
            use_vp_ode = path is not None and solver not in ("reverse_sde", "pc")

            def sample_batch(
                count,
                generator,
                *,
                solver=solver,
                steps=steps,
                initial_rng=initial_rng,
                use_vp_ode=use_vp_ode,
            ):
                noise = torch.randn(
                    count,
                    3,
                    args.image_size,
                    args.image_size,
                    generator=initial_rng,
                    device=device,
                )
                if use_vp_ode:
                    return sample_diffusion_ode(
                        model,
                        path,
                        noise,
                        solver=solver,
                        num_steps=steps,
                        prediction_type=checkpoint.get("prediction_type", "score"),
                        clip_x0=args.clip_x0,
                    )
                if kind == "edm":
                    return sample_edm(
                        cast(EDMPreconditioner, model),
                        noise.shape,
                        num_steps=steps,
                        solver=cast(Literal["euler", "heun"], solver),
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        rho=args.rho,
                        initial_noise=noise,
                    )[0]
                return sample_score_model(
                    model,
                    process,
                    noise.shape,
                    num_steps=steps,
                    sampler="probability_flow"
                    if solver in ("euler", "heun")
                    else solver,
                    ode_solver=solver if solver in ("euler", "heun") else "euler",
                    time_epsilon=checkpoint["time_epsilon"],
                    time_embedding_scale=checkpoint["time_embedding_scale"],
                    corrector_steps=args.corrector_steps,
                    langevin_step_size=args.langevin_step_size,
                    initial_noise=noise,
                    generator=generator,
                )[0]

            monitor.evaluate(
                model,
                sample_batch,
                name=f"{solver}_{steps}",
                checkpoint=args.checkpoint.name,
                checkpoint_epoch=checkpoint["epoch"],
                algorithm=kind,
                model_state="ema",
                sampler=solver,
                sampling_steps=steps,
                initial_seed=args.eval_seed + 1,
                clip_x0=args.clip_x0,
                grid="uniform lambda"
                if use_vp_ode
                else "rho sigma"
                if kind == "edm"
                else "uniform time",
                sigma_min=args.sigma_min if kind == "edm" else None,
                sigma_max=args.sigma_max if kind == "edm" else None,
                rho=args.rho if kind == "edm" else None,
                time_epsilon=checkpoint.get("time_epsilon"),
                corrector_steps=args.corrector_steps,
                langevin_step_size=args.langevin_step_size,
            )


if __name__ == "__main__":
    main()
