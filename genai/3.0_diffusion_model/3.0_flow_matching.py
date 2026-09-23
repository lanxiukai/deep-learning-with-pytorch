r"""Conditional Flow Matching: independent endpoints, velocity MSE, then ODE.

Default: 128px CelebA, x_0 ~ N(0,I), x_1 ~ data, t ~ Uniform[0,1).
The linear path x_t=(1-t)x_0+t*x_1 has target x_1-x_0. The network sees only
(x_t,t) and learns a marginal velocity; its trajectories need not be straight.
The same baseline is the starting rectified-flow objective, without reflow.
"""

from __future__ import annotations

import argparse
import copy
import math

import torch
from tqdm import tqdm

from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.flow_matching import GaussianConditionalPath, sample_flow_model
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
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.ema import update_ema

TIME_EMBEDDING_SCALE = 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, "flow_matching")
    parser.add_argument("--path", choices=("linear", "trigonometric"), default="linear")
    parser.add_argument(
        "--ode-solver", choices=("euler", "midpoint", "heun"), default="heun"
    )
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def flow_matching_loss(model, path, data):
    # Independent coupling: do not sort, optimize pairings, or use model outputs.
    noise = torch.randn_like(data)
    time = torch.rand(len(data), device=data.device)
    state, target = path.sample(noise, data, time)  # Analytic and detached.
    velocity = model(state, time * TIME_EMBEDDING_SCALE)
    per_image = (velocity - target).square().flatten(1).mean(1)
    # No ODE simulation, score division, or divergence estimation in this loss.
    return per_image, time, velocity.detach()


@torch.no_grad()
def endpoint_diagnostics(model, averaged, path, data, seed):
    """Scale/EMA probes on the last training minibatch, not validation quality.

    At each epoch the current model and EMA see the same states in eval mode.
    The noise is fixed by seed; the minibatch may change between epochs.
    """
    generator = torch.Generator(device=data.device).manual_seed(seed)
    noise = torch.randn(
        data.shape, device=data.device, dtype=data.dtype, generator=generator
    )
    was_training = model.training
    model.eval()
    result = {}
    try:
        for name, value in (
            ("noise_endpoint", 0.0),
            ("middle", 0.5),
            ("data_endpoint", 1.0),
        ):
            time = torch.full((len(data),), value, device=data.device)
            state, target = path.sample(noise, data, time)
            current = model(state, time * TIME_EMBEDDING_SCALE)
            ema = averaged(state, time * TIME_EMBEDDING_SCALE)
            result[name] = {
                "time": value,
                "state_rms": state.square().mean().sqrt().item(),
                "target_rms": target.square().mean().sqrt().item(),
                "velocity_rms": current.square().mean().sqrt().item(),
                "ema_velocity_rms": ema.square().mean().sqrt().item(),
                "current_mse": (current - target).square().mean().item(),
                "ema_mse": (ema - target).square().mean().item(),
                "current_ema_rms_gap": (current - ema)
                .square()
                .mean()
                .sqrt()
                .item(),
            }
    finally:
        model.train(was_training)
    return result


def train(args):
    if args.max_grad_norm <= 0 or args.sampling_steps < 1:
        raise ValueError("Gradient norm limit and sampling steps must be positive.")
    set_seed(args.seed)
    device = try_gpu()
    loader = make_image_loader(args, device)
    path = GaussianConditionalPath(args.path)
    model = DiffusionUNet(
        image_size=args.image_size, hidden_dims=args.hidden_dims, dropout=args.dropout
    ).to(device)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sampling_config = {
        "solver": args.ode_solver,
        "steps": args.sampling_steps,
        "grid": "uniform increasing time",
        "time_range": [0.0, 1.0],
        "final_denoise": False,
        "postprocessing": "terminal RGB clamp to [-1,1]",
    }
    metadata = {
        "algorithm": "flow_matching",
        "flow_config": path.config(),
        "prediction_type": "velocity",
        "time_direction": "noise_to_data",
        "time_embedding_scale": TIME_EMBEDDING_SCALE,
        "coupling": "independent",
        "base_distribution": "standard Gaussian",
        "time_sampling": "uniform[0,1)",
        "loss_weight": 1.0,
        "loss_reduction": "mean over images and pixels",
        "condition": None,
        "max_grad_norm": args.max_grad_norm,
        "sampling_config": sampling_config,
        "training": training_metadata(args),
    }
    start = restore_checkpoint(args.resume_from, model, averaged, optimizer, **metadata)
    prepare_output(args)
    monitor = DiffusionQualityMonitor(args, device) if args.eval_every else None
    print(
        f"CelebA training_batches={len(loader)}; RGB={args.image_size}; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}; path={args.path}"
    )

    def sample_batch(count, generator):
        raw, _ = sample_flow_model(
            averaged,
            (count, 3, args.image_size, args.image_size),
            num_steps=args.sampling_steps,
            solver=args.ode_solver,
            time_embedding_scale=TIME_EMBEDDING_SCALE,
            generator=generator,
        )
        # The ODE already ends at data time 1. Clip only for RGB evaluation.
        return raw.clamp(-1, 1)

    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss(coordinate_name="time")
        gradient_sum, velocity_squared_sum, examples = 0.0, 0.0, 0
        for data, _ in tqdm(loader, desc=f"CFM {epoch}/{args.epochs}"):
            data = data.to(device, non_blocking=True)
            per_image, time, velocity = flow_matching_loss(model, path, data)
            optimizer.zero_grad(set_to_none=True)
            per_image.mean().backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)
            meter.update(per_image, time)
            gradient_sum += gradient_norm.item()
            velocity_squared_sum += velocity.square().flatten(1).mean(1).sum().item()
            examples += len(data)
        append_record(
            args.output_dir / "training.jsonl",
            {
                "epoch": epoch,
                "time_coordinate": "0=noise, 1=data; bins [0,1/3), [1/3,2/3), [2/3,1]",
                **meter.result(),
                "gradient_norm_before_clip": gradient_sum / len(loader),
                "velocity_rms": math.sqrt(velocity_squared_sum / examples),
                "diagnostic_source": "last training minibatch; eval mode; fixed probe noise",
                "diagnostic_seed": args.eval_seed,
                "diagnostics": endpoint_diagnostics(
                    model, averaged, path, data[: args.eval_batch_size], args.eval_seed
                ),
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
                name=f"{args.ode_solver}_{args.sampling_steps}",
                epoch=epoch,
                model_state="ema",
                algorithm="flow_matching",
                path=path.config(),
                coupling="independent",
                prediction_type="velocity",
                time_direction="noise_to_data",
                time_embedding_scale=TIME_EMBEDDING_SCALE,
                sampler=args.ode_solver,
                sampling_steps=args.sampling_steps,
                final_denoise=False,
                grid="uniform increasing time",
                time_range=[0.0, 1.0],
                postprocessing="terminal RGB clamp to [-1,1]",
            )


if __name__ == "__main__":
    train(parse_args())
