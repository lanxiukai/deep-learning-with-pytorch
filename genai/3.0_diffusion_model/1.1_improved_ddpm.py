r"""Improved DDPM: learn reverse variance with a separate hybrid objective.

Compare with 1.0 using the same data/backbone. Cosine is the default here;
--beta-schedule linear isolates learned variance from the schedule change.
L = epsilon_MSE + lambda * T * VLB_t uses uniform time sampling, a detached
mean in VLB, and an 8-bit endpoint likelihood. Importance sampling of the pure
VLB is an independent extension, deliberately omitted from this hybrid lesson.
"""

from __future__ import annotations

import argparse
import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.improved_ddpm import ImprovedDDPM
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
    add_training_arguments(parser, "improved_ddpm")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument(
        "--beta-schedule", choices=("linear", "cosine"), default="cosine"
    )
    parser.add_argument("--vlb-weight", type=float, default=0.001)
    parser.add_argument("--eval-sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--ddim-steps", type=int, default=50)
    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_image_loader(args, device)
    model = DiffusionUNet(
        image_size=args.image_size,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        out_channels=6,
    ).to(device)
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = ImprovedDDPM(
        num_steps=args.num_steps, beta_schedule=args.beta_schedule
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metadata = {
        "algorithm": "improved_ddpm",
        "prediction_type": "epsilon",
        "variance_type": "learned_range_unclamped",
        "vlb_weight": args.vlb_weight,
        "endpoint_likelihood": "8bit_discretized_gaussian",
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
        return diffusion.sample(
            averaged,
            (count, 3, args.image_size, args.image_size),
            sampler=args.eval_sampler,
            num_inference_steps=args.ddim_steps
            if args.eval_sampler == "ddim"
            else None,
            generator=generator,
        )[0]

    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = NoiseLossBins()
        sums = torch.zeros(3, device=device)
        examples = 0
        for clean, _ in tqdm(loader, desc=f"Improved DDPM {epoch}/{args.epochs}"):
            clean = clean.to(device, non_blocking=True)
            # The endpoint models 8-bit bins. Quantize AFTER crop/resize.
            clean = ((clean + 1) * 127.5).round() / 127.5 - 1
            time = torch.randint(diffusion.num_steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            hybrid, simple, vb_sum = diffusion.training_losses(
                model, clean, time, noise, vlb_weight=args.vlb_weight
            )
            optimizer.zero_grad(set_to_none=True)
            hybrid.mean().backward()
            optimizer.step()
            update_ema(averaged, model, args.ema_decay)
            meter.update(simple, time / (diffusion.num_steps - 1))
            sums += torch.stack(
                (hybrid.detach().sum(), simple.detach().sum(), vb_sum.detach().sum())
            )
            examples += len(clean)
        append_record(
            args.output_dir / "training.jsonl",
            {
                "epoch": epoch,
                "hybrid": float(sums[0] / examples),
                "epsilon_mse": float(sums[1] / examples),
                "trainable_vlb_sum_bpd_mc": float(sums[2] / examples),
                "vlb_scope": "uniform-time estimate; detached mean; prior KL omitted, not full likelihood",
                "noise_coordinate": "t/(T-1)",
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
                variance_type="learned_range"
                if args.eval_sampler == "ddpm"
                else "ddim_eta",
            )


if __name__ == "__main__":
    train(parse_args())
