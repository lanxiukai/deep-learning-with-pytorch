"""Train sr_regression: the shared Food-101 images provide aligned 128-to-256 pairs.

Original SR3 has no label input. CDM adds independently sampled condition-noise
levels and, optionally, labels. All variants require their own training run.
"""

import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.data import data_config, low_resolution, make_loader, upsample
from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.lesson_utils import (
    BinnedLoss,
    autocast,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.modern.monitoring import monitor_generation
from dl_utils.modern.sr3 import SR3, augment_condition
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("sr_regression")
    parser.add_argument(
        "--shuffle-condition",
        action="store_true",
        help="Retrained condition-use ablation.",
    )
    parser.add_argument(
        "--no-class-condition", action="store_true", help="CDM spatial-only ablation."
    )
    parser.add_argument("--augmentation-max", type=float, default=0.3)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    return parser.parse_args()


def main(args):
    device = setup(args)
    task, augmentation = "sr_regression", False
    conditional = augmentation and not args.no_class_condition
    if not 0 <= args.augmentation_max < 1:
        raise ValueError("Condition-noise standard deviation must lie in [0, 1).")
    if task == "sr_regression":
        model = DiffusionUNet(
            image_size=args.image_size, hidden_dims=args.hidden_dims
        ).to(device)
        kind = "unet"
    else:
        model = SR3(
            image_size=args.image_size,
            hidden_dims=args.hidden_dims,
            augmentation=augmentation,
            class_conditional=conditional,
        ).to(device)
        kind = "sr3"
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    diffusion = GaussianDiffusion(num_steps=args.diffusion_steps).to(device)
    algorithm = {
        "task": task,
        "conditional": conditional,
        "latent": False,
        "diffusion": diffusion.config(),
        "scale_factor": 2,
        "shuffle_condition": args.shuffle_condition,
        "augmentation_max": args.augmentation_max if augmentation else 0.0,
        "precision": args.precision,
        "ema_decay": args.ema_decay,
    }
    metadata = {
        "kind": kind,
        "model_config": model.config(),
        "data_config": data_config(args),
        "algorithm": algorithm,
    }
    start, step, _ = resume_training(args, model, ema, [optimizer], metadata)
    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        for images, labels in tqdm(make_loader(args), desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            low = low_resolution(images)
            if args.shuffle_condition:
                # A nontrivial cyclic permutation avoids accidental fixed points.
                if len(low) < 2:
                    continue
                low = low.roll(1, 0)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                if task == "sr_regression":
                    prediction = model(
                        upsample(low, images.shape[-2:]),
                        torch.zeros(len(images), device=device),
                    )
                    per_image = (
                        (prediction.float() - images).square().flatten(1).mean(1)
                    )
                    coordinate = torch.zeros(len(images), device=device)
                else:
                    index = torch.randint(
                        diffusion.num_steps, (len(images),), device=device
                    )
                    # Uniform cumulative POWER inside the chosen interval, not uniform sqrt(power).
                    low_power, high_power = (
                        diffusion.alpha_bars[index],
                        diffusion.alpha_bars_prev[index],
                    )
                    power = low_power + torch.rand_like(low_power) * (
                        high_power - low_power
                    )
                    noise = torch.randn_like(images)
                    noisy = (
                        power.sqrt()[:, None, None, None] * images
                        + (1 - power).sqrt()[:, None, None, None] * noise
                    )
                    level = (
                        torch.rand_like(power) * args.augmentation_max
                        if augmentation
                        else torch.zeros_like(power)
                    )
                    observation = augment_condition(low, level) if augmentation else low
                    prediction = model(noisy, power, observation, labels, level)
                    per_image = (prediction.float() - noise).abs().flatten(1).mean(1)
                    coordinate = index.float() / diffusion.num_steps
                loss = per_image.mean()
            loss.backward()
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            meter.update(per_image, coordinate)
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        record_epoch(args, epoch, step, meter.result())
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer],
            epoch,
            step,
            metadata,
        )
        monitor_generation(args, ema, metadata, epoch)
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
