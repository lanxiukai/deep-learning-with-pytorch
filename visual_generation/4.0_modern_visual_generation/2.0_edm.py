"""Train edm on the common Food-101 split; evaluation has a separate numbered entry.

The target, loss, optimizer update and EMA are visible below. All reported
training evidence must come from an actual run, not this source file.
"""

import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.data import NUM_CLASSES, data_config, make_loader
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
from dl_utils.modern.edm import EDMPreconditioner
from dl_utils.modern.monitoring import monitor_generation
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("edm")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    return parser.parse_args()


def main(args):
    device = setup(args)
    conditional = True
    latent = False
    codec, scale = None, 1.0
    data = data_config(args)
    algorithm = {
        "task": "edm",
        "conditional": conditional,
        "latent": latent,
        "condition_dropout": args.condition_dropout if conditional else 0.0,
        "diffusion": {"num_steps": args.diffusion_steps},
        "precision": args.precision,
        "ema_decay": args.ema_decay,
    }
    channels, size = 3, args.image_size
    network = DiffusionUNet(
        image_size=size,
        in_channels=channels,
        hidden_dims=args.hidden_dims,
        num_classes=NUM_CLASSES if conditional else None,
        class_conditioning="additive",
    ).to(device)
    model, kind = network, "unet"
    model, kind = EDMPreconditioner(network).to(device), "edm"
    algorithm.update(
        sigma_min=0.002,
        sigma_max=80.0,
        log_sigma_mean=-1.2,
        log_sigma_std=1.2,
        condition_dropout=0.0,
    )
    GaussianDiffusion(num_steps=args.diffusion_steps).to(device)
    parameters = list(model.parameters())
    alignment = None
    loader = make_loader(args)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    metadata = {
        "kind": kind,
        "model_config": model.config(),
        "data_config": data,
        "algorithm": algorithm,
    }
    start, step, saved = resume_training(args, model, ema, [optimizer], metadata)
    if alignment is not None and saved:
        alignment.projector.load_state_dict(saved["repa_projector"])
    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        diagnostics = {}
        diagnostic_sums, diagnostic_count = {}, 0
        for images, classes in tqdm(loader, desc=f"Epoch {epoch}"):
            images, classes = images.to(device), classes.to(device)
            with torch.no_grad():
                clean = (
                    codec.encode_latent(images, latent_scale=scale)
                    if latent
                    else images
                )
            labels = classes if conditional else None
            if conditional and algorithm["condition_dropout"]:
                labels = torch.where(
                    torch.rand(len(classes), device=device) < args.condition_dropout,
                    NUM_CLASSES,
                    classes,
                )
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                log_sigma = -1.2 + 1.2 * torch.randn(len(clean), device=device)
                sigma = log_sigma.exp()
                s = sigma[:, None, None, None]
                noisy = clean + s * torch.randn_like(clean)
                prediction = model(noisy, sigma, labels).float()
                weight = (sigma.square() + model.sigma_data**2) / (
                    sigma * model.sigma_data
                ).square()
                per_image = weight * (prediction - clean).square().flatten(1).mean(1)
                coordinate = ((log_sigma + 6.2) / 10.6).clamp(0, 1)
                loss = per_image.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite objective; checkpoint not overwritten."
                )
            loss.backward()
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            step += 1
            meter.update(per_image, coordinate)
            for key, value in diagnostics.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0) + value * len(images)
            diagnostic_count += len(images)
            if args.max_steps is not None and step >= args.max_steps:
                break
        record_epoch(
            args,
            epoch,
            step,
            {
                **meter.result(),
                **{k: v / diagnostic_count for k, v in diagnostic_sums.items()},
            },
        )
        extra = (
            {"repa_projector": alignment.projector.state_dict()}
            if alignment is not None
            else {}
        )
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer],
            epoch,
            step,
            metadata,
            **extra,
        )
        monitor_generation(args, ema, metadata, epoch, codec=codec, latent_scale=scale)
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
