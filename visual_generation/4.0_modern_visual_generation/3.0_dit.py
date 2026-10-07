"""Train dit on the common Food-101 split; evaluation has a separate numbered entry.

The target, loss, optimizer update and EMA are visible below. All reported
training evidence must come from an actual run, not this source file.
"""

import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.data import NUM_CLASSES, data_config, make_loader
from dl_utils.diffusion.lesson_utils import (
    BinnedLoss,
    autocast,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.modern.checkpoints import load_codec
from dl_utils.modern.learned_variance import LearnedVarianceDiffusion
from dl_utils.modern.monitoring import monitor_generation
from dl_utils.modern.transformer import DiffusionTransformer
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("dit")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument(
        "--unconditional",
        action="store_true",
        help="Train a separate no-label baseline.",
    )
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--qk-norm", action="store_true")
    return parser.parse_args()


def main(args):
    device = setup(args)
    conditional = True
    latent = True
    codec, scale = None, 1.0
    data = data_config(args)
    algorithm = {
        "task": "dit",
        "conditional": conditional,
        "latent": latent,
        "condition_dropout": args.condition_dropout if conditional else 0.0,
        "diffusion": {"num_steps": args.diffusion_steps},
        "precision": args.precision,
        "ema_decay": args.ema_decay,
    }
    channels, size = 3, args.image_size
    conditional = not args.unconditional
    algorithm["conditional"] = conditional
    codec, scale, codec_state = load_codec(args.autoencoder_checkpoint, device, data)
    algorithm.update(
        codec_checkpoint=str(args.autoencoder_checkpoint.resolve()),
        codec_id=codec_state["checkpoint_id"],
        latent_scale=scale,
    )
    channels, size = codec.latent_channels, codec.latent_size
    model = DiffusionTransformer(
        image_size=size,
        in_channels=channels,
        out_channels=channels * 2,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        patch_size=args.patch_size,
        num_classes=NUM_CLASSES,
        rope=args.rope,
        qk_norm=args.qk_norm,
    ).to(device)
    for block in model.blocks:
        block.attention.collect_diagnostics = True
    kind = "dit"
    diffusion = LearnedVarianceDiffusion(num_steps=args.diffusion_steps).to(device)
    parameters = list(model.parameters())
    alignment = None
    loader = make_loader(args)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    for block in ema.blocks:
        block.attention.collect_diagnostics = False
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
                time = torch.randint(diffusion.num_steps, (len(clean),), device=device)
                per_image, simple, vb = diffusion.training_losses(
                    model, clean, time, torch.randn_like(clean), labels
                )
                diagnostics = {
                    "epsilon_mse": simple.mean().item(),
                    "vb_sum_nats": vb.mean().item(),
                    "variance_fraction_min": diffusion.last_variance_fraction[0],
                    "variance_fraction_max": diffusion.last_variance_fraction[1],
                    "attention_max_logit": max(
                        b.attention.last_max_logit for b in model.blocks
                    ),
                    "attention_entropy": sum(
                        b.attention.last_entropy for b in model.blocks
                    )
                    / len(model.blocks),
                }
                coordinate = time.float() / diffusion.num_steps
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
