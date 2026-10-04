"""Train iMF from scratch on the shared frozen latent space.

FP32 is the initial derivative-validation configuration. Adjustable guidance
is learned as an input; inference does not evaluate the JVP or the auxiliary head.
"""

import copy

import torch
from tqdm import tqdm

from dl_utils.diffusion.checkpoints import load_codec
from dl_utils.diffusion.data import data_config, make_loader
from dl_utils.diffusion.improved_mean_flow import ImprovedMeanFlow, imf_loss
from dl_utils.diffusion.lesson_utils import (
    BinnedLoss,
    autocast,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.diffusion.monitoring import monitor_generation
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("imf")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--no-auxiliary-head", action="store_true")
    parser.add_argument("--conditioning", choices=("tokens", "adaln"), default="tokens")
    parser.add_argument("--guidance-max", type=float, default=5.0)
    parser.add_argument("--fixed-guidance-interval", action="store_true")
    parser.add_argument("--adaptive-power", type=float, default=0.5)
    return parser.parse_args()


def main(args):
    device = setup(args)
    if args.guidance_max < 1 or not 0 <= args.adaptive_power <= 1:
        raise ValueError("Invalid guidance range or adaptive power.")
    data = data_config(args)
    codec, scale, codec_state = load_codec(args.autoencoder_checkpoint, device, data)
    model = ImprovedMeanFlow(
        image_size=codec.latent_size,
        in_channels=codec.latent_channels,
        out_channels=codec.latent_channels,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        auxiliary_head=not args.no_auxiliary_head,
        conditioning=args.conditioning,
    ).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    algorithm = {
        "task": "imf",
        "conditional": True,
        "latent": True,
        "codec_checkpoint": str(args.autoencoder_checkpoint.resolve()),
        "codec_id": codec_state["checkpoint_id"],
        "latent_scale": scale,
        "guidance_max": args.guidance_max,
        "variable_interval": not args.fixed_guidance_interval,
        "adaptive_power": args.adaptive_power,
        "precision": args.precision,
        "ema_decay": args.ema_decay,
    }
    metadata = {
        "kind": "imf",
        "model_config": model.config(),
        "data_config": data,
        "algorithm": algorithm,
    }
    start, step, _ = resume_training(args, model, ema, [optimizer], metadata)
    for epoch in range(start, args.epochs + 1):
        model.train()
        meter, sums, count = BinnedLoss(), {}, 0
        for images, labels in tqdm(make_loader(args), desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                clean = codec.encode_latent(images, latent_scale=scale)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                per_image, coordinate, diagnostics = imf_loss(
                    model,
                    clean,
                    labels,
                    guidance_max=args.guidance_max,
                    variable_interval=not args.fixed_guidance_interval,
                    adaptive_power=args.adaptive_power,
                )
                loss = per_image.float().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite iMF objective.")
            loss.backward()
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            meter.update(per_image, coordinate)
            for key, value in diagnostics.items():
                sums[key] = sums.get(key, 0) + value.item() * len(images)
            count += len(images)
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        record_epoch(
            args,
            epoch,
            step,
            {**meter.result(), **{k: v / count for k, v in sums.items()}},
        )
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer],
            epoch,
            step,
            metadata,
        )
        monitor_generation(args, ema, metadata, epoch, codec=codec, latent_scale=scale)
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
