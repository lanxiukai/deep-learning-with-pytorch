"""Periodic fixed-noise previews and held-out KID; no hidden optimizer updates."""

import copy

import torch

from dl_utils.diffusion.data import make_loader
from dl_utils.diffusion.lesson_utils import append_record, preview
from dl_utils.diffusion.sampling import make_sampler


def monitor_generation(
    args,
    model,
    metadata,
    epoch,
    *,
    codec=None,
    latent_scale=1.0,
    classifier=None,
    sampler_factory=make_sampler,
):
    pictures = args.sample_every and epoch % args.sample_every == 0
    quality = args.eval_every and epoch % args.eval_every == 0
    if not pictures and not quality:
        return
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    sampler = sampler_factory(
        model, metadata, codec=codec, latent_scale=latent_scale, classifier=classifier
    )
    task = metadata["algorithm"]["task"]
    steps = (
        len(metadata["algorithm"]["student_sigmas"])
        if task == "dmd2"
        else (1 if task in ("consistency", "imf") else args.sample_steps)
    )
    # Monitoring must not alter the training noise stream or module mode.
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(20261004)
            if pictures:
                channels = codec.latent_channels if codec else 3
                size = codec.latent_size if codec else args.image_size
                base = torch.randn(4, channels, size, size, device=device)
                labels = torch.arange(0, 101, 10, device=device).repeat_interleave(4)
                noise = base.repeat(11, 1, 1, 1)
                images = []
                low = None
                if task in ("sr3", "sr_regression"):
                    from dl_utils.diffusion.data import low_resolution

                    source, labels = next(
                        iter(make_loader(args, "validation", limit=44))
                    )
                    source = source[:4].repeat_interleave(4, dim=0).to(device)
                    labels = labels[:4].repeat_interleave(4).to(device)
                    low = low_resolution(source)
                    noise = noise[: len(source)]
                    preview(
                        args.output_dir / f"condition_{epoch:04d}.png", source, nrow=4
                    )
                for start in range(0, len(noise), args.batch_size):
                    end = min(start + args.batch_size, len(noise))
                    images.append(
                        sampler(
                            end - start,
                            labels=labels[start:end] if task != "vp" else None,
                            noise=noise[start:end],
                            steps=steps,
                            low=low[start:end] if low is not None else None,
                        )
                    )
                preview(
                    args.output_dir / f"epoch_{epoch:04d}.png",
                    torch.cat(images),
                    nrow=4,
                )
            if quality:
                from dl_utils.diffusion.quality import evaluate_generation

                config = copy.copy(args)
                config.eval_examples = args.eval_examples
                # Validation only; final testing belongs to the separate entry.
                record = evaluate_generation(
                    config,
                    model,
                    metadata,
                    sampler,
                    split="validation",
                    steps=steps,
                    guidance=0 if task == "vp" else 1,
                    full=False,
                )
                append_record(
                    args.output_dir / "validation.jsonl", {"epoch": epoch, **record}
                )
    finally:
        model.train(was_training)
