"""Held-out reconstruction evidence and training-only posterior calibration."""

import torch


@torch.no_grad()
def latent_statistics(model, loader, device):
    sums = squares = variances = None
    count = 0
    for images, _ in loader:
        mu, logvar = model.encode(images.to(device))
        mu, logvar = mu.double(), logvar.double()
        first, second, variance = (
            mu.sum((0, 2, 3)),
            mu.square().sum((0, 2, 3)),
            logvar.exp().sum((0, 2, 3)),
        )
        sums = first if sums is None else sums + first
        squares = second if squares is None else squares + second
        variances = variance if variances is None else variances + variance
        count += mu.shape[0] * mu.shape[2] * mu.shape[3]
    mean = sums / count
    variance = squares / count - mean.square()
    # RMS includes posterior randomness; no per-image or per-channel renormalization.
    scale = ((squares + variances) / count).mean().rsqrt().item()
    return {
        "latent_scale": scale,
        "posterior_mean": mean.tolist(),
        "posterior_mean_std": variance.clamp_min(0).sqrt().tolist(),
        "posterior_std": (variances / count).sqrt().tolist(),
        "active_channels": (variance > 0.01).sum().item(),
    }


def main():
    import argparse
    import json
    from pathlib import Path

    from dl_utils.diffusion.data import data_config, make_loader
    from dl_utils.diffusion.lesson_utils import DATA_DIR, OUTPUT_ROOT, preview
    from dl_utils.modern.checkpoints import load_codec
    from dl_utils.modern.kl_autoencoder import PerceptualLoss

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=OUTPUT_ROOT / "kl_autoencoder/latest.pth"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--examples", type=int, default=25250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    model, scale, state = load_codec(args.checkpoint, args.device)
    args.image_size = state["data_config"]["image_size"]
    if data_config(args) != state["data_config"]:
        raise ValueError("Different data protocol.")
    output = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    perceptual = PerceptualLoss().to(args.device)
    totals, count = {"l1": 0.0, "psnr": 0.0, "lpips": 0.0, "sample_lpips": 0.0}, 0
    torch.manual_seed(123)
    with torch.no_grad():
        for images, _ in make_loader(args, args.split, limit=args.examples):
            images = images.to(args.device)
            mean = model.reconstruct(images)
            random = model(images)[0]
            totals["l1"] += (mean - images).abs().flatten(1).mean(1).sum().item()
            mse = ((mean - images) / 2).square().flatten(1).mean(1)
            totals["psnr"] += (-10 * mse.clamp_min(1e-12).log10()).sum().item()
            totals["lpips"] += perceptual(mean, images).sum().item()
            totals["sample_lpips"] += perceptual(random, images).sum().item()
            if count == 0:
                preview(
                    output / "reconstructions.png",
                    torch.stack((images, mean, random), 1).flatten(0, 1),
                    nrow=3,
                )
            count += len(images)
    metrics = {k: v / count for k, v in totals.items()}
    metrics.update(
        examples=count,
        split=args.split,
        latent_scale=scale,
        training_statistics=state["latent_statistics"],
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(metrics)
