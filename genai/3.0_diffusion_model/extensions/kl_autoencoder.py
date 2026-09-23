"""Train Latent Diffusion's KL-regularized continuous first stage.

After pixel-space DDPM, this lesson revisits the continuous sibling of the
VQGAN tokenizer. It keeps the paired pixel, frozen-feature, and delayed
PatchGAN reconstruction path, but replaces the codebook with a spatial
diagonal Gaussian posterior and a weak KL cost.

The final artifact exposes four distinct operations: posterior parameters,
scaled posterior-mean encoding, scaled posterior-sample encoding, and inverse-
scaled decoding. Its dataset-level scale is the inverse posterior RMS from a
fixed training-data window. A standard-normal decode is saved only as an
interface diagnostic: this autoencoder has no unconditional latent model.
"""

from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.diffusion.image_quality import structural_similarity_index
from dl_utils.diffusion.lesson_utils import (
    OUTPUT_ROOT,
    add_data_arguments,
    make_image_loader,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.gan.sn_gan import (
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.perceptual_autoencoder import (
    KLPerceptualAutoencoder,
    PatchDiscriminator,
    VGGPerceptualLoss,
    adaptive_adversarial_weight,
)
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)


def kl_autoencoder_step(
    model: KLPerceptualAutoencoder,
    discriminator: PatchDiscriminator,
    perceptual: nn.Module,
    reconstruction_logvar: Tensor,
    images: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
    perceptual_weight: float,
    kl_weight: float,
    discriminator_weight: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Update the autoencoder while keeping D frozen but differentiable."""
    reconstruction, mu, logvar, _ = model(images)
    pixel_map = (reconstruction - images).abs()
    perceptual_per_sample = perceptual(reconstruction, images, reduction="none")
    feature_loss = perceptual_per_sample.mean()
    reconstruction_map = (
        pixel_map
        + float(perceptual_weight) * (perceptual_per_sample[:, None, None, None])
    )
    nll = (
        (reconstruction_map * torch.exp(-reconstruction_logvar) + reconstruction_logvar)
        .flatten(1)
        .sum(dim=1)
        .mean()
    )
    kl = diagonal_gaussian_kl_from_logvar(mu, logvar).flatten(1).sum(dim=1).mean()

    discriminator.requires_grad_(False)
    try:
        adversarial = generator_hinge_loss(discriminator(reconstruction))
        if step >= discriminator_start:
            adaptive = adaptive_adversarial_weight(
                nll,
                adversarial,
                model.decoder.last_layer,
                scale=discriminator_weight,
            )
        else:
            adaptive = images.new_zeros(())
        loss = nll + float(kl_weight) * kl + adaptive * adversarial
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    finally:
        discriminator.requires_grad_(True)
    return (
        reconstruction.detach(),
        mu.detach(),
        logvar.detach(),
        {
            "autoencoder": loss.detach(),
            "nll": nll.detach(),
            "pixel_l1": pixel_map.mean().detach(),
            "frozen_feature": feature_loss.detach(),
            "kl": kl.detach(),
            "generator_adversarial": adversarial.detach(),
            "adaptive_weight": adaptive.detach(),
        },
    )


def discriminator_step(
    discriminator: PatchDiscriminator,
    images: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
) -> dict[str, Tensor]:
    """Update D from real images and detached posterior reconstructions."""
    if step < discriminator_start:
        zero = images.new_zeros(())
        return {"discriminator": zero, "real_logit": zero, "fake_logit": zero}
    real_logits = discriminator(images)
    fake_logits = discriminator(reconstruction.detach())
    loss = 0.5 * discriminator_hinge_loss(real_logits, fake_logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "discriminator": loss.detach(),
        "real_logit": real_logits.mean().detach(),
        "fake_logit": fake_logits.mean().detach(),
    }


@torch.inference_mode()
def estimate_latent_interface(
    model: KLPerceptualAutoencoder,
    loader: DataLoader,
    device: torch.device,
    *,
    maximum_batches: int,
    active_variance_threshold: float = 1e-2,
) -> dict[str, object]:
    """Estimate exact posterior moments and one checkpoint-level RMS scale."""
    if maximum_batches < 1:
        raise ValueError("maximum_batches must be positive")
    model.eval()
    channels = model.latent_channels
    mu_total = torch.zeros(channels, device=device, dtype=torch.float64)
    mu_square_total = torch.zeros_like(mu_total)
    variance_total = torch.zeros_like(mu_total)
    kl_total = torch.zeros_like(mu_total)
    values_per_channel = 0
    examples = 0
    batches = 0
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        images = images.to(device, non_blocking=True)
        mu, logvar = model.encode(images)
        mu = mu.double()
        logvar = logvar.double()
        variance = logvar.exp()
        reduce_dimensions = (0, 2, 3)
        mu_total += mu.sum(dim=reduce_dimensions)
        mu_square_total += mu.square().sum(dim=reduce_dimensions)
        variance_total += variance.sum(dim=reduce_dimensions)
        kl_total += (0.5 * (mu.square() + variance - 1.0 - logvar)).sum(
            dim=reduce_dimensions
        )
        values_per_channel += mu.shape[0] * mu.shape[2] * mu.shape[3]
        examples += mu.shape[0]
        batches += 1
    if values_per_channel == 0:
        raise ValueError("cannot estimate latent statistics from an empty loader")
    mean = mu_total / values_per_channel
    mean_variance = variance_total / values_per_channel
    mean_square = mu_square_total / values_per_channel
    variance_of_mu = (mean_square - mean.square()).clamp_min(0.0)
    expected_second_moment = mean_square + mean_variance
    latent_scale = expected_second_moment.mean().rsqrt().item()
    return {
        "examples": examples,
        "batches": batches,
        "posterior_mean_by_channel": mean.cpu().tolist(),
        "posterior_mean_std_by_channel": variance_of_mu.sqrt().cpu().tolist(),
        "posterior_std_by_channel": mean_variance.sqrt().cpu().tolist(),
        "posterior_rms_by_channel": expected_second_moment.sqrt().cpu().tolist(),
        "kl_nats_per_channel_per_image": (kl_total / examples).cpu().tolist(),
        "active_variance_threshold": active_variance_threshold,
        "active_channel_definition": (
            "variance of posterior means over examples and spatial positions"
        ),
        "active_channels": int(
            (variance_of_mu > active_variance_threshold).sum().item()
        ),
        "latent_scale": latent_scale,
        "scaled_expected_second_moment": (
            expected_second_moment.mean() * latent_scale**2
        ).item(),
        "estimator": "inverse_rms_from_exact_E_q[z_squared]",
    }


@torch.inference_mode()
def evaluate(
    model: KLPerceptualAutoencoder,
    discriminator: PatchDiscriminator,
    perceptual: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[dict[str, object], Tensor, Tensor]:
    """Compare posterior-mean and one-sample reconstruction on held-out data."""
    model.eval()
    discriminator.eval()
    perceptual.eval()
    totals = torch.zeros(13, device=device, dtype=torch.float64)
    examples = 0
    comparison = None
    normal_decode = None
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        mu, logvar = model.encode(images)
        mean_reconstruction = model.decoder(mu)
        sample_reconstruction = model.decoder(reparameterize_logvar(mu, logvar))
        mean_mse = F.mse_loss(mean_reconstruction, images)
        sample_mse = F.mse_loss(sample_reconstruction, images)
        values = torch.stack(
            [
                F.l1_loss(mean_reconstruction, images),
                mean_mse,
                structural_similarity_index(mean_reconstruction, images),
                perceptual(mean_reconstruction, images),
                F.l1_loss(sample_reconstruction, images),
                sample_mse,
                structural_similarity_index(sample_reconstruction, images),
                perceptual(sample_reconstruction, images),
                F.mse_loss(sample_reconstruction, mean_reconstruction),
                diagonal_gaussian_kl_from_logvar(mu, logvar)
                .flatten(1)
                .sum(dim=1)
                .mean(),
                discriminator(images).mean(),
                discriminator(mean_reconstruction).mean(),
                discriminator(sample_reconstruction).mean(),
            ]
        ).double()
        totals += values * images.shape[0]
        examples += images.shape[0]
        if comparison is None:
            comparison = torch.cat(
                (images[:8], mean_reconstruction[:8], sample_reconstruction[:8])
            ).cpu()
            normal_decode = model.decoder(torch.randn_like(mu[:8])).cpu()
    if examples == 0 or comparison is None or normal_decode is None:
        raise ValueError("cannot evaluate an empty loader")
    values = (totals / examples).tolist()

    def reconstruction_metrics(offset: int) -> dict[str, float]:
        mse = values[offset + 1]
        return {
            "pixel_l1": values[offset],
            "pixel_mse": mse,
            "psnr_for_minus_one_to_one_range": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
            "ssim_uniform_window": values[offset + 2],
            "frozen_feature_l1": values[offset + 3],
        }

    return (
        {
            "examples": examples,
            "posterior_mean_reconstruction": reconstruction_metrics(0),
            "posterior_sample_reconstruction": reconstruction_metrics(4),
            "sample_mean_reconstruction_mse": values[8],
            "kl_nats_per_image": values[9],
            "patch_discriminator_logits": {
                "real": values[10],
                "posterior_mean_reconstruction": values[11],
                "posterior_sample_reconstruction": values[12],
            },
            "unconditional_generation": None,
            "generation_boundary": (
                "A downstream continuous latent model is not trained in this lesson."
            ),
        },
        comparison,
        normal_decode,
    )


class RandomFeaturePerceptualLoss(nn.Module):
    """Frozen multiscale conv features for offline smoke tests.

    This is deliberately named *random* and must not be reported as LPIPS.
    It lets the algorithmic gradient path be tested without downloading
    pretrained weights. The default KL-autoencoder run uses learned VGG
    features instead.
    """

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(channels, channels * 2, 3, padding=1),
                    nn.ReLU(),
                ),
            ]
        )
        self.eval().requires_grad_(False)

    def forward(
        self, prediction: Tensor, target: Tensor, *, reduction: str = "mean"
    ) -> Tensor:
        per_sample = prediction.new_zeros(prediction.shape[0])
        for block in self.blocks:
            prediction = block(prediction)
            target = block(target)
            per_sample = per_sample + (prediction - target).abs().flatten(1).mean(1)
        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()
        raise ValueError("reduction must be 'none' or 'mean'")


def make_loaders(args, device):
    return (
        make_image_loader(args, device),
        make_image_loader(args, device, augment=False, shuffle=False),
        make_image_loader(args, device, split="validation"),
    )


def train(args: argparse.Namespace) -> None:
    device = try_gpu()
    out_dir = args.output_dir
    reset_dir(str(out_dir))
    train_loader, statistics_loader, validation_loader = make_loaders(args, device)
    config = {
        "latent_channels": args.latent_channels,
        "hidden_channels": args.hidden_channels,
        "downsample_steps": 3,
        "image_size": args.image_size,
    }
    model = KLPerceptualAutoencoder(**config).to(device)
    discriminator = PatchDiscriminator(args.discriminator_channels).to(device)
    perceptual = (
        VGGPerceptualLoss()
        if args.perceptual == "vgg"
        else RandomFeaturePerceptualLoss()
    ).to(device)
    reconstruction_logvar = nn.Parameter(
        torch.tensor(float(args.reconstruction_logvar), device=device),
        requires_grad=not args.fixed_reconstruction_logvar,
    )
    parameters = list(model.parameters())
    if reconstruction_logvar.requires_grad:
        parameters.append(reconstruction_logvar)
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.9)
    )
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        discriminator.train()
        sums = torch.zeros(10, device=device)
        examples = 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            reconstruction, _, _, metrics = kl_autoencoder_step(
                model,
                discriminator,
                perceptual,
                reconstruction_logvar,
                images,
                optimizer,
                step=global_step,
                discriminator_start=args.discriminator_start,
                perceptual_weight=args.perceptual_weight,
                kl_weight=args.kl_weight,
                discriminator_weight=args.discriminator_weight,
            )
            d_metrics = discriminator_step(
                discriminator,
                images,
                reconstruction,
                d_optimizer,
                step=global_step,
                discriminator_start=args.discriminator_start,
            )
            sums += torch.stack([*metrics.values(), *d_metrics.values()]) * images.shape[0]
            examples += images.shape[0]
            global_step += 1
        means = (sums / examples).tolist()
        print(
            f"epoch {epoch:03d}: AE={means[0]:.3f}, NLL={means[1]:.3f}, "
            f"L1={means[2]:.4f}, feature={means[3]:.4f}, KL={means[4]:.3f}, "
            f"G={means[5]:.4f}, lambda={means[6]:.3f}, D={means[7]:.4f}, "
            f"logits(real,fake)=({means[8]:.3f},{means[9]:.3f}), "
            f"s={reconstruction_logvar.item():.3f} "
            f"({'learned' if reconstruction_logvar.requires_grad else 'fixed'})"
        )
        if epoch == 1 or epoch % args.sample_every == 0:
            model.eval()
            with torch.inference_mode():
                deterministic = model.reconstruct(images[:8])
            save_image(
                torch.cat((images[:8], deterministic)).mul(0.5).add(0.5),
                out_dir / f"reconstruction_{epoch:03d}.png",
                nrow=8,
            )

    latent_interface = estimate_latent_interface(
        model,
        statistics_loader,
        device,
        maximum_batches=args.scale_batches,
    )
    validation, comparison, normal_decode = evaluate(
        model,
        discriminator,
        perceptual,
        validation_loader,
        device=device,
    )
    save_image(
        comparison.mul(0.5).add(0.5),
        out_dir / "real_mean_and_sample_reconstruction.png",
        nrow=8,
    )
    save_image(
        normal_decode.mul(0.5).add(0.5),
        out_dir / "standard_normal_decode_not_generation.png",
        nrow=8,
    )
    interface_id = str(uuid.uuid4())
    torch.save(
        {
            "format_version": 3,
            "model_name": "kl_perceptual_autoencoder",
            "roadmap_role": "latent_diffusion_first_stage",
            "direct_baseline": "perceptual autoencoder with Gaussian latent",
            "visible_increment": ("frozen scaled continuous-latent interface for LDM"),
            "interface_id": interface_id,
            "state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "discriminator_config": {
                "base_channels": args.discriminator_channels,
            },
            "model_config": config,
            "reconstruction_logvar": reconstruction_logvar.detach().cpu(),
            "reconstruction_logvar_learned": reconstruction_logvar.requires_grad,
            "reconstruction_reduction": "sum_elements_then_mean_batch",
            "perceptual_network": args.perceptual,
            "perceptual_metric_name": (
                "frozen torchvision VGG16 feature L1"
                if args.perceptual == "vgg"
                else "random frozen feature debug distance"
            ),
            "loss_weights": {
                "perceptual": args.perceptual_weight,
                "kl": args.kl_weight,
                "discriminator": args.discriminator_weight,
            },
            "discriminator_start": args.discriminator_start,
            "global_step": global_step,
            "latent_shape": [
                args.latent_channels,
                model.latent_size,
                model.latent_size,
            ],
            "latent_interface": {
                **latent_interface,
                "statistics_split": "CelebA train, deterministic order",
                "encode_modes": ["posterior_mean", "posterior_sample"],
                "decode_rule": "decoder(scaled_latent / latent_scale)",
            },
            "dataset": "CelebA",
            "image_size": args.image_size,
            "value_range": [-1.0, 1.0],
            "validation_metrics": validation,
            "seed": args.seed,
            "training_budget": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer_updates": global_step,
            },
            "preprocessing": "178px center crop, bilinear resize, random horizontal flip",
            "generation_model": None,
        },
        out_dir / "kl_autoencoder.pth",
    )
    mean_metrics = validation["posterior_mean_reconstruction"]
    assert isinstance(mean_metrics, dict)
    print(
        f"validation mean reconstruction: L1={mean_metrics['pixel_l1']:.4f}, "
        f"PSNR={mean_metrics['psnr_for_minus_one_to_one_range']:.2f}, "
        f"SSIM={mean_metrics['ssim_uniform_window']:.3f}"
    )
    print(
        f"latent scale={latent_interface['latent_scale']:.6f}; "
        "standard-normal decode saved as a diagnostic, not generation"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / "kl_autoencoder"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--perceptual", choices=("vgg", "random"), default="vgg")
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--reconstruction-logvar", type=float, default=0.0)
    parser.add_argument("--fixed-reconstruction-logvar", action="store_true")
    parser.add_argument("--discriminator-weight", type=float, default=1.0)
    parser.add_argument("--discriminator-start", type=int, default=1000)
    parser.add_argument("--scale-batches", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
