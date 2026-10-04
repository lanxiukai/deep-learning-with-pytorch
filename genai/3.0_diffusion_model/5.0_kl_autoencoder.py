"""Train the shared f=8 continuous codec with L1, LPIPS, KL and delayed PatchGAN.

KL is summed over latent coordinates and reconstruction is averaged over RGB.
The fixed adversarial weight is a declared teaching recipe, not an ELBO.
"""

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.diffusion.autoencoder_evaluation import latent_statistics
from dl_utils.diffusion.data import data_config, make_loader
from dl_utils.diffusion.kl_autoencoder import (
    KLPerceptualAutoencoder,
    PatchDiscriminator,
    PerceptualLoss,
)
from dl_utils.diffusion.lesson_utils import (
    autocast,
    preview,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("kl_autoencoder")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--discriminator-start", type=int, default=10000)
    parser.add_argument("--calibration-examples", type=int, default=2020)
    return parser.parse_args()


def main(args):
    device = setup(args)
    model = KLPerceptualAutoencoder(
        hidden_channels=args.width, image_size=args.image_size
    ).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    discriminator = PatchDiscriminator().to(device)
    perceptual = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9)
    )
    algorithm = {
        "task": "autoencoder",
        "kl_weight": args.kl_weight,
        "perceptual_weight": args.perceptual_weight,
        "adversarial_weight": args.adversarial_weight,
        "discriminator_start": args.discriminator_start,
        "precision": args.precision,
        "ema_decay": args.ema_decay,
    }
    metadata = {
        "kind": "autoencoder",
        "model_config": model.config(),
        "data_config": data_config(args),
        "algorithm": algorithm,
    }
    start, step, saved = resume_training(
        args, model, ema, [optimizer, optimizer_d], metadata
    )
    if saved:
        discriminator.load_state_dict(saved["discriminator"])
    loader = make_loader(args)
    for epoch in range(start, args.epochs + 1):
        model.train()
        totals, count = {}, 0
        for images, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            discriminator.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                reconstruction, mu, logvar, _ = model(images)
                l1 = (reconstruction.float() - images).abs().mean()
                lpips = perceptual(reconstruction, images).mean()
                kl = (
                    0.5
                    * (mu.float().square() + logvar.float().exp() - 1 - logvar.float())
                    .flatten(1)
                    .sum(1)
                    .mean()
                )
                adversarial = (
                    -discriminator(reconstruction).float().mean()
                    if step >= args.discriminator_start
                    else images.new_zeros(())
                )
                loss = (
                    l1
                    + args.perceptual_weight * lpips
                    + args.kl_weight * kl
                    + args.adversarial_weight * adversarial
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite codec loss.")
            loss.backward()
            optimizer.step()
            discriminator.requires_grad_(True)
            optimizer_d.zero_grad(set_to_none=True)
            d_loss = images.new_zeros(())
            if step >= args.discriminator_start:
                with autocast(args):
                    real_logit = discriminator(images)
                    fake_logit = discriminator(reconstruction.detach())
                    d_loss = (
                        F.relu(1 - real_logit.float()).mean()
                        + F.relu(1 + fake_logit.float()).mean()
                    )
                d_loss.backward()
                optimizer_d.step()
            update_ema(ema, model, args.ema_decay)
            for name, value in {
                "l1": l1,
                "lpips": lpips,
                "kl_nats": kl,
                "generator_adversarial": adversarial,
                "discriminator": d_loss,
            }.items():
                totals[name] = totals.get(name, 0) + value.item() * len(images)
            count += len(images)
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        # Calibrate on unaugmented TRAIN images, using the exact EMA codec saved below.
        calibration = make_loader(
            args, "train", augment=False, shuffle=False, limit=args.calibration_examples
        )
        statistics = latent_statistics(ema, calibration, device)
        record_epoch(
            args,
            epoch,
            step,
            {**{k: v / count for k, v in totals.items()}, **statistics},
        )
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer, optimizer_d],
            epoch,
            step,
            metadata,
            discriminator=discriminator.state_dict(),
            latent_scale=statistics["latent_scale"],
            latent_statistics=statistics,
        )
        if args.sample_every and epoch % args.sample_every == 0:
            images, _ = next(
                iter(make_loader(args, "validation", limit=args.batch_size))
            )
            images = images.to(device)
            with torch.no_grad():
                means = ema.reconstruct(images)
                random = ema(images)[0]
            preview(
                args.output_dir / f"reconstruction_{epoch:04d}.png",
                torch.stack((images, means, random), 1).flatten(0, 1),
                nrow=3,
            )
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
