"""Train pix2pix for paired CelebA grayscale -> color translation.

Compared with 5.2_cycle_gan.py:
    - Each grayscale/RGB pair comes from the same photograph.
    - One U-Net generator replaces two ResNet generators.
    - D sees (source, target), and G uses GAN + 100 * paired L1 loss.
    - No reverse generator or cycle-consistency loss is needed.

The reusable dataset and complete model definitions live in
dl_utils/gan/pix2pix.py.  The configuration uses a 128x128 U-Net, 70x70
PatchGAN, batch size 8, Adam(lr=2e-4, beta1=0.5), and 20 epochs with linear
learning-rate decay after epoch 10.

Data:
    data/celeba/{black,blond}, also used by 5.2_cycle_gan.py.  Grayscale inputs
    are derived in memory, so no second dataset copy is required.

Outputs:
    output/gan/pix2pix/training/, reset at the start of every run. The directory
    contains titled input/generated/target grids for fixed validation pairs.
    - output/gan/pix2pix/loss_curves.png: total D/G, G GAN, and
      LAMBDA_L1-weighted G L1 losses in one four-panel figure
    - output/gan/pix2pix/pix2pix_generator.pth: trained grayscale -> color generator

Related scripts:
    - 5.1_pix2pix_evaluation.py evaluates the saved colorization generator.

Training data — grayscale-to-color task:
Black-hair images:       48,472
Blond-hair images:       29,980
Available total:         78,452 unique images
Training split:          63,171 paired images
Validation split:         7,199 paired images
Test split:               8,082 paired images
Samples per epoch:       63,171 aligned grayscale/RGB pairs
Note: The grayscale source is derived from the same RGB target in memory, so
each training image contributes one pixel-aligned pair per epoch.

Default image sizes:
Training input:  128x128 grayscale
Generated image: 128x128 RGB

Generator:              41.8 M params
Discriminator:           2.8 M params
Total:                  44.6 M params
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.artifacts import save_loss_curves
from dl_utils.gan.pix2pix import (
    CelebAColorizationDataset,
    ConditionalPatchDiscriminator,
    UNetGenerator,
    build_paired_transform,
    denormalize_images,
    initialize_pix2pix_weights,
)
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
CELEBA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "pix2pix" / "training"
PIX2PIX_OUT_DIR = OUT_DIR.parent
GENERATOR_PATH = PIX2PIX_OUT_DIR / "pix2pix_generator.pth"

NUM_EPOCHS = 20
DECAY_START_EPOCH = 10
BATCH_SIZE = 8
NUM_WORKERS = 8
IMAGE_SIZE = 128
LOAD_SIZE = 143
LEARNING_RATE = 2e-4
LAMBDA_L1 = 100
LOSS_UPDATES_PER_EPOCH = 1
SAMPLES_TO_DISPLAY = 4


def train_epoch(
    generator,
    discriminator,
    loader,
    opt_G,
    opt_D,
    bce,
    l1,
    scaler_G,
    scaler_D,
    device,
    loss_callback=None,
    loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
    progress_bar=None,
):
    """Train one epoch; these two updates are the core pix2pix algorithm.

    The optional callback receives progress, total D, total G, G GAN, and the
    LAMBDA_L1-weighted G L1 loss.
    """
    if loss_updates_per_epoch <= 0:
        raise ValueError("loss_updates_per_epoch must be positive.")

    loop = (
        loader
        if progress_bar is not None
        else tqdm(loader, leave=True, mininterval=1.0)
    )
    loss_sums = torch.zeros(4, device=device)
    # D, G GAN, and LAMBDA_L1-weighted G L1.
    window_loss_sums = [0.0] * 3
    num_examples = 0
    window_examples = 0
    num_batches = len(loader)
    update_interval = max(
        1,
        (num_batches + loss_updates_per_epoch - 1) // loss_updates_per_epoch,
    )

    for batch_idx, (source, target) in enumerate(loop, start=1):
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = source.shape[0]

        # D(source, real target) -> 1; D(source, generated target) -> 0.
        opt_G.zero_grad(set_to_none=True)
        opt_D.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            generated = generator(source)
            real_logits = discriminator(source, target)
            fake_logits = discriminator(source, generated.detach())
            loss_D = 0.5 * (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
            )
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        # G(source) must fool D and reconstruct its aligned target.
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            fake_logits = discriminator(source, generated)
            loss_G_GAN = bce(fake_logits, torch.ones_like(fake_logits))
            loss_G_L1 = l1(generated, target)
            weighted_loss_G_L1 = LAMBDA_L1 * loss_G_L1
            loss_G = loss_G_GAN + weighted_loss_G_L1
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)

        batch_losses = torch.stack(
            [loss_D, loss_G, loss_G_GAN, weighted_loss_G_L1]
        ).detach()
        loss_sums += batch_losses * batch_size
        num_examples += batch_size
        (
            loss_D_value,
            loss_G_value,
            loss_G_GAN_value,
            weighted_loss_G_L1_value,
        ) = batch_losses.tolist()
        if loss_callback is not None:
            loss_values = (
                loss_D_value,
                loss_G_GAN_value,
                weighted_loss_G_L1_value,
            )
            for index, loss_value in enumerate(loss_values):
                window_loss_sums[index] += loss_value * batch_size
            window_examples += batch_size
            if batch_idx % update_interval == 0 or batch_idx == num_batches:
                window_losses = [
                    loss_sum / window_examples for loss_sum in window_loss_sums
                ]
                window_loss_D, *window_generator_components = window_losses
                loss_callback(
                    batch_idx / num_batches,
                    window_loss_D,
                    sum(window_generator_components),
                    *window_generator_components,
                )
                window_loss_sums = [0.0] * 3
                window_examples = 0
        if progress_bar is None:
            loop.set_postfix(
                loss_D=loss_D_value,
                loss_G=loss_G_value,
                refresh=False,
            )
        else:
            progress_bar.set_postfix(
                loss_D=loss_D_value,
                loss_G=loss_G_value,
                refresh=False,
            )
            progress_bar.update(1)

    loss_D, _, loss_G_GAN, weighted_loss_G_L1 = (
        value / num_examples for value in loss_sums.tolist()
    )
    loss_G = loss_G_GAN + weighted_loss_G_L1
    return loss_D, loss_G, loss_G_GAN, weighted_loss_G_L1


def save_training_samples(
    generator,
    source,
    target,
    device,
    output_path,
    epoch,
):
    """Save fixed validation pairs as one titled comparison grid."""
    generator.eval()
    with torch.inference_mode():
        generated = generator(source.to(device)).cpu()
    generator.train()

    image_rows = (
        denormalize_images(source),
        denormalize_images(generated),
        denormalize_images(target),
    )
    row_labels = ("Grayscale input", "Generated color", "Ground truth")
    num_samples = source.shape[0]
    fig, axes = plt.subplots(
        len(image_rows),
        num_samples,
        figsize=(2.4 * num_samples, 6.8),
        squeeze=False,
    )
    for row, (images, label) in enumerate(zip(image_rows, row_labels)):
        for column, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image.permute(1, 2, 0).numpy())
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"Sample {column + 1}")
        axes[row, 0].set_ylabel(label, fontsize=11)
    fig.suptitle(
        f"pix2pix grayscale-to-color validation — epoch {epoch:03d}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    reset_dir(str(OUT_DIR))
    device = try_gpu()

    train_dataset = CelebAColorizationDataset(
        CELEBA_DIR,
        "train",
        build_paired_transform(
            training=True,
            image_size=IMAGE_SIZE,
            load_size=LOAD_SIZE,
        ),
    )
    validation_dataset = CelebAColorizationDataset(
        CELEBA_DIR,
        "validation",
        build_paired_transform(
            training=False,
            image_size=IMAGE_SIZE,
        ),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    sample_source, sample_target = next(
        iter(
            DataLoader(
                validation_dataset,
                batch_size=SAMPLES_TO_DISPLAY,
                shuffle=False,
            )
        )
    )

    generator = UNetGenerator().to(device)
    discriminator = ConditionalPatchDiscriminator().to(device)
    generator.apply(initialize_pix2pix_weights)
    discriminator.apply(initialize_pix2pix_weights)

    opt_G = torch.optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    scaler_G = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scaler_D = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    def lr_factor(epoch):
        if epoch < DECAY_START_EPOCH:
            return 1.0
        decay_epochs = NUM_EPOCHS - DECAY_START_EPOCH
        return max(0.0, 1 - (epoch - DECAY_START_EPOCH + 1) / (decay_epochs + 1))

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_factor)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_factor)
    loss_steps = []
    discriminator_losses = []
    generator_losses = []
    generator_adversarial_losses = {"G GAN loss": []}
    generator_reconstruction_losses = {
        f"{LAMBDA_L1} × G L1 loss": [],
    }

    with tqdm(
        total=NUM_EPOCHS * len(loader),
        desc=f"Epoch 1/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, NUM_EPOCHS + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{NUM_EPOCHS}",
                refresh=False,
            )

            def update_loss_curve(
                progress,
                window_loss_D,
                window_loss_G,
                window_loss_G_GAN,
                window_weighted_loss_G_L1,
                epoch=epoch,
            ):
                loss_steps.append(epoch - 1 + progress)
                discriminator_losses.append(window_loss_D)
                generator_losses.append(window_loss_G)
                generator_adversarial_losses["G GAN loss"].append(window_loss_G_GAN)
                generator_reconstruction_losses[f"{LAMBDA_L1} × G L1 loss"].append(
                    window_weighted_loss_G_L1
                )

            loss_D, loss_G, loss_G_GAN, weighted_loss_G_L1 = train_epoch(
                generator,
                discriminator,
                loader,
                opt_G,
                opt_D,
                bce,
                l1,
                scaler_G,
                scaler_D,
                device,
                loss_callback=update_loss_curve,
                loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
                progress_bar=progress_bar,
            )
            save_training_samples(
                generator,
                sample_source,
                sample_target,
                device,
                OUT_DIR / f"epoch_{epoch:03d}.png",
                epoch,
            )
            progress_bar.write(
                f"epoch {epoch:03d}: D={loss_D:.3f}, G={loss_G:.3f}, "
                f"GAN={loss_G_GAN:.3f}, "
                f"{LAMBDA_L1}×L1={weighted_loss_G_L1:.3f}"
            )
            scheduler_G.step()
            scheduler_D.step()

    torch.save(generator.state_dict(), GENERATOR_PATH)
    save_loss_curves(
        loss_steps,
        discriminator_losses,
        generator_losses,
        generator_adversarial_losses,
        generator_reconstruction_losses,
        PIX2PIX_OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
