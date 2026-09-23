"""Train a CycleGAN for unpaired black-hair <-> blond-hair translation.

Adapted from the companion code for "Generative Deep Learning", 2nd ed.
(book_repos/DGAI/ch06CycleGAN.ipynb). Models and shared training helpers live
in dl_utils/gan/cyclegan.py.

Data:
    data/celeba/{black,blond}, prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Outputs:
    output/gan/cyclegan/training/, reset at the start of every run. The directory
    contains titled 2x2 training grids (real black hair -> generated blond on
    top and real blond hair -> generated black on the bottom).
    - output/gan/cyclegan/loss_curves.png: total D/G, directional adversarial, and
      10 × directional cycle-consistency losses in one four-panel figure
    - output/gan/cyclegan/gen_black.pth: blond -> black generator
    - output/gan/cyclegan/gen_blond.pth: black -> blond generator

Related scripts:
    - 5.3_cycle_gan_evaluation.py evaluates the saved hair-color generators.

Training data — hair-color task:
Black hair:              48,472 images
Blond hair:              29,980 images
Available total:         78,452 unique images
Samples per epoch:       48,472 unpaired image pairs
Note: The unpaired dataset uses the larger domain as its length, so the smaller
blond-hair domain cycles and 18,492 of its images are reused once per epoch.

Default image sizes:
Training input:  128x128 RGB
Generated image: 128x128 RGB

Generator (each):      11.4 M params
Discriminator (each):   2.8 M params
Total (2 of each):     28.3 M params
"""

import random

import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.artifacts import save_loss_curves
from dl_utils.gan.cyclegan import (
    LAMBDA_CYCLE,
    CycleGANDiscriminator,
    CycleGANGenerator,
    UnpairedImageDataset,
    initialize_cyclegan_weights,
    train_cyclegan_epoch,
)
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu

# installation
# !pip install pandas albumentations


# Data lives under <project_root>/data; generated images and checkpoints
# are written under <project_root>/output/gan/cyclegan/training
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "cyclegan" / "training"
CYCLEGAN_OUT_DIR = OUT_DIR.parent
CELEBA_DIR = DATA_DIR / "celeba"
BLACK_DIR = CELEBA_DIR / "black"
BLOND_DIR = CELEBA_DIR / "blond"
GEN_BLACK_PATH = CYCLEGAN_OUT_DIR / "gen_black.pth"
GEN_BLOND_PATH = CYCLEGAN_OUT_DIR / "gen_blond.pth"

NUM_EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
IMAGE_SIZE = 128
NUM_RESIDUAL_BLOCKS = 9
LOSS_UPDATES_PER_EPOCH = 20
SNAPSHOT_UPDATES_PER_EPOCH = 10


def save_dataset_samples(image_dirs, output_path, samples_per_dir=8, seed=42):
    """Save a reproducible sample grid from each image directory."""
    rng = random.Random(seed)
    sampled_paths = []
    for image_dir in image_dirs:
        image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
        if len(image_paths) < samples_per_dir:
            raise ValueError(
                f"{image_dir} contains {len(image_paths)} images; "
                f"at least {samples_per_dir} are required."
            )
        sampled_paths.append(rng.sample(image_paths, samples_per_dir))

    fig, axes = plt.subplots(
        len(image_dirs),
        samples_per_dir,
        dpi=100,
        figsize=(1.78 * samples_per_dir, 2.18 * len(image_dirs)),
        squeeze=False,
    )
    for row, paths in enumerate(sampled_paths):
        for column, image_path in enumerate(paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.subplots_adjust(wspace=-0.01, hspace=-0.1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Train the hair-color CycleGAN and save its artifacts."""
    reset_dir(str(OUT_DIR))

    if not BLACK_DIR.is_dir() or not BLOND_DIR.is_dir():
        raise FileNotFoundError(
            "CycleGAN requires the local CelebA splits at data/celeba/black "
            "and data/celeba/blond. Run: "
            "python tool_scripts/download_dataset.py --dataset celeba"
        )

    save_dataset_samples(
        [BLACK_DIR, BLOND_DIR],
        OUT_DIR / "celeba_hair_samples.png",
    )

    transforms = albumentations.Compose(
        [
            albumentations.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=255,
            ),
            ToTensorV2(),  # numpy H×W×C → torch C×H×W
        ],
        additional_targets={"image0": "image"},
    )  # Apply the same transforms to image0.
    dataset = UnpairedImageDataset(
        domain_a_roots=[str(BLACK_DIR)],
        domain_b_roots=[str(BLOND_DIR)],
        transform=transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    device = try_gpu()
    discriminator_black = CycleGANDiscriminator().to(device)
    discriminator_blond = CycleGANDiscriminator().to(device)
    discriminator_black.apply(initialize_cyclegan_weights)
    discriminator_blond.apply(initialize_cyclegan_weights)

    generator_black = CycleGANGenerator(
        image_channels=3,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
    ).to(device)
    generator_blond = CycleGANGenerator(
        image_channels=3,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
    ).to(device)
    generator_black.apply(initialize_cyclegan_weights)
    generator_blond.apply(initialize_cyclegan_weights)

    reconstruction_loss = nn.L1Loss()
    adversarial_loss = nn.MSELoss()
    # Mixed-precision (AMP) gradient scalers prevent FP16 gradient underflow.
    # One scaler is used per optimizer because their update patterns differ.
    generator_scaler = torch.amp.GradScaler(device.type)
    discriminator_scaler = torch.amp.GradScaler(device.type)

    discriminator_optimizer = torch.optim.Adam(
        list(discriminator_black.parameters()) + list(discriminator_blond.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    generator_optimizer = torch.optim.Adam(
        list(generator_black.parameters()) + list(generator_blond.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    loss_steps = []
    discriminator_losses = []
    generator_losses = []
    generator_adversarial_losses = {
        "Blond → Black": [],
        "Black → Blond": [],
    }
    generator_reconstruction_losses = {
        f"Black → Blond → Black ({LAMBDA_CYCLE} × cycle)": [],
        f"Blond → Black → Blond ({LAMBDA_CYCLE} × cycle)": [],
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
                window_discriminator_loss,
                window_generator_loss,
                window_generator_black_loss,
                window_generator_blond_loss,
                window_black_cycle_loss,
                window_blond_cycle_loss,
                epoch=epoch,
            ):
                loss_steps.append(epoch - 1 + progress)
                discriminator_losses.append(window_discriminator_loss)
                generator_losses.append(window_generator_loss)
                generator_adversarial_losses["Blond → Black"].append(
                    window_generator_black_loss
                )
                generator_adversarial_losses["Black → Blond"].append(
                    window_generator_blond_loss
                )
                generator_reconstruction_losses[
                    f"Black → Blond → Black ({LAMBDA_CYCLE} × cycle)"
                ].append(window_black_cycle_loss)
                generator_reconstruction_losses[
                    f"Blond → Black → Blond ({LAMBDA_CYCLE} × cycle)"
                ].append(window_blond_cycle_loss)

            discriminator_loss, generator_loss = train_cyclegan_epoch(
                discriminator_black,
                discriminator_blond,
                generator_black,
                generator_blond,
                loader,
                discriminator_optimizer,
                generator_optimizer,
                reconstruction_loss,
                adversarial_loss,
                discriminator_scaler,
                generator_scaler,
                device,
                OUT_DIR,
                epoch=epoch,
                loss_callback=update_loss_curve,
                loss_updates_per_epoch=LOSS_UPDATES_PER_EPOCH,
                snapshot_updates_per_epoch=SNAPSHOT_UPDATES_PER_EPOCH,
                domain_a_label="Black hair",
                domain_b_label="Blond hair",
                progress_bar=progress_bar,
            )

    print(f"loss_D {discriminator_loss:.3f}, loss_G {generator_loss:.3f} on {device}")

    torch.save(generator_black.state_dict(), GEN_BLACK_PATH)
    torch.save(generator_blond.state_dict(), GEN_BLOND_PATH)
    save_loss_curves(
        loss_steps,
        discriminator_losses,
        generator_losses,
        generator_adversarial_losses,
        generator_reconstruction_losses,
        CYCLEGAN_OUT_DIR / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
