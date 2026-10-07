"""Train a conditional WGAN-GP to generate faces with or without glasses.

Adapted from "Generative Deep Learning", 2nd ed.
Data: data/glasses-256, prepared by
tool_scripts/download_dataset.py --dataset glasses.
Outputs: output/gan/cgan/.
Checkpoint evaluation: 4.1_conditional_gan_evaluation.py.

Training data:
With glasses (G):       2,543 images
Without glasses (NoG):  1,957 images
Available total:         4,500 images
Samples per epoch:       4,480
Note: Counts include 517 repository-tracked label corrections. ``drop_last``
omits 20 shuffled images per epoch.

Default image sizes:
Training input:  256x256 RGB
Generated image: 256x256 RGB

Generator:  12.9 M params
Critic:      2.8 M params
Total:      15.7 M params
"""

import os

import torch

from dl_utils.data.vision import image_folder_dataset
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.conditional_gan import (
    ConditionalCritic,
    ConditionalGenerator,
    train_conditional_gan,
)
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
CACHED_DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "cgan"

NUM_EPOCHS = 120
BATCH_SIZE = 32
MAX_NUM_WORKERS = 8
Z_DIM = 100
NUM_CLASSES = 2
IMAGE_CHANNELS = 3
FEATURES = 16
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 5e-5
N_CRITIC = 5
LAMBDA_GP = 10
SAMPLE_GRID_ROWS = 4
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY_EPOCHS = 6


def main():
    reset_dir(str(OUT_DIR))
    device = try_gpu()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if not CACHED_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Resized dataset cache not found: {CACHED_DATA_DIR}\n"
            "Create it with:\n"
            "python tool_scripts/download_dataset.py --dataset glasses"
        )
    dataset = image_folder_dataset(
        CACHED_DATA_DIR,
        normalize=(0.5, 0.5),
    )
    num_workers = min(MAX_NUM_WORKERS, os.cpu_count() or 1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )

    generator = ConditionalGenerator(
        Z_DIM + NUM_CLASSES,
        IMAGE_CHANNELS,
        FEATURES,
    )
    critic = ConditionalCritic(
        IMAGE_CHANNELS + NUM_CLASSES,
        FEATURES,
    )
    train_conditional_gan(
        generator,
        critic,
        data_loader,
        num_epochs=NUM_EPOCHS,
        z_dim=Z_DIM,
        num_classes=NUM_CLASSES,
        output_dir=OUT_DIR,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
        generator_lr=LEARNING_RATE_G,
        critic_lr=LEARNING_RATE_D,
        sample_grid_rows=SAMPLE_GRID_ROWS,
        sample_grid_columns=SAMPLE_GRID_COLUMNS,
        sample_every_epochs=SAMPLE_EVERY_EPOCHS,
        device=device,
    )


if __name__ == "__main__":
    main()
