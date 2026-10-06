"""Train WGAN or WGAN-GP on anime faces.

The generator uses the DCGAN topology. The BatchNorm-free critic is trained
with WGAN-GP by default; set ``LIPSCHITZ`` to ``"clip"`` for original WGAN
weight clipping.

Training data:
Anime faces:      63,565 images
Samples per epoch: 63,488
Note: ``drop_last=True`` discards the final incomplete batch of 77 images.

Default image sizes:
Training input:  64x64 RGB
Generated image: 64x64 RGB

Generator:   3.6 M params
Critic:      2.8 M params
Total:       6.4 M params
"""

import os
import warnings

import torch
import torchvision

from dl_utils.data.images import load_rgb_image
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.wgan import WGANCritic, WGANGenerator, train_wgan
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "anime_face"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "wgan_gp"

NUM_EPOCHS = 120
BATCH_SIZE = 128
MAX_NUM_WORKERS = 8
Z_DIM = 100
GENERATOR_BASE_CHANNELS = 64
CRITIC_BASE_CHANNELS = 64
LEARNING_RATE = 1e-4
N_CRITIC = 5
LAMBDA_GP = 10
LIPSCHITZ = "gp"
SAMPLES_TO_DISPLAY = 18
SAMPLE_GRID_COLUMNS = 6
SAMPLE_EVERY_EPOCHS = 6


def main():
    reset_dir(str(OUT_DIR))
    device = try_gpu()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        DATA_DIR,
        transform=transform,
        loader=load_rgb_image,
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

    warnings.filterwarnings("ignore")
    generator = WGANGenerator(
        z_dim=Z_DIM,
        base_channels=GENERATOR_BASE_CHANNELS,
    )
    critic = WGANCritic(base_channels=CRITIC_BASE_CHANNELS)

    train_wgan(
        generator,
        critic,
        data_loader,
        num_epochs=NUM_EPOCHS,
        z_dim=Z_DIM,
        output_dir=OUT_DIR,
        lipschitz=LIPSCHITZ,
        learning_rate=LEARNING_RATE,
        n_critic=N_CRITIC,
        lambda_gp=LAMBDA_GP,
        samples_to_display=SAMPLES_TO_DISPLAY,
        sample_grid_columns=SAMPLE_GRID_COLUMNS,
        sample_every_epochs=SAMPLE_EVERY_EPOCHS,
        device=device,
    )


if __name__ == "__main__":
    main()
