"""Train a DCGAN on Pokémon images.

Training data:
Pokémon:          40,597 images across 721 class folders
Samples per epoch: 40,597
Note: Class labels are ignored; folders only provide the ImageFolder layout.

Default image sizes:
Training input:  64x64 RGB
Generated image: 64x64 RGB

Generator:       3.6 M params
Discriminator:   2.8 M params
Total:           6.4 M params
"""

import torch
import torchvision

from dl_utils.data.images import load_rgb_image
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.dcgan import (
    DCGANDiscriminator,
    DCGANGenerator,
    save_leaky_relu_curves,
    train_dcgan,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "pokemon"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "dcgan" / "pokemon"

NUM_EPOCHS = 20
BATCH_SIZE = 128
NUM_WORKERS = 8
Z_DIM = 100
GENERATOR_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 64
LEARNING_RATE_D = 1e-4
LEARNING_RATE_G = 2e-4
SAMPLES_TO_DISPLAY = 18
SAMPLE_GRID_COLUMNS = 6
SAMPLE_EVERY_EPOCHS = 2


def main():
    reset_dir(str(OUT_DIR))

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
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    save_leaky_relu_curves(OUT_DIR / "leaky_relu_curves.png")

    generator = DCGANGenerator(
        z_dim=Z_DIM,
        base_channels=GENERATOR_BASE_CHANNELS,
    )
    discriminator = DCGANDiscriminator(
        base_channels=DISCRIMINATOR_BASE_CHANNELS,
    )
    example_noise = torch.zeros((1, Z_DIM, 1, 1))
    example_images = torch.zeros((1, 3, 64, 64))
    print(generator(example_noise).shape)  # torch.Size([1, 3, 64, 64])
    print(discriminator(example_images).shape)  # torch.Size([1, 1, 1, 1])

    train_dcgan(
        generator,
        discriminator,
        data_loader,
        num_epochs=NUM_EPOCHS,
        generator_lr=LEARNING_RATE_G,
        discriminator_lr=LEARNING_RATE_D,
        z_dim=Z_DIM,
        output_dir=OUT_DIR,
        samples_to_display=SAMPLES_TO_DISPLAY,
        sample_grid_columns=SAMPLE_GRID_COLUMNS,
        sample_every_epochs=SAMPLE_EVERY_EPOCHS,
    )


if __name__ == "__main__":
    main()
