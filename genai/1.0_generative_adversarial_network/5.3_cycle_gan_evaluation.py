"""Evaluate trained CycleGAN generators for hair-color translation.

This script performs both translation directions and cycle reconstruction on
the first ten unpaired black/blond image pairs.

Data:
    data/celeba/{black,blond}, prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Inputs:
    - output/gan/cyclegan/gen_black.pth: blond -> black generator
    - output/gan/cyclegan/gen_blond.pth: black -> blond generator
    Run 5.2_cycle_gan.py first to create both checkpoints.

Outputs:
    output/gan/cyclegan/evaluation/, reset at the start of every run.
    - black_to_blond_cycle.png: black, fake_blond, and fake2black rows
    - blond_to_black_cycle.png: blond, fake_black, and fake2blond rows
"""

import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.cyclegan import CycleGANGenerator, UnpairedImageDataset
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "cyclegan" / "evaluation"
CHECKPOINT_DIR = OUT_DIR.parent

BLACK_DIR = DATA_DIR / "celeba" / "black"
BLOND_DIR = DATA_DIR / "celeba" / "blond"
GEN_BLACK_PATH = CHECKPOINT_DIR / "gen_black.pth"
GEN_BLOND_PATH = CHECKPOINT_DIR / "gen_blond.pth"
NUM_SAMPLES = 10
IMAGE_SIZE = 128
NUM_RESIDUAL_BLOCKS = 9


def main():
    """Translate ten black/blond image pairs and reconstruct each input."""
    reset_dir(str(OUT_DIR))

    missing_data_dirs = [path for path in (BLACK_DIR, BLOND_DIR) if not path.is_dir()]
    if missing_data_dirs:
        missing = ", ".join(str(path) for path in missing_data_dirs)
        raise FileNotFoundError(
            f"CycleGAN evaluation data not found: {missing}. "
            "Run: python tool_scripts/download_dataset.py --dataset celeba"
        )

    missing_checkpoints = [
        path for path in (GEN_BLACK_PATH, GEN_BLOND_PATH) if not path.is_file()
    ]
    if missing_checkpoints:
        missing = ", ".join(str(path) for path in missing_checkpoints)
        raise FileNotFoundError(
            f"CycleGAN checkpoints not found: {missing}. Run 5.2_cycle_gan.py first."
        )

    transforms = albumentations.Compose(
        [
            albumentations.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
            albumentations.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=255,
            ),
            ToTensorV2(),  # numpy H×W×C → torch C×H×W
        ],
        additional_targets={"image0": "image"},
    )
    dataset = UnpairedImageDataset(
        domain_a_roots=[str(BLACK_DIR)],
        domain_b_roots=[str(BLOND_DIR)],
        transform=transforms,
    )

    device = try_gpu()
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=device.type == "cuda"
    )

    black_generator = CycleGANGenerator(
        image_channels=3,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
    ).to(device)
    blond_generator = CycleGANGenerator(
        image_channels=3,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
    ).to(device)
    black_generator.load_state_dict(
        torch.load(
            GEN_BLACK_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    blond_generator.load_state_dict(
        torch.load(
            GEN_BLOND_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    black_generator.eval()
    blond_generator.eval()

    black_images = []
    fake_blond_images = []
    fake2black_images = []
    blond_images = []
    fake_black_images = []
    fake2blond_images = []

    with torch.inference_mode():
        for index, (black, blond) in enumerate(loader, start=1):
            black = black.to(device)
            blond = blond.to(device)

            fake_blond = blond_generator(black)
            fake2black = black_generator(fake_blond)
            fake_black = black_generator(blond)
            fake2blond = blond_generator(fake_black)

            black_images.append(black[0].cpu())
            fake_blond_images.append(fake_blond[0].cpu())
            fake2black_images.append(fake2black[0].cpu())
            blond_images.append(blond[0].cpu())
            fake_black_images.append(fake_black[0].cpu())
            fake2blond_images.append(fake2blond[0].cpu())

            if index == NUM_SAMPLES:
                break

    save_image_row_grid(
        [black_images, fake_blond_images, fake2black_images],
        ["black", "fake_blond", "fake2black"],
        OUT_DIR / "black_to_blond_cycle.png",
    )
    save_image_row_grid(
        [blond_images, fake_black_images, fake2blond_images],
        ["blond", "fake_black", "fake2blond"],
        OUT_DIR / "blond_to_black_cycle.png",
    )


if __name__ == "__main__":
    main()
