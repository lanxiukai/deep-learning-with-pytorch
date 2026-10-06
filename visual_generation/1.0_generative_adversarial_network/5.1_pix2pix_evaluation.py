"""Evaluate pix2pix on ten paired CelebA test images.

This mirrors 5.3_cycle_gan_evaluation.py, but every generated image has an
aligned target that can be inspected directly.

Input:
    output/gan/pix2pix/pix2pix_generator.pth, created by 5.0_pix2pix.py.

Outputs:
    output/gan/pix2pix/evaluation/, reset at the start of every run.
    - pix2pix_evaluation.png: input, generated, and target rows
"""

import torch
from torch.utils.data import DataLoader

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.pix2pix import (
    CelebAColorizationDataset,
    UNetGenerator,
    build_paired_transform,
)
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
CELEBA_DIR = PROJECT_ROOT / "data" / "celeba"
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "pix2pix" / "evaluation"
GENERATOR_PATH = OUT_DIR.parent / "pix2pix_generator.pth"
IMAGE_SIZE = 128
NUM_SAMPLES = 10


def main():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"pix2pix checkpoint not found: {GENERATOR_PATH}. Run 5.0_pix2pix.py first."
        )
    reset_dir(str(OUT_DIR))

    dataset = CelebAColorizationDataset(
        CELEBA_DIR,
        "test",
        build_paired_transform(
            training=False,
            image_size=IMAGE_SIZE,
        ),
    )
    device = try_gpu()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    generator = UNetGenerator().to(device)
    generator.load_state_dict(
        torch.load(GENERATOR_PATH, map_location=device, weights_only=True)
    )
    generator.eval()

    source_images = []
    generated_images = []
    target_images = []

    with torch.inference_mode():
        for index, (source, target) in enumerate(loader, start=1):
            source = source.to(device)
            generated = generator(source)

            source_images.append(source[0].cpu())
            generated_images.append(generated[0].cpu())
            target_images.append(target[0].cpu())

            if index == NUM_SAMPLES:
                break

    save_image_row_grid(
        [source_images, generated_images, target_images],
        ["input", "generated", "target"],
        OUT_DIR / "pix2pix_evaluation.png",
    )


if __name__ == "__main__":
    main()
