"""Completeness checks and local attribute splits for CelebA preparation."""

import csv
import os
import shutil
from pathlib import Path

CELEBA_IMAGE_COUNT = 202_599


def celeba_dataset_is_ready(
    celeba_dir: Path,
    *,
    expected_images: int = CELEBA_IMAGE_COUNT,
) -> bool:
    """Return whether the official CelebA metadata and images are complete."""
    celeba_dir = Path(celeba_dir)
    required_files = (
        celeba_dir / "list_attr_celeba.csv",
        celeba_dir / "list_eval_partition.csv",
    )
    image_dir_candidates = (
        celeba_dir / "img_align_celeba" / "img_align_celeba",
        celeba_dir / "img_align_celeba",
    )
    image_dir = next(
        (path for path in image_dir_candidates if path.is_dir()),
        None,
    )
    if image_dir is None or not all(path.is_file() for path in required_files):
        return False
    return sum(1 for path in image_dir.glob("*.jpg") if path.is_file()) == (
        expected_images
    )


def prepare_celeba_cyclegan_splits(celeba_dir: Path) -> bool:
    """Create black- and blond-hair splits without changing source images."""
    celeba_dir = Path(celeba_dir)
    attributes_file = celeba_dir / "list_attr_celeba.csv"
    image_dir_candidates = (
        celeba_dir / "img_align_celeba" / "img_align_celeba",
        celeba_dir / "img_align_celeba",
    )
    image_dir = next(
        (path for path in image_dir_candidates if path.is_dir()),
        None,
    )
    if not attributes_file.is_file() or image_dir is None:
        print("  SKIP: CelebA files are incomplete; cannot prepare CycleGAN splits.")
        return False

    black_dir = celeba_dir / "black"
    blond_dir = celeba_dir / "blond"
    black_dir.mkdir(exist_ok=True)
    blond_dir.mkdir(exist_ok=True)

    linked = 0
    with attributes_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target_dir = (
                black_dir
                if int(row["Black_Hair"]) == 1
                else blond_dir
                if int(row["Blond_Hair"]) == 1
                else None
            )
            if target_dir is None:
                continue
            source = image_dir / row["image_id"]
            target = target_dir / row["image_id"]
            if target.exists() or not source.is_file():
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            linked += 1

    black_count = sum(1 for path in black_dir.iterdir() if path.is_file())
    blond_count = sum(1 for path in blond_dir.iterdir() if path.is_file())
    print(
        f"  CycleGAN local splits ready: {black_dir} "
        f"({black_count} images), {blond_dir} ({blond_count} images); "
        f"added {linked}."
    )
    return True
