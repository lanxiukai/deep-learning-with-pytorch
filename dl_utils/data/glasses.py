"""Shared contract for the locally prepared glasses-256 image dataset."""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder

from dl_utils.data.vision import image_folder_dataset

GLASSES_CLASS_NAMES = ("G", "NoG")
GLASSES_IMAGE_SIZE = 256


def glasses_data_config() -> dict[str, object]:
    """Describe the cache and image contract used for checkpoint validation."""
    return {
        "dataset": "glasses-256",
        "class_names": list(GLASSES_CLASS_NAMES),
        "image_size": GLASSES_IMAGE_SIZE,
        "image_channels": 3,
        "pixel_range": [0.0, 1.0],
        "split": "all images (training set)",
    }


def glasses_dataset(root: Path) -> ImageFolder:
    """Read the prepared RGB cache and validate its class-directory contract."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset cache not found: {root}. Prepare it with "
            "python tool_scripts/download_dataset.py --dataset glasses"
        )
    dataset = image_folder_dataset(root)
    if dataset.classes != list(GLASSES_CLASS_NAMES):
        raise ValueError(
            f"Expected classes {GLASSES_CLASS_NAMES}, got {dataset.classes}"
        )
    return dataset


__all__ = [
    "GLASSES_CLASS_NAMES",
    "GLASSES_IMAGE_SIZE",
    "glasses_data_config",
    "glasses_dataset",
]
