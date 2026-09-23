"""Generic ImageFolder helpers and retained D2L/EBM data entry points."""

from pathlib import Path

import torch
import torchvision
from torchvision import transforms

from dl_utils.data.datasets.vision import vision_loaders
from dl_utils.data.images import load_rgb_image
from dl_utils.data.loading import TensorBatch, TensorDataLoader, load_array, make_loader


def image_folder_dataset(
    root: str | Path,
    resize: int | tuple[int, int] | None = None,
    normalize: tuple[float, float] | None = None,
) -> torchvision.datasets.ImageFolder:
    """
    Build an ``ImageFolder`` dataset (subdirs = classes) with a standard transform.

    Args:
        root:      Root directory path for ``ImageFolder``.
        resize:    If provided, resize images to this size (applied BEFORE ``ToTensor``).
        normalize: If provided as ``(mean, std)``, append
                   ``transforms.Normalize(mean, std)`` after ``ToTensor`` —
                   e.g. ``(0.5, 0.5)`` maps [0, 1] to [-1, 1].

    Returns:
        The dataset; ``ds[i]`` yields ``(tensor, label)`` with tensor float32
        (C×H×W) in [0, 1] (or normalized), label an int class index.

    Images with transparency are composited onto white before conversion to
    RGB so palette alpha tables are preserved without Pillow warnings.
    """
    # Transform chain: optional Resize (on PIL) → ToTensor (→float32 [0,1])
    # → optional Normalize
    transform_list: list = [transforms.ToTensor()]
    if resize is not None:
        transform_list.insert(0, transforms.Resize(resize))
    if normalize is not None:
        transform_list.append(transforms.Normalize(*normalize))
    transform = transforms.Compose(transform_list)
    return torchvision.datasets.ImageFolder(
        root=root,
        transform=transform,
        loader=load_rgb_image,
    )


def image_folder_loader(
    root: str | Path,
    batch_size: int,
    resize: int | tuple[int, int] | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize: tuple[float, float] | None = None,
) -> TensorDataLoader:
    """
    Load images from a directory organized in ``ImageFolder`` layout (subdirs = classes).

    Produces a single DataLoader — no built-in train/test split.

    Args:
        root:        Root directory path for ``ImageFolder``.
        batch_size:  Batch size.
        resize:      If provided, resize images to this size (applied BEFORE ``ToTensor``).
        shuffle:     Whether to shuffle each epoch (default ``True``).
        num_workers: Number of data-loading subprocesses (default 0).
        pin_memory:  Pin memory for faster GPU transfer (default ``False``).
        normalize:   If provided as ``(mean, std)``, normalize after ``ToTensor``
                     — e.g. ``(0.5, 0.5)`` maps [0, 1] to [-1, 1].

    Returns:
        A single ``DataLoader`` over the images in ``root``.
    """
    # ds[i] returns (tensor, label): tensor is float32 (C×H×W) in [0,1]
    # (or normalized), label is int
    ds = image_folder_dataset(root, resize=resize, normalize=normalize)

    pin = pin_memory and torch.cuda.is_available()
    loader = make_loader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    return loader


__all__ = [
    "TensorBatch",
    "TensorDataLoader",
    "image_folder_dataset",
    "image_folder_loader",
    "load_array",
    "vision_loaders",
]
