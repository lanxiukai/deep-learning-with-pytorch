import os
from pathlib import Path
from typing import TypeAlias

import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

from dl_utils.data.images import load_rgb_image


TensorBatch: TypeAlias = tuple[torch.Tensor, torch.Tensor]
TensorDataLoader: TypeAlias = data.DataLoader[TensorBatch]


def vision_loaders(
    dataset: str,
    data_dir: str | Path,
    batch_size: int,
    resize: int | tuple[int, int] | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[TensorDataLoader, TensorDataLoader]:
    """
    A unified torchvision vision dataset loader (train/test).

    Supported datasets:
      - "mnist"
      - "fashion_mnist" (also accepts "fashionmnist", "fmnist")

    Notes:
      - We normalize to [0,1] via ToTensor(), then binarize (if needed) in the training loop.
      - Resize (if provided) is applied BEFORE ToTensor() (i.e. on PIL images).
    """
    os.makedirs(data_dir, exist_ok=True)

    # Basic image transform: optional resize (on PIL) -> ToTensor (float32 in [0,1])
    transform_list: list[transforms.ToTensor | transforms.Resize] = [transforms.ToTensor()]
    if resize is not None:
        transform_list.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform_list)

    key = dataset.strip().lower().replace("-", "").replace("_", "")
    if key == "mnist":
        ds_cls = torchvision.datasets.MNIST
    elif key in {"fashionmnist", "fmnist"}:
        ds_cls = torchvision.datasets.FashionMNIST
    else:
        raise ValueError(
            f"Unknown dataset={dataset!r}. Supported: 'mnist', "
            "'fashion_mnist' (or 'fmnist')."
        )

    train_ds = ds_cls(root=data_dir, train=True, transform=transform, download=True)
    test_ds = ds_cls(root=data_dir, train=False, transform=transform, download=True)

    pin = pin_memory and torch.cuda.is_available()
    persistent = num_workers > 0
    train_iter = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
    )
    test_iter = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )
    return train_iter, test_iter


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
    persistent = num_workers > 0
    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
    )
    return loader


def load_array(data_arrays, batch_size, is_train=True):
    """
    Construct a PyTorch data iterator from raw tensors.

    Args:
        data_arrays: a tuple of tensors (features, labels)
        batch_size:  the batch size
        is_train:    whether to shuffle the data (default True)

    Returns:
        A ``DataLoader`` over the tensor dataset.
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
