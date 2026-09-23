"""MNIST and Fashion-MNIST train/test loaders with explicit lesson defaults."""

import os
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

from dl_utils.data.loading import TensorDataLoader, make_loader


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
    transform_list: list[transforms.ToTensor | transforms.Resize] = [
        transforms.ToTensor()
    ]
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
    train_iter = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    test_iter = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    return train_iter, test_iter


__all__ = [
    "vision_loaders",
]
