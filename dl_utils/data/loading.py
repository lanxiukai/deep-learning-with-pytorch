"""DataLoader construction, tensor batches, and explicit worker policies."""

import os
from functools import partial

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

type TensorBatch = tuple[torch.Tensor, torch.Tensor]
type TensorDataLoader = DataLoader[TensorBatch]


def _initialize_worker_sharing(worker_id: int, *, strategy: str) -> None:
    """Configure tensor sharing before a worker serializes batches."""
    torch.multiprocessing.set_sharing_strategy(strategy)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    collate_fn=None,
    worker_sharing_strategy: str | None = None,
) -> DataLoader:
    """Construct a loader without changing the caller's batching or device policy."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive.")
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        if worker_sharing_strategy is not None:
            loader_kwargs["worker_init_fn"] = partial(
                _initialize_worker_sharing,
                strategy=worker_sharing_strategy,
            )
    return DataLoader(**loader_kwargs)


def make_device_aware_loader(
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    collate_fn=None,
    worker_sharing_strategy: str | None = None,
) -> DataLoader:
    """Use device-appropriate pinning and preserve the parent worker-sharing policy."""
    if num_workers > 0 and worker_sharing_strategy is None:
        worker_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    return make_loader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        worker_sharing_strategy=worker_sharing_strategy,
    )


def resolve_num_workers(requested: int | None, fallback: int = 4) -> int:
    """Resolve a bounded DataLoader worker count for the current host."""
    if fallback < 0:
        raise ValueError("fallback must be non-negative.")
    num_workers = (
        min(8, os.cpu_count() or fallback) if requested is None else int(requested)
    )
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    return num_workers


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
    return make_loader(dataset, batch_size, shuffle=is_train)


__all__ = [
    "TensorBatch",
    "TensorDataLoader",
    "load_array",
    "make_device_aware_loader",
    "make_loader",
    "resolve_num_workers",
]
