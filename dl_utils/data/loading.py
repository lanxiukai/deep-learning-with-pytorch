"""Device-aware DataLoader construction shared by image lessons."""

from __future__ import annotations

import os
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset


def _initialize_worker_sharing(worker_id: int, *, strategy: str) -> None:
    """Configure tensor sharing before a worker serializes batches."""
    torch.multiprocessing.set_sharing_strategy(strategy)


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
    """Create a DataLoader with device-appropriate memory settings."""
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
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = partial(
            _initialize_worker_sharing,
            strategy=(
                torch.multiprocessing.get_sharing_strategy()
                if worker_sharing_strategy is None
                else worker_sharing_strategy
            ),
        )
    return DataLoader(
        **loader_kwargs,
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


__all__ = ["make_device_aware_loader", "resolve_num_workers"]
