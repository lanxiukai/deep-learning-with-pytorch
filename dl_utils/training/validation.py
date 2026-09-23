"""Finite-value validation for model and optimizer state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def _iter_named_tensors(value: Any, prefix: str):
    """Yield tensors from nested mappings and sequences with readable names."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_named_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_named_tensors(child, f"{prefix}[{index}]")


def validate_finite_training_state(
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    *,
    extra_tensors: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Reject non-finite model, optimizer, or auxiliary training state."""
    named_tensors = []
    for name, model in models.items():
        named_tensors.extend(_iter_named_tensors(model.state_dict(), f"models.{name}"))
    for name, optimizer in optimizers.items():
        named_tensors.extend(
            _iter_named_tensors(tuple(optimizer.state.values()), f"optimizers.{name}")
        )
    named_tensors.extend(_iter_named_tensors(dict(extra_tensors or {}), "extra"))
    tensors_by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in named_tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            tensors_by_device.setdefault(tensor.device, []).append((name, tensor))

    nonfinite = []
    for entries in tensors_by_device.values():
        finite = torch.stack([torch.isfinite(tensor).all() for _, tensor in entries])
        finite_values = finite.cpu().tolist()
        nonfinite.extend(
            name
            for (name, _), is_finite in zip(entries, finite_values, strict=True)
            if not is_finite
        )
    if nonfinite:
        shown = ", ".join(nonfinite[:5])
        suffix = "" if len(nonfinite) <= 5 else f" (+{len(nonfinite) - 5} more)"
        raise FloatingPointError(f"non-finite training state: {shown}{suffix}")


__all__ = ["validate_finite_training_state"]
