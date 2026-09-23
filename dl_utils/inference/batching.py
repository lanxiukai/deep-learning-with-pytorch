"""Bounded inference for aligned tensor inputs and tensor outputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


@torch.inference_mode()
def generate_in_batches(
    inputs: torch.Tensor | Sequence[torch.Tensor],
    batch_size: int,
    generate: Callable[..., torch.Tensor],
    *,
    module: nn.Module | None = None,
    output_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate aligned tensor inputs in bounded batches and concatenate."""
    input_tensors = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not input_tensors:
        raise ValueError("batched inference requires at least one input tensor.")
    if any(not isinstance(value, torch.Tensor) for value in input_tensors):
        raise TypeError("all batched inference inputs must be tensors.")
    if input_tensors[0].ndim == 0 or input_tensors[0].shape[0] == 0:
        raise ValueError("batched inference requires at least one input.")
    num_inputs = input_tensors[0].shape[0]
    if any(
        value.ndim == 0 or value.shape[0] != num_inputs for value in input_tensors[1:]
    ):
        raise ValueError("batched inference inputs must have equal lengths.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    was_training = module.training if module is not None else False
    if module is not None:
        module.eval()
    try:
        outputs = []
        split_inputs = [value.split(batch_size) for value in input_tensors]
        for batches in zip(*split_inputs, strict=True):
            generated = generate(*batches)
            if not isinstance(generated, torch.Tensor):
                raise TypeError("generate must return a tensor.")
            outputs.append(generated.to(output_device))
        return torch.cat(outputs)
    finally:
        if module is not None:
            module.train(was_training)


__all__ = ["generate_in_batches"]
