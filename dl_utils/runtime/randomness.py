"""Seed selection and complete random-state capture and restoration."""

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def set_seed(seed: int | None = None) -> None:
    # Use a high-entropy / high-resolution seed when not provided.
    # NOTE: int(time.time()) has only 1s resolution and can collide easily.
    if seed is None:
        seed = int(torch.seed())
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG states."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore random-number generators captured by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"RNG state is missing keys: {missing}.")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())

    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        for device_index, device_state in enumerate(
            cuda_states[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(device_state.cpu(), device_index)


__all__ = ["capture_rng_state", "restore_rng_state", "set_seed"]
