"""Image reconstruction metrics shared by model families."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def structural_similarity_index(
    prediction: Tensor,
    target: Tensor,
    *,
    value_range: float = 2.0,
    window_size: int = 7,
) -> Tensor:
    """Return mean local SSIM for equally shaped image batches.

    The implementation uses a uniform local window so the lesson scripts do
    not need an additional image-metrics dependency.  It is intended for
    comparisons made with this exact protocol, not as a drop-in replacement
    for every SSIM package and preprocessing convention.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching [B,C,H,W] shapes")
    if value_range <= 0:
        raise ValueError("value_range must be positive")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    covariance = (
        F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding)
        - mean_prediction * mean_target
    )
    prediction_variance = (
        F.avg_pool2d(prediction.square(), window_size, stride=1, padding=padding)
        - mean_prediction.square()
    )
    target_variance = (
        F.avg_pool2d(target.square(), window_size, stride=1, padding=padding)
        - mean_target.square()
    )
    prediction_variance = prediction_variance.clamp_min(0.0)
    target_variance = target_variance.clamp_min(0.0)
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    score = (
        (2.0 * mean_prediction * mean_target + c1)
        * (2.0 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (prediction_variance + target_variance + c2)
        )
    )
    return score.flatten(1).mean(dim=1).mean()


__all__ = ["structural_similarity_index"]
