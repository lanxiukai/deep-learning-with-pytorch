"""Compatibility facade for the diffusion teaching utilities.

The implementation is split by algorithmic responsibility:

``diffusion_ddpm``
    Discrete VP schedules, prediction conversions, and DDPM/DDIM transitions.
``diffusion_score_sde``
    Continuous VP-SDE marginals, reverse SDE, and probability-flow ODE.
``diffusion_unet``
    The shared time-conditioned convolutional backbone.

Existing notebooks may continue importing from ``dl_utils.diffusion.ddpm``.  New
lessons import the focused modules directly so their dependencies remain clear.
"""

from dl_utils.diffusion.diffusion_ddpm import (
    DiffusionPrediction,
    DiscreteSampler,
    GaussianDiffusion,
    PredictionType,
    cosine_beta_schedule,
    linear_beta_schedule,
)
from dl_utils.diffusion.diffusion_score_sde import (
    VPSDE,
    ScoreSampler,
    sample_vp_sde,
)
from dl_utils.diffusion.diffusion_unet import DiffusionUNet, UNet

__all__ = [
    "VPSDE",
    "DiffusionPrediction",
    "DiffusionUNet",
    "DiscreteSampler",
    "GaussianDiffusion",
    "PredictionType",
    "ScoreSampler",
    "UNet",
    "cosine_beta_schedule",
    "linear_beta_schedule",
    "sample_vp_sde",
]
