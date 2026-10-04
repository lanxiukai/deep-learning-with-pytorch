"""Foundation model construction and shared format-4 checkpoint restoration."""

import torch

from dl_utils.diffusion.classifiers import ImageClassifier
from dl_utils.diffusion.diffusion_unet import DiffusionUNet


def build_model(kind, config):
    constructors = {"unet": DiffusionUNet, "classifier": ImageClassifier}
    if kind not in constructors:
        raise ValueError(
            "This is a modern model; use the modern visual generation loader."
        )
    return constructors[kind](**config)


def load_model(path, device="cpu", *, ema=True, model_builder=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 4:
        raise ValueError(
            "Expected a Food-101 format-4 checkpoint; retrain the old CelebA lessons."
        )
    model = (model_builder or build_model)(state["kind"], state["model_config"])
    model.load_state_dict(state["ema" if ema else "model"])
    return model.to(device).eval().requires_grad_(False), state
