"""Reconstruct an explicit lesson model; reject old dataset/codec contracts."""

from pathlib import Path

import torch

from dl_utils.diffusion.classifiers import ImageClassifier
from dl_utils.diffusion.consistency import ConsistencyModel
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.edm import EDMPreconditioner
from dl_utils.diffusion.improved_mean_flow import ImprovedMeanFlow
from dl_utils.diffusion.kl_autoencoder import KLPerceptualAutoencoder
from dl_utils.diffusion.sr3 import SR3
from dl_utils.diffusion.transformer import DiffusionTransformer


def build_model(kind, config):
    if kind == "edm":
        options = dict(config)
        return EDMPreconditioner(
            DiffusionUNet(**options.pop("network_config")), **options
        )
    constructors = {
        "unet": DiffusionUNet,
        "dit": DiffusionTransformer,
        "imf": ImprovedMeanFlow,
        "autoencoder": KLPerceptualAutoencoder,
        "classifier": ImageClassifier,
        "sr3": SR3,
        "consistency": ConsistencyModel,
    }
    return constructors[kind](**config)


def load_model(path, device="cpu", *, ema=True):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 4:
        raise ValueError(
            "Expected a Food-101 format-4 checkpoint; retrain the old CelebA lessons."
        )
    model = build_model(state["kind"], state["model_config"])
    model.load_state_dict(state["ema" if ema else "model"])
    return model.to(device).eval().requires_grad_(False), state


def load_codec(path, device, expected=None):
    if path is None:
        raise ValueError("Supply --autoencoder-checkpoint from 5.0_kl_autoencoder.py.")
    codec, state = load_model(path, device)
    if state["kind"] != "autoencoder" or state.get("latent_scale", 0) <= 0:
        raise ValueError(
            "The codec checkpoint must contain calibrated posterior statistics."
        )
    if expected is not None:
        for key in ("classes", "image_size", "split_seed", "preprocessing"):
            if state["data_config"][key] != expected[key]:
                raise ValueError(f"Codec data contract differs: {key}.")
    return codec, state["latent_scale"], state


def codec_for_generation(state, device, override=None):
    algorithm = state["algorithm"]
    if not algorithm.get("latent", False):
        return None, 1.0
    path = override or algorithm["codec_checkpoint"]
    codec, scale, codec_state = load_codec(Path(path), device, state["data_config"])
    if (
        codec_state["checkpoint_id"] != algorithm["codec_id"]
        or scale != algorithm["latent_scale"]
    ):
        raise ValueError(
            "Different codec checkpoint or latent scale; specify the exact frozen first stage."
        )
    return codec, scale
