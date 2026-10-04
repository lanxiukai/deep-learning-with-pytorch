"""Reconstruct an explicit lesson model; reject old dataset/codec contracts."""

from pathlib import Path

from dl_utils.diffusion.checkpoints import build_model as build_foundation_model
from dl_utils.diffusion.checkpoints import load_model as load_checkpoint
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.modern.consistency import ConsistencyModel
from dl_utils.modern.edm import EDMPreconditioner
from dl_utils.modern.improved_mean_flow import ImprovedMeanFlow
from dl_utils.modern.kl_autoencoder import KLPerceptualAutoencoder
from dl_utils.modern.sr3 import SR3
from dl_utils.modern.transformer import DiffusionTransformer


def build_model(kind, config):
    if kind == "edm":
        options = dict(config)
        return EDMPreconditioner(
            DiffusionUNet(**options.pop("network_config")), **options
        )
    constructors = {
        "dit": DiffusionTransformer,
        "imf": ImprovedMeanFlow,
        "autoencoder": KLPerceptualAutoencoder,
        "sr3": SR3,
        "consistency": ConsistencyModel,
    }
    if kind in constructors:
        return constructors[kind](**config)
    return build_foundation_model(kind, config)


def load_model(path, device="cpu", *, ema=True):
    return load_checkpoint(path, device, ema=ema, model_builder=build_model)


def load_codec(path, device, expected=None):
    if path is None:
        raise ValueError("Supply --autoencoder-checkpoint from 1.0_kl_autoencoder.py.")
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
