"""Load the explicit model/path contracts written by the pixel lessons."""

import torch

from dl_utils.diffusion.diffusion_ddpm import GaussianDiffusion
from dl_utils.diffusion.diffusion_score_sde import make_sde
from dl_utils.diffusion.diffusion_unet import DiffusionUNet
from dl_utils.diffusion.edm import EDMPreconditioner
from dl_utils.diffusion.flow_matching import GaussianConditionalPath
from dl_utils.diffusion.improved_ddpm import ImprovedDDPM


def load_pixel_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    kind = checkpoint["algorithm"]
    config = checkpoint["model_config"]
    if kind == "edm":
        config = dict(config)
        model = EDMPreconditioner(
            DiffusionUNet(**config.pop("network_config")), **config
        )
        process = None
        resolution = model.network.sample_size
    else:
        model = DiffusionUNet(**config)
        resolution = model.sample_size
        if kind in ("vp_ddpm", "improved_ddpm"):
            diffusion_type = (
                ImprovedDDPM if kind == "improved_ddpm" else GaussianDiffusion
            )
            process = diffusion_type(**checkpoint["diffusion_config"]).to(device)
        elif kind == "score_sde":
            process = make_sde(checkpoint["sde_name"], **checkpoint["sde_config"])
        elif kind == "flow_matching":
            if (
                checkpoint["prediction_type"] != "velocity"
                or checkpoint["time_direction"] != "noise_to_data"
            ):
                raise ValueError("CFM requires dx/dt with noise at 0 and data at 1.")
            process = GaussianConditionalPath(**checkpoint["flow_config"])
        else:
            raise ValueError(
                "Expected a pixel DDPM, score-SDE, flow-matching, or extension checkpoint."
            )
    if resolution < 128 or checkpoint.get("training", {}).get("dataset") != "CelebA":
        raise ValueError("Use a CelebA checkpoint from the 128px-or-larger lessons.")
    model.load_state_dict(checkpoint["ema_state"])
    return (
        model.to(device).eval().requires_grad_(False),
        process,
        checkpoint,
        resolution,
    )
