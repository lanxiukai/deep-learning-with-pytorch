"""Compare VAE, VQ-VAE, and FSQ with reconstruction and temperature sample grids.

Run 1.0, 6.0, and 6.1 first. Reconstructions use the same training images;
generation rows are independent unconditional samples, with no matched identities.
TEMPERATURES controls the discrete priors; the VAE row stays fixed for comparison.
"""

import math
from typing import cast

import torch
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights
from dl_utils.vae.discrete_workflow import glasses_loader, load_pair
from dl_utils.vae.token_priors import PixelCNNPrior
from dl_utils.vae.vae import VAE

# Paths and checkpoints
PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = OUTPUT_ROOT / "evaluation" / "discrete_tokenizer"
VAE_CHECKPOINT = OUTPUT_ROOT / "vae" / "vae.pth"
PAIR_CHECKPOINTS = {
    "vq_vae": OUTPUT_ROOT / "vq_vae" / "model.pth",
    "fsq": OUTPUT_ROOT / "fsq" / "model.pth",
}

# Image configuration
IMAGE_SIZE = 256
NUM_SAMPLES = 8

# Sampling configuration
TEMPERATURES = (0.7, 1.0, 1.3)
SEED = 123


@torch.inference_mode()
def evaluate():
    if not TEMPERATURES or any(
        not math.isfinite(temperature) or temperature <= 0
        for temperature in TEMPERATURES
    ):
        raise ValueError("TEMPERATURES must contain finite, positive values.")
    reset_dir(OUTPUT_DIR)
    set_seed(SEED)
    device = try_gpu()
    loader = glasses_loader(DATA_DIR, IMAGE_SIZE, NUM_SAMPLES, device)
    originals = next(iter(loader))[0].to(device)
    vae, _ = load_model_weights(
        VAE_CHECKPOINT,
        VAE,
        device=device,
        expected_metadata={
            "model_name": "vae",
            "backbone": VAE.backbone,
            "dataset": "glasses-256",
            "value_range": [0.0, 1.0],
            "beta": 1.0,
        },
    )
    vae = cast(VAE, vae)
    reconstruction = vae.reconstruct(originals.mul(0.5).add(0.5), sample=False)[2]
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        vae_samples = vae.decoder(torch.randn(NUM_SAMPLES, vae.z_dim, device=device))
    reconstructions = [originals, reconstruction.mul(2).sub(1)]
    pairs = {}
    for name, path in PAIR_CHECKPOINTS.items():
        tokenizer, prior = load_pair(path, name, device, image_size=IMAGE_SIZE)
        assert isinstance(prior, PixelCNNPrior)
        reconstructions.append(tokenizer(originals)[0])
        pairs[name] = (tokenizer, prior)
    save_image_row_grid(
        reconstructions,
        ["Original", "VAE (mean)", "VQ-VAE", "FSQ"],
        OUTPUT_DIR / "vae_vq_vae_fsq_reconstructions.png",
        title="Same training images: reconstruction",
        dpi=160,
    )
    with tqdm(
        TEMPERATURES, desc="Generate VQ-VAE / FSQ", unit="temperature"
    ) as progress:
        for temperature in progress:
            progress.set_postfix(temperature=temperature, refresh=False)
            generations = [vae_samples.mul(2).sub(1)]
            for tokenizer, prior in pairs.values():
                side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
                with torch.random.fork_rng():
                    torch.manual_seed(SEED)
                    indices = prior.sample(
                        NUM_SAMPLES, side, side, device=device, temperature=temperature
                    )
                    generations.append(tokenizer.decode_indices(indices))
            save_image_row_grid(
                generations,
                ["VAE", "VQ-VAE", "FSQ"],
                OUTPUT_DIR / f"vae_vq_vae_fsq_samples_temperature_{temperature}.png",
                title=f"Independent unconditional prior samples: temperature={temperature}",
                dpi=160,
            )


if __name__ == "__main__":
    evaluate()
