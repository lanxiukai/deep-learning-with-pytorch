"""Compare each discrete tokenizer with VAE in separate evaluation directories.

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
    for name, path in PAIR_CHECKPOINTS.items():
        output_dir = path.parent / "evaluation"
        reset_dir(output_dir)
        model_label = {"vq_vae": "VQ-VAE", "fsq": "FSQ"}[name]
        with tqdm(
            total=1 + len(TEMPERATURES), desc=f"Evaluate {model_label}", unit="grid"
        ) as progress:
            tokenizer, prior = load_pair(path, name, device, image_size=IMAGE_SIZE)
            assert isinstance(prior, PixelCNNPrior)
            save_image_row_grid(
                [originals, reconstruction.mul(2).sub(1), tokenizer(originals)[0]],
                ["Original", "VAE (mean)", model_label],
                output_dir / f"vae_{name}_reconstructions.png",
                title="Same training images: reconstruction",
                dpi=160,
            )
            progress.update(1)
            side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
            for temperature in TEMPERATURES:
                progress.set_postfix(temperature=temperature, refresh=False)
                with torch.random.fork_rng():
                    torch.manual_seed(SEED)
                    indices = prior.sample(
                        NUM_SAMPLES, side, side, device=device, temperature=temperature
                    )
                    samples = tokenizer.decode_indices(indices)
                save_image_row_grid(
                    [vae_samples.mul(2).sub(1), samples],
                    ["VAE", model_label],
                    output_dir / f"vae_{name}_samples_temperature_{temperature}.png",
                    title=f"Independent unconditional prior samples: temperature={temperature}",
                    dpi=160,
                )
                progress.update(1)


if __name__ == "__main__":
    evaluate()
