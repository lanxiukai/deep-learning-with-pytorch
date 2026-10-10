"""Inspect VQGAN reconstructions and unconditional generation after training.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the first 8 training images without shuffling.
    Resize to 256x256 RGB and normalize from [0, 1] to [-1, 1].
    Class labels are ignored for both reconstruction and generation.

Checkpoint:
    output/vae/vqgan/model.pth: final tokenizer/unconditional Transformer pair.
    Run 7.0 first. Model dimensions are loaded and checked.
    Recovery files tokenizer_latest.pth and prior_latest.pth are not used here.

Outputs:
    output/vae/vqgan/evaluation/vqgan_real_and_reconstruction.png
    output/vae/vqgan/evaluation/vqgan_prior_samples_temperature_<temperature>.png
    The evaluation directory is reset before writing its four grids.

Evaluation defaults:
    Reconstruction images: 8 training inputs; one batch.
    Generated images:      8 per temperature, arranged in two rows; 256x256 RGB.
    Token grid / sequence: 16x16 / 256 tokens with default four downsampling stages.
    Prior temperatures:    0.7, 1.0, 1.3.
    Seed:                  123 for repeatable unconditional sampling.
    Model dimensions:      loaded from the checkpoint produced by 7.0.

Run without arguments after training 7.0; edit the constants below to change
the checkpoint, reconstruction count, sample count, or temperature sweep.
"""

import math

import torch
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.discrete_workflow import glasses_loader, load_pair
from dl_utils.vae.token_priors import CausalTransformerPrior

# Paths and checkpoints
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "vqgan" / "model.pth"
OUTPUT_DIR = CHECKPOINT.parent / "evaluation"

# Image configuration
IMAGE_SIZE = 256
RECONSTRUCTION_SAMPLES = 8
NUM_SAMPLES = 8
SAMPLES_PER_ROW = 4

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
    with tqdm(
        total=1 + len(TEMPERATURES), desc="Evaluate VQGAN", unit="grid"
    ) as progress:
        tokenizer, prior = load_pair(CHECKPOINT, "vqgan", device, image_size=IMAGE_SIZE)
        assert isinstance(prior, CausalTransformerPrior)
        loader = glasses_loader(DATA_DIR, IMAGE_SIZE, RECONSTRUCTION_SAMPLES, device)
        originals = next(iter(loader))[0].to(device)
        reconstruction = tokenizer(originals)[0]
        side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
        save_image_row_grid(
            [originals, reconstruction],
            ["Original", "VQGAN"],
            OUTPUT_DIR / "vqgan_real_and_reconstruction.png",
            title="Same training images: reconstruction",
            dpi=160,
        )
        progress.update(1)
        for temperature in TEMPERATURES:
            progress.set_postfix(temperature=temperature, refresh=False)
            with torch.random.fork_rng():
                torch.manual_seed(SEED)
                indices = prior.sample(
                    NUM_SAMPLES, device=device, temperature=temperature
                )
                samples = tokenizer.decode_indices(
                    indices.reshape(NUM_SAMPLES, side, side)
                )
            rows = samples.split(SAMPLES_PER_ROW)
            save_image_row_grid(
                rows,
                ["VQGAN"] * len(rows),
                OUTPUT_DIR / f"vqgan_prior_samples_temperature_{temperature}.png",
                title=f"Unconditional prior samples: temperature={temperature}",
                dpi=160,
            )
            progress.update(1)


if __name__ == "__main__":
    evaluate()
