"""Inspect VQ-VAE and FSQ reconstructions and unconditional prior samples.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the first 8 training images without shuffling; class labels are ignored.
    Discrete inputs are resized to 256x256 RGB and normalized to [-1, 1].

Checkpoints:
    output/vae/vq_vae/model.pth: VQ-VAE/PixelCNN pair; run 6.0 first
    output/vae/fsq/model.pth: FSQ/PixelCNN pair; run 6.1 first
    Model dimensions are loaded from each checkpoint.

Outputs:
    Under output/vae/<name>/evaluation/, name = vq_vae or fsq:
    <name>_reconstructions.png: original/tokenizer rows
    <name>_samples_temperature_<temperature>.png: tokenizer sample grid
    Each evaluation directory is reset before writing its four grids.

Evaluation defaults:
    Reconstruction images: 8 shared training inputs; one batch.
    Generated images:      16 per model and temperature; 2 rows of 8; 256x256 RGB.
    Discrete token grid:   32x32 with the default three downsampling stages.
    Prior temperatures:    0.9, 1.0, 1.1.
    Seed:                  123 for controlled prior sampling.

Run without arguments after training 6.0 and 6.1; edit the constants
below to change the checkpoints, sample count, or temperature sweep.
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
from dl_utils.vae.token_priors import PixelCNNPrior

# Paths and checkpoints
PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
PAIR_CHECKPOINTS = {
    "vq_vae": OUTPUT_ROOT / "vq_vae" / "model.pth",
    "fsq": OUTPUT_ROOT / "fsq" / "model.pth",
}

# Image configuration
IMAGE_SIZE = 256
RECONSTRUCTION_SAMPLES = 8
NUM_SAMPLES = 16
SAMPLE_GRID_COLUMNS = 8

# Sampling configuration
TEMPERATURES = (0.9, 1.0, 1.1)
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
    loader = glasses_loader(DATA_DIR, IMAGE_SIZE, RECONSTRUCTION_SAMPLES, device)
    originals = next(iter(loader))[0].to(device)
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
                [originals, tokenizer(originals)[0]],
                ["Original", model_label],
                output_dir / f"{name}_reconstructions.png",
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
                sample_rows = samples.split(SAMPLE_GRID_COLUMNS)
                save_image_row_grid(
                    sample_rows,
                    [model_label] * len(sample_rows),
                    output_dir / f"{name}_samples_temperature_{temperature}.png",
                    title=f"Independent unconditional prior samples: temperature={temperature}",
                    dpi=160,
                )
                progress.update(1)


if __name__ == "__main__":
    evaluate()
