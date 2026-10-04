"""Inspect VQGAN reconstructions and class-conditional generation after training.

Reconstruction and generation flow:
    x -> encoder -> nearest codebook tokens -> decoder -> reconstruction
    c -> causal Transformer -> sampled token sequence -> codebook -> decoder
The saved tokenizer is unconditional; the Transformer prior supplies G/NoG
conditioning. A two-row reconstruction grid compares each original with its
own reconstruction. Each temperature gets a separate two-row prior grid,
with four G samples followed by four NoG samples. Generation columns do not
match identities across classes and are independent of the reconstruction inputs.

Temperature divides prior logits before categorical sampling. The seed is
reset for every temperature to control the random stream, without guaranteeing
matched identities or token sequences. These visual training-set diagnostics
do not compute numerical reconstruction, likelihood, or generation metrics.
See dl_utils/vae/discrete_workflow.py for pair loading and token-grid checks.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the first 8 training images without shuffling.
    Resize to 256x256 RGB and normalize from [0, 1] to [-1, 1].
    Class order is G=0 (with glasses), NoG=1 (without glasses).
    Reconstruction inputs do not need a class label; prior samples do.

Checkpoint:
    output/vae/vqgan/model.pth: final tokenizer/conditional Transformer pair.
    Run 7.0 first. Model dimensions and class order are loaded and checked.
    Recovery files tokenizer_latest.pth and prior_latest.pth are not used here.

Outputs:
    output/vae/vqgan/evaluation/vqgan_real_and_reconstruction.png
    output/vae/vqgan/evaluation/vqgan_prior_samples_temperature_<temperature>.png
    The evaluation directory is reset before writing its four grids.

Evaluation defaults:
    Reconstruction images: 8 training inputs; one batch.
    Generated images:      4 per class, 8 per temperature; 256x256 RGB.
    Token grid / sequence: 16x16 / 256 tokens with default four downsampling stages.
    Prior temperatures:    0.7, 1.0, 1.3.
    Seed:                  123 for controlled conditional sampling.
    Model dimensions:      loaded from the checkpoint produced by 7.0.

Run without arguments after training 7.0; edit the constants below to change
the checkpoint, reconstruction count, class sample count, or temperature sweep.
"""

import math

import torch
from tqdm.auto import tqdm

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
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
SAMPLES_PER_CLASS = 4

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
        loader = glasses_loader(
            DATA_DIR, IMAGE_SIZE, RECONSTRUCTION_SAMPLES, device, conditional=True
        )
        originals = next(iter(loader))[0].to(device)
        reconstruction = tokenizer(originals)[0]
        labels = torch.arange(
            len(GLASSES_CLASS_NAMES), device=device
        ).repeat_interleave(SAMPLES_PER_CLASS)
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
                    len(labels), device=device, labels=labels, temperature=temperature
                )
                samples = tokenizer.decode_indices(
                    indices.reshape(len(labels), side, side)
                )
            save_image_row_grid(
                samples.split(SAMPLES_PER_CLASS),
                ["G (with glasses)", "NoG (without glasses)"],
                OUTPUT_DIR / f"vqgan_prior_samples_temperature_{temperature}.png",
                title=f"Independent class-conditional prior samples: temperature={temperature}",
                dpi=160,
            )
            progress.update(1)


if __name__ == "__main__":
    evaluate()
