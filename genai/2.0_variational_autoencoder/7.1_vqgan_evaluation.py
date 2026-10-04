"""Load the final VQGAN pair and save reconstruction and conditional sample grids.

Run 7.0 first. Eight training images are reconstructed. Four independent G
samples and four NoG samples are labeled by row; columns do not match identities.
"""

import torch

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.discrete_workflow import glasses_loader, load_pair
from dl_utils.vae.token_priors import CausalTransformerPrior

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "vqgan" / "model.pth"
OUTPUT_DIR = CHECKPOINT.parent / "evaluation"
IMAGE_SIZE = 256
RECONSTRUCTION_SAMPLES = 8
SAMPLES_PER_CLASS = 4
TEMPERATURE = 1.0
SEED = 123


@torch.inference_mode()
def evaluate():
    set_seed(SEED)
    device = try_gpu()
    tokenizer, prior = load_pair(CHECKPOINT, "vqgan", device, image_size=IMAGE_SIZE)
    assert isinstance(prior, CausalTransformerPrior)
    loader = glasses_loader(
        DATA_DIR, IMAGE_SIZE, RECONSTRUCTION_SAMPLES, device, conditional=True
    )
    originals = next(iter(loader))[0].to(device)
    reconstruction = tokenizer(originals)[0]
    labels = torch.arange(len(GLASSES_CLASS_NAMES), device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
    indices = prior.sample(
        len(labels), device=device, labels=labels, temperature=TEMPERATURE
    )
    samples = tokenizer.decode_indices(indices.reshape(len(labels), side, side))
    reset_dir(str(OUTPUT_DIR))
    save_image_row_grid(
        [originals, reconstruction],
        ["Original", "VQGAN"],
        OUTPUT_DIR / "vqgan_real_and_reconstruction.png",
        title="Same training images: reconstruction",
        dpi=160,
    )
    save_image_row_grid(
        samples.split(SAMPLES_PER_CLASS),
        ["G (with glasses)", "NoG (without glasses)"],
        OUTPUT_DIR / "vqgan_prior_samples.png",
        title="Independent class-conditional prior samples",
        dpi=160,
    )


if __name__ == "__main__":
    evaluate()
