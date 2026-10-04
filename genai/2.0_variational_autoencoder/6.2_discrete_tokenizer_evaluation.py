"""Compare VAE, VQ-VAE, and FSQ using two eight-image grids.

Run 1.0, 6.0, and 6.1 first. Reconstructions use the same training images;
generation rows are independent unconditional samples, with no matched identities.
"""

from typing import cast

import torch

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights
from dl_utils.vae.discrete_workflow import glasses_loader, load_pair
from dl_utils.vae.token_priors import PixelCNNPrior
from dl_utils.vae.vae import VAE

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = OUTPUT_ROOT / "evaluation" / "discrete_tokenizer"
VAE_CHECKPOINT = OUTPUT_ROOT / "vae" / "vae.pth"
PAIR_CHECKPOINTS = {
    "vq_vae": OUTPUT_ROOT / "vq_vae" / "model.pth",
    "fsq": OUTPUT_ROOT / "fsq" / "model.pth",
}
IMAGE_SIZE = 256
NUM_SAMPLES = 8
TEMPERATURE = 1.0
SEED = 123


@torch.inference_mode()
def evaluate():
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
    generations = [vae_samples.mul(2).sub(1)]
    for name, path in PAIR_CHECKPOINTS.items():
        tokenizer, prior = load_pair(path, name, device, image_size=IMAGE_SIZE)
        assert isinstance(prior, PixelCNNPrior)
        reconstructions.append(tokenizer(originals)[0])
        side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            indices = prior.sample(
                NUM_SAMPLES, side, side, device=device, temperature=TEMPERATURE
            )
            generations.append(tokenizer.decode_indices(indices))
    reset_dir(str(OUTPUT_DIR))
    save_image_row_grid(
        reconstructions,
        ["Original", "VAE (mean)", "VQ-VAE", "FSQ"],
        OUTPUT_DIR / "vae_vq_vae_fsq_reconstructions.png",
        title="Same training images: reconstruction",
        dpi=160,
    )
    save_image_row_grid(
        generations,
        ["VAE", "VQ-VAE", "FSQ"],
        OUTPUT_DIR / "vae_vq_vae_fsq_samples.png",
        title="Independent unconditional prior samples",
        dpi=160,
    )


if __name__ == "__main__":
    evaluate()
