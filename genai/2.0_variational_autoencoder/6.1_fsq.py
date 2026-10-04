"""FSQ tokenizer with fixed scalar levels and an unconditional PixelCNN prior.

Reconstruction and generation flow (k is a raster-ordered token grid):
    encoder(x)       -> z_e -> bounded, rounded scalar values z_q
    pack(z_q)        -> mixed-radix indices k
    decoder(z_q)     -> reconstruction
    PixelCNN(k_<t)   -> logits for k_t -> sample k_t
    unpack(k)        -> z_q -> decoder -> generated image
model(x) returns (reconstruction, indices). Three eight-level scalars give
8 * 8 * 8 = 512 possible tokens per spatial position. The scalar grid is fixed;
there is no learned codebook, EMA update, or commitment loss. Rounding uses
straight-through gradients. See dl_utils/vae/quantization.py and token_priors.py.

Stage 1 minimizes mean RGB MSE. Stage 2 freezes the tokenizer, encodes every
image once into memory, and fits p(k) = product_t p(k_t | k_<t) with mean token
cross-entropy. The PixelCNN matches 6.0 and has the same receptive-field blind
spot. Sampling visits all 256 positions sequentially, then decodes the grid.
The image blocks match VQ-VAE, but the three-channel bottleneck changes capacity.

RESUME=True restores the same recipe at an epoch boundary, including optimizer
state and loss history. A prior checkpoint also restores its frozen tokenizer.
The final checkpoint stores the tokenizer/prior pair and model configuration.
There is no validation split or model selection.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Resize to 256x256 RGB and normalize from [0, 1] to [-1, 1].
    Use all 4,500 training images; class labels are ignored.

Outputs:
    output/vae/fsq/model.pth: final tokenizer/prior pair
    output/vae/fsq/tokenizer_latest.pth: stage-1 recovery checkpoint
    output/vae/fsq/prior_latest.pth: stage-2 recovery checkpoint
    output/vae/fsq/tokenizer_loss.png: MSE curve
    output/vae/fsq/prior_loss.png: token cross-entropy curve
    output/vae/fsq/training/tokenizer_epoch_*.png: original/reconstruction rows
    output/vae/fsq/training/prior_epoch_*.png: unconditional sample grids

Training data -- glasses-256 (fresh run):
Training images:          4,500
Batch size:                  16
Samples per epoch:        4,500 (281 full batches + 4 images; drop_last=False)
Tokenizer/prior epochs:     100 / 100
Optimizer updates:       28,200 per stage; 56,400 total

Default dimensions:
Training/generated image: 256x256 RGB in [-1, 1]
Token grid:               16x16; four 2x downsampling stages
Scalar levels:            (8, 8, 8); 3 latent channels; 512 possible tokens
Quantized scalar values:  [-4, -3, ..., 3] / 4 for each eight-level channel
Encoder/decoder width:    128 channels; first/last image stage: 64 channels
PixelCNN:                 128 channels, 16 masked layers, 512 output logits
Model size:               tokenizer 2.126 M / prior 3.165 M parameters
Optimizer:                Adam, betas (0.9, 0.999), constant 2e-4 for both stages
Previews:                 8 images; epochs 1, every 10, and final; temperature 0.6

Run without arguments; edit the constants below to experiment. Matching the
VQ-VAE vocabulary and prior does not guarantee equal token use or sample quality.
"""

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_workflow import (
    epoch_checkpoint,
    fixed_images,
    glasses_loader,
    prepare_training_output,
    save_loss_curves,
    save_pair,
    save_reconstruction,
    seed_epoch_loader,
)
from dl_utils.vae.pixelcnn_training import train_pixelcnn_prior
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS, FSQAutoencoder

# Paths and data
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "fsq"
IMAGE_SIZE = 256

# Tokenizer configuration
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS
HIDDEN_CHANNELS = 128
LEVELS = (8, 8, 8)

# Prior configuration
# A wider prior improves sample coherence within the same epoch budget.
PRIOR_HIDDEN_CHANNELS = 128
PRIOR_LAYERS = 16

# Training configuration
RESUME = True
TOKENIZER_EPOCHS = 100
PRIOR_EPOCHS = 100
BATCH_SIZE = 16
LR = 2e-4
PRIOR_LR = 2e-4
WORKERS = 4
SEED = 42

# Preview configuration
SAMPLE_EVERY = 10
NUM_SAMPLES = 8
# Trade some diversity for cleaner teaching previews.
TEMPERATURE = 0.6


def train_tokenizer(
    model: FSQAutoencoder,
    loader: DataLoader,
    device: torch.device,
    recipe: Mapping[str, Any],
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoint = epoch_checkpoint(
        OUTPUT_DIR / "tokenizer_latest.pth",
        {"tokenizer": model},
        {"tokenizer": optimizer},
        recipe,
    )
    completed, state = checkpoint.resume(
        checkpoint.path if RESUME and checkpoint.path.is_file() else None,
        initial_state={"history": []},
    )
    originals = fixed_images(loader, NUM_SAMPLES)
    with tqdm(
        total=TOKENIZER_EPOCHS * len(loader),
        initial=completed * len(loader),
        desc="Stage 1/2: FSQ tokenizer",
        unit="batch",
    ) as progress:
        for epoch in range(completed + 1, TOKENIZER_EPOCHS + 1):
            model.train()
            seed_epoch_loader(loader, SEED, epoch)
            metrics = MetricAccumulator(("mse",), device=device)
            progress.set_description(
                f"Stage 1/2: FSQ tokenizer {epoch}/{TOKENIZER_EPOCHS}", refresh=False
            )
            for images, _ in loader:
                images = images.to(device)
                reconstruction, _ = model(images)
                mse = F.mse_loss(reconstruction, images)
                loss = mse
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.add_batch_means((mse,), num_examples=len(images))
                progress.update(1)
            losses = metrics.compute_weighted_means(require_finite=True)
            progress.set_postfix(mse=f"{losses['mse']:.4f}", refresh=False)
            state["history"].append(losses)
            checkpoint.save(epoch, state)
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
                training_dir = OUTPUT_DIR / "training"
                save_reconstruction(
                    model,
                    originals,
                    training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                    device,
                )
    save_loss_curves(state["history"], OUTPUT_DIR / "tokenizer_loss.png")


def train() -> None:
    config = {
        "image_size": IMAGE_SIZE,
        "tokenizer": {
            "levels": LEVELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "downsample_steps": DOWNSAMPLE_STEPS,
        },
        "prior": {
            "vocabulary_size": math.prod(LEVELS),
            "hidden_channels": PRIOR_HIDDEN_CHANNELS,
            "layers": PRIOR_LAYERS,
        },
    }
    recipe = {
        "model": config,
        "tokenizer_epochs": TOKENIZER_EPOCHS,
        "prior_epochs": PRIOR_EPOCHS,
        "lr": LR,
        "prior_lr": PRIOR_LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }
    prepare_training_output(OUTPUT_DIR, resume=RESUME, recipe=recipe)
    set_seed(SEED)
    device = try_gpu()
    loader = glasses_loader(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE, device, shuffle=True, num_workers=WORKERS
    )
    tokenizer = FSQAutoencoder(**config["tokenizer"]).to(device)
    if not (RESUME and (OUTPUT_DIR / "prior_latest.pth").is_file()):
        train_tokenizer(tokenizer, loader, device, recipe)
    prior = train_pixelcnn_prior(
        tokenizer,
        loader,
        device,
        recipe,
        OUTPUT_DIR,
        resume=RESUME,
        sample_every=SAMPLE_EVERY,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
    )
    save_pair(OUTPUT_DIR / "model.pth", "fsq", tokenizer, prior, config)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
