"""FSQ tokenizer with fixed scalar levels and an unconditional PixelCNN prior.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Resize to 256x256 RGB and normalize from [0, 1] to [-1, 1].
    Use all 4,500 training images; class labels are ignored.

Outputs:
    output/vae/fsq/model.pth: final tokenizer/prior pair
    output/vae/fsq/tokenizer_latest.pth: stage-1 recovery checkpoint
    output/vae/fsq/prior_latest.pth: stage-2 recovery checkpoint
    output/vae/fsq/tokenizer_loss.png: L1 and MSE curves
    output/vae/fsq/prior_loss.png: token cross-entropy curve
    output/vae/fsq/training/tokenizer_epoch_*.png: original/reconstruction rows
    output/vae/fsq/training/prior_epoch_*.png: unconditional sample grids

Training data -- glasses-256 (fresh run):
Training images:          4,500
Tokenizer/prior batches:     16 / 32
Examples per epoch:      4,500 images / 9,000 original-and-flipped token grids
Tokenizer/prior epochs:     100 / 50
Optimizer updates:       28,200 tokenizer + 14,100 prior = 42,300 total

Default dimensions:
Training/generated image: 256x256 RGB in [-1, 1]
Token grid:               32x32; three 2x downsampling stages
Scalar levels:            (8, 6, 5); 3 latent channels; 240 possible tokens
Quantized scalar values:  [-4,...,3]/4, [-3,...,2]/3, and [-2,...,2]/2
Encoder/decoder width:    128 channels; first/last image stage: 64 channels
PixelCNN:                 256 channels, 15 bottleneck residual blocks
                          128 bottleneck / 1,024 head channels; 240 output logits
                          First kernel 63x63, 32 groups, 32-dimensional embeddings
Optimizer:                Adam, betas (0.9, 0.999); tokenizer 2e-4, prior 5e-4
Previews:                 8 images; epochs 1, every 10, and final; temperature 1.0

Run without arguments; edit the constants below to experiment. The vocabulary
differs from VQ-VAE: equal token counts do not imply equal reconstruction or generation.
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
from dl_utils.vae.discrete_tokenizers import FSQAutoencoder
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

# Paths and data
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "fsq"
IMAGE_SIZE = 256

# Tokenizer configuration
DOWNSAMPLE_STEPS = 3
HIDDEN_CHANNELS = 128
LEVELS = (8, 6, 5)

# Prior configuration
# The first mask covers the complete causal history of a 32x32 token grid.
PRIOR_HIDDEN_CHANNELS = 256
PRIOR_BLOCKS = 15
PRIOR_BOTTLENECK_CHANNELS = 128
PRIOR_HEAD_CHANNELS = 1024
PRIOR_KERNEL_SIZE = 63
PRIOR_FIRST_GROUPS = 32
PRIOR_EMBEDDING_DIM = 32
PRIOR_DROPOUT = 0.1

# Training configuration
RESUME = True
TOKENIZER_EPOCHS = 100
PRIOR_EPOCHS = 50
BATCH_SIZE = 16
PRIOR_BATCH_SIZE = 32
PRIOR_HORIZONTAL_FLIP = True
LR = 2e-4
PRIOR_LR = 5e-4
WORKERS = 4
SEED = 42

# Preview configuration
SAMPLE_EVERY = 10
NUM_SAMPLES = 8
# Sample the learned categorical distribution without temperature sharpening.
TEMPERATURE = 1.0


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
            metrics = MetricAccumulator(("mse", "l1"), device=device)
            progress.set_description(
                f"Stage 1/2: FSQ tokenizer {epoch}/{TOKENIZER_EPOCHS}", refresh=False
            )
            for images, _ in loader:
                images = images.to(device)
                reconstruction, _ = model(images)
                mse = F.mse_loss(reconstruction, images)
                l1 = F.l1_loss(reconstruction, images)
                optimizer.zero_grad(set_to_none=True)
                l1.backward()
                optimizer.step()
                metrics.add_batch_means((mse, l1), num_examples=len(images))
                progress.update(1)
            losses = metrics.compute_weighted_means(require_finite=True)
            progress.set_postfix(l1=f"{losses['l1']:.4f}", refresh=False)
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
            "blocks": PRIOR_BLOCKS,
            "bottleneck": PRIOR_BOTTLENECK_CHANNELS,
            "head_channels": PRIOR_HEAD_CHANNELS,
            "first_kernel_size": PRIOR_KERNEL_SIZE,
            "first_groups": PRIOR_FIRST_GROUPS,
            "embedding_dim": PRIOR_EMBEDDING_DIM,
            "dropout": PRIOR_DROPOUT,
        },
    }
    recipe = {
        "model": config,
        "tokenizer_epochs": TOKENIZER_EPOCHS,
        "prior_epochs": PRIOR_EPOCHS,
        "lr": LR,
        "reconstruction_loss": "l1",
        "prior_lr": PRIOR_LR,
        "batch_size": BATCH_SIZE,
        "prior_batch_size": PRIOR_BATCH_SIZE,
        "prior_horizontal_flip": PRIOR_HORIZONTAL_FLIP,
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


if __name__ == "__main__":
    train()
