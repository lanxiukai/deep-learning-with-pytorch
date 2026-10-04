"""Train vq_vae's tokenizer, encode images once, then train a PixelCNN prior.

Follow train() for the complete order. RESUME restores the same configuration
at an epoch boundary; previews show eight training images or prior samples.
Outputs: model.pth (the final pair), tokenizer_latest.pth, prior_latest.pth,
two loss figures, and training/*.png. There is no validation or model selection.
"""

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
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS, VQVAE

# Paths and data
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vq_vae"
IMAGE_SIZE = 256

# Tokenizer configuration
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 64
CODEBOOK_SIZE = 512
COMMITMENT = 0.25
EMA_DECAY = 0.99
EMA_EPSILON = 1e-5

# Prior configuration
PRIOR_HIDDEN_CHANNELS = 64
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
TEMPERATURE = 1.0


def train_tokenizer(
    model: VQVAE,
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
        desc="Stage 1/2: VQ-VAE tokenizer",
        unit="batch",
    ) as progress:
        for epoch in range(completed + 1, TOKENIZER_EPOCHS + 1):
            model.train()
            seed_epoch_loader(loader, SEED, epoch)
            metrics = MetricAccumulator(("mse", "commitment_loss"), device=device)
            progress.set_description(
                f"Stage 1/2: VQ-VAE tokenizer {epoch}/{TOKENIZER_EPOCHS}", refresh=False
            )
            for images, _ in loader:
                images = images.to(device)
                reconstruction, _, commitment = model(images)
                mse = F.mse_loss(reconstruction, images)
                loss = mse + commitment
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.add_batch_means((mse, commitment), num_examples=len(images))
                progress.update(1)
            losses = metrics.compute_weighted_means(require_finite=True)
            progress.set_postfix(
                mse=f"{losses['mse']:.4f}",
                commitment=f"{losses['commitment_loss']:.4f}",
                refresh=False,
            )
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
            "embedding_dim": EMBEDDING_DIM,
            "codebook_size": CODEBOOK_SIZE,
            "commitment": COMMITMENT,
            "ema_decay": EMA_DECAY,
            "ema_epsilon": EMA_EPSILON,
            "hidden_channels": HIDDEN_CHANNELS,
            "downsample_steps": DOWNSAMPLE_STEPS,
        },
        "prior": {
            "vocabulary_size": CODEBOOK_SIZE,
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
    tokenizer = VQVAE(**config["tokenizer"]).to(device)
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
    save_pair(OUTPUT_DIR / "model.pth", "vq_vae", tokenizer, prior, config)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
