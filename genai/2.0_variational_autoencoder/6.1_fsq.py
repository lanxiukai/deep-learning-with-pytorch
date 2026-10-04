"""Train fsq's tokenizer, encode images once, then train a PixelCNN prior.

Follow train() for the complete order. RESUME restores the same configuration
at an epoch boundary; previews show eight training images or prior samples.
Outputs: model.pth (the final pair), tokenizer_latest.pth, prior_latest.pth,
two loss figures, and training/*.png. There is no validation or model selection.
"""

import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_workflow import (
    epoch_checkpoint,
    fixed_images,
    glasses_loader,
    save_loss_curves,
    save_pair,
    save_reconstruction,
    seed_epoch_loader,
)
from dl_utils.vae.pixelcnn_training import train_pixelcnn_prior
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS, FSQAutoencoder

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "fsq"
IMAGE_SIZE = 256
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS
RESUME = True
TOKENIZER_EPOCHS = 100
PRIOR_EPOCHS = 100
HIDDEN_CHANNELS = 128
LEVELS = (8, 8, 8)
PRIOR_HIDDEN_CHANNELS = 64
PRIOR_LAYERS = 16
BATCH_SIZE = 16
LR = 2e-4
PRIOR_LR = 2e-4
WORKERS = 4
SEED = 42
SAMPLE_EVERY = 10
NUM_SAMPLES = 8
TEMPERATURE = 1.0


def train_tokenizer(model, loader, device, recipe):
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
    for epoch in range(completed + 1, TOKENIZER_EPOCHS + 1):
        model.train()
        seed_epoch_loader(loader, SEED, epoch)
        metrics = MetricAccumulator(("mse",), device=device)
        for images, _ in tqdm(loader, desc=f"fsq {epoch}/{TOKENIZER_EPOCHS}"):
            images = images.to(device)
            reconstruction, _ = model(images)
            mse = F.mse_loss(reconstruction, images)
            loss = mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics.add_batch_means((mse,), num_examples=len(images))
        losses = metrics.compute_weighted_means(require_finite=True)
        print(f"Epoch {epoch}: {losses}")
        state["history"].append(losses)
        checkpoint.save(epoch, state)
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
            save_reconstruction(
                model,
                originals,
                OUTPUT_DIR / "training" / f"tokenizer_epoch_{epoch:03d}.png",
                device,
            )
    save_loss_curves(state["history"], OUTPUT_DIR / "tokenizer_loss.png")


def train():
    set_seed(SEED)
    device = try_gpu()
    (OUTPUT_DIR / "training").mkdir(parents=True, exist_ok=True)
    if not RESUME:
        (OUTPUT_DIR / "tokenizer_latest.pth").unlink(missing_ok=True)
        (OUTPUT_DIR / "prior_latest.pth").unlink(missing_ok=True)
    loader = glasses_loader(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE, device, shuffle=True, num_workers=WORKERS
    )
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


def main():
    train()


if __name__ == "__main__":
    main()
