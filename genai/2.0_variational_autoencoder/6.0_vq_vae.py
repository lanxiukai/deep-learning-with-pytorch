"""VQ-VAE: train a 256px tokenizer, then an unconditional PixelCNN.

Use the same 4,500 glasses-256 faces as the standard Gaussian VAE.
Both stages ignore folder labels. VQ-VAE and FSQ share four downsampling
steps, a 16x16 token grid, 512 possible codes, MSE reconstruction, and the
same prior budget.
The codebook uses EMA updates; quantizer loss is commitment loss.

Edit the constants below. RESUME continues an epoch-boundary checkpoint;
set TRAIN_TOKENIZER=False to train only the prior from selected weights.
A fixed 64-image training subset selects snapshots on the first epoch,
every five epochs, and the final epoch;
the stored val_* fields are training-set diagnostics, not held-out results.
A prior is bound to its exact tokenizer snapshot.
Former gated PixelCNN weights require prior retraining: set
TRAIN_TOKENIZER=False and RESUME=False to reuse the selected tokenizer.
Run 6.2 for simple VAE comparisons.

Outputs under output/vae/vq_vae/:
    tokenizer/{latest,best}.pth, metrics.csv, and loss_curves.png
    prior/{latest,best}.pth, metrics.csv, and loss_curves.png
    vq_vae.pth and pixelcnn_prior.pth: selected, paired evaluation weights
    token_cache_train.pth: shared training/monitoring tokens
    training/*.png: eight fixed reconstructions or prior samples
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_workflow import (
    TokenizerStage,
    evaluate_mse_tokenizer,
    glasses_loader,
    image_contract,
    load_tokenizer_weights,
    save_stage_loss_curves,
    save_tokenizer_preview,
    seed_epoch_loader,
)
from dl_utils.vae.pixelcnn_training import train_pixelcnn_prior
from dl_utils.vae.quantization import (
    TOKENIZER_DOWNSAMPLE_STEPS,
    VQVAE,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vq_vae"
TOKENIZER_CHECKPOINT_NAME = "vq_vae.pth"
PRIOR_CHECKPOINT_NAME = "pixelcnn_prior.pth"
IMAGE_SIZE = 256
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS

# Edit these defaults to explore the lesson.
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
TRAIN_TOKENIZER = True
RESUME = True  # Continue from the latest epoch when a checkpoint exists.
TOKENIZER_EPOCHS = 100
PRIOR_EPOCHS = 100
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 64
CODEBOOK_SIZE = 512
COMMITMENT = 0.25
EMA_DECAY = 0.99
EMA_EPSILON = 1e-5
PRIOR_HIDDEN_CHANNELS = 64
PRIOR_LAYERS = 16  # Basic single-stream A/B masked convolutions with ReLU.
BATCH_SIZE = 16
LR = 2e-4
PRIOR_LR = 2e-4
TEMPERATURE = 1.0
SAMPLE_EVERY = 10
LOG_EVERY = 100
MONITOR_EXAMPLES = 64
MONITOR_EVERY = 5
MONITOR_SEED = 123
WORKERS = 4
SEED = 42
RECONSTRUCTION_SAMPLES = 8
NUM_FIXED_SAMPLES = 8
SAMPLE_GRID_COLUMNS = 4
PROGRESS_INTERVAL = 0.5


def train_tokenizer(train_loader, monitor_loader, device, out_dir, monitor_protocol):
    config = {
        "hidden_channels": HIDDEN_CHANNELS,
        "embedding_dim": EMBEDDING_DIM,
        "codebook_size": CODEBOOK_SIZE,
        "commitment": COMMITMENT,
        "ema_decay": EMA_DECAY,
        "ema_epsilon": EMA_EPSILON,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = VQVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    stage = TokenizerStage(
        out_dir / "tokenizer",
        models={"model": model},
        optimizers={"model": optimizer},
        metadata={
            **image_contract(IMAGE_SIZE),
            "model_name": "vq_vae_tokenizer",
            "model_config": config,
            "monitor_protocol": monitor_protocol,
            "selection_metric": "training_subset_mse",
        },
        recipe={"lr": LR, "batch_size": BATCH_SIZE, "seed": SEED},
        resume=RESUME,
        metric_names=("mse", "commitment_loss", "val_mse"),
    )
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for epoch in stage.epochs(TOKENIZER_EPOCHS):
        if stage.needs_training(epoch):
            model.train()
            seed_epoch_loader(train_loader, SEED, epoch)
            metrics = MetricAccumulator(("mse", "commitment_loss"), device=device)
            with tqdm(
                train_loader,
                desc=f"vq_vae {epoch}/{TOKENIZER_EPOCHS}",
                unit="batch",
                mininterval=PROGRESS_INTERVAL,
            ) as progress:
                for batch_index, (images, _) in enumerate(progress, 1):
                    images = images.to(device, non_blocking=True)
                    reconstruction, _, quantizer_loss = model(images)
                    distortion = F.mse_loss(reconstruction, images)
                    loss = distortion + quantizer_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    metrics.add_batch_means(
                        (
                            distortion,
                            quantizer_loss.detach(),
                        ),
                        num_examples=images.shape[0],
                    )
                    if batch_index % LOG_EVERY == 0:
                        progress.set_postfix(
                            metrics.compute_weighted_means(require_finite=True),
                            refresh=False,
                        )
            means = metrics.compute_weighted_means(require_finite=True)
            stage.record_training(epoch, means)
        # latest.pth is already durable if monitoring or plotting fails.
        if stage.should_monitor(epoch, TOKENIZER_EPOCHS, MONITOR_EVERY):
            validation = evaluate_mse_tokenizer(model, monitor_loader, device=device)
            stage.finish_epoch(epoch, validation, score=validation["mse"])
        else:
            stage.finish_epoch(epoch)
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
            save_tokenizer_preview(
                model,
                monitor_loader,
                training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                count=RECONSTRUCTION_SAMPLES,
                device=device,
            )
    payload = stage.export_best(out_dir / TOKENIZER_CHECKPOINT_NAME)
    save_stage_loss_curves(stage)
    return model.eval().requires_grad_(False), payload


def train_prior(
    tokenizer,
    tokenizer_payload,
    train_loader,
    monitor_loader,
    device,
    out_dir,
    monitor_protocol,
):
    train_pixelcnn_prior(
        tokenizer,
        tokenizer_payload,
        train_loader,
        monitor_loader,
        device,
        out_dir,
        monitor_protocol,
        model_name="vq_vae_pixelcnn_prior",
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        hidden_channels=PRIOR_HIDDEN_CHANNELS,
        layers=PRIOR_LAYERS,
        lr=PRIOR_LR,
        epochs=PRIOR_EPOCHS,
        resume=RESUME,
        seed=SEED,
        sample_every=SAMPLE_EVERY,
        monitor_every=MONITOR_EVERY,
        log_every=LOG_EVERY,
        sample_count=NUM_FIXED_SAMPLES,
        sample_columns=SAMPLE_GRID_COLUMNS,
        temperature=TEMPERATURE,
        checkpoint_name=PRIOR_CHECKPOINT_NAME,
        progress_interval=PROGRESS_INTERVAL,
    )


def train() -> None:
    device = try_gpu()
    train_loader, _ = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        shuffle=True,
        seed=SEED,
        num_workers=WORKERS,
    )
    monitor_loader, protocol = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        max_examples=MONITOR_EXAMPLES,
        seed=MONITOR_SEED,
        num_workers=WORKERS,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_TOKENIZER:
        tokenizer, payload = train_tokenizer(
            train_loader, monitor_loader, device, OUTPUT_DIR, protocol
        )
    else:
        tokenizer, payload = load_tokenizer_weights(
            OUTPUT_DIR / TOKENIZER_CHECKPOINT_NAME,
            VQVAE,
            name="vq_vae_tokenizer",
            image_size=IMAGE_SIZE,
            device=device,
            downsample_steps=DOWNSAMPLE_STEPS,
        )
    train_prior(
        tokenizer,
        payload,
        train_loader,
        monitor_loader,
        device,
        OUTPUT_DIR,
        protocol,
    )


def main() -> None:
    set_seed(SEED)
    train()


if __name__ == "__main__":
    main()
