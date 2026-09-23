"""Conditional VAE: make the training and generation information paths explicit.

The conditional-prior model factorizes

    p(x, z | c) = p(z | c) p(x | z, c)

and trains the recognition posterior q(z | x, c) with

    summed RGB MSE + KL[q(z | x, c) || p(z | c)].

The target image enters q during training. Generation uses only base Gaussian
noise and the requested label. The learned conditional prior transforms that
noise before decoding. Corresponding columns share base noise across classes
and epochs; changing the label does not guarantee identity preservation.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the existing cache directly, without resizing or making another cache.
    All images are used for training, matching the cGAN data scope.

Outputs:
    output/vae/conditional_vae/conditional_vae.pth: final checkpoint
    output/vae/conditional_vae/conditional_samples.png: final class grid
    output/vae/conditional_vae/training/epoch_*.png: saved after each selected epoch
    output/vae/conditional_vae/cvae_metrics.csv: saved after all training epochs
    output/vae/conditional_vae/cvae_metrics_*.png: final metric curves

Training data -- glasses-256:
With glasses (G=0):        2,543
Without glasses (NoG=1):   1,957
Training images:          4,500 (including repository-tracked label corrections)
Batch size:                  16
Samples per epoch:        4,496 (281 full batches; drop_last=True)
Training epochs:              80
Optimizer updates:        22,480
Four shuffled images are omitted per epoch. Labels match the cGAN.

Default dimensions:
Training input:           256x256 RGB in [0, 1]
Generated image:          256x256 RGB in [0, 1]
Latent vector:                128 values
Condition embedding:           32 values
Encoder channels:         32, 64, 128, 256, 256, 256
Context hidden layer:         512 units before the mean/log-variance heads
Decoder channels:         256, 256, 256, 128, 64, 32, 3 (including input)
Model size:                8.558 M parameters
Optimizer:                Adam, betas (0.9, 0.999)
Learning rate:            2e-4, cosine decay toward 2e-5 over 80 epochs

Run this script without arguments; edit the constants below to experiment.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dl_utils.data.glasses import (
    GLASSES_CLASS_NAMES,
    GLASSES_IMAGE_SIZE,
    glasses_data_config,
    glasses_dataset,
)
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.artifacts import save_training_metrics
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.conditional_vae import (
    ConditionalVAE,
    conditional_vae_loss,
    save_conditional_samples,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "conditional_vae"
CHECKPOINT_NAME = "conditional_vae.pth"
NUM_CLASSES = len(GLASSES_CLASS_NAMES)
SAMPLES_PER_CLASS = 8
SAMPLE_EVERY = 5  # Save after epochs 1, 5, 10, ... and the final epoch.
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4


# Edit these defaults to explore the lesson.
EPOCHS = 80
BATCH_SIZE = 16
IMAGE_SIZE = GLASSES_IMAGE_SIZE
LATENT_DIM = 128
CONDITION_DIM = 32
HIDDEN_CHANNELS = 256
CONTEXT_DIM = 512  # Posterior MLP width; independent of image channels.
LR = 2e-4
MIN_LR = 2e-5
WORKERS = 4
SEED = 42


def make_train_loader(device: torch.device) -> DataLoader:
    train_set = glasses_dataset(DATA_DIR)
    return make_device_aware_loader(
        train_set,
        BATCH_SIZE,
        device,
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
    )


def train_cvae(
    *,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    out_dir = OUTPUT_DIR
    training_dir = out_dir / "training"
    if not out_dir.exists():
        reset_dir(str(out_dir))
    reset_dir(str(training_dir))
    history = []
    model_config = {
        "num_classes": NUM_CLASSES,
        "latent_dim": LATENT_DIM,
        "condition_dim": CONDITION_DIM,
        "hidden_channels": HIDDEN_CHANNELS,
        "posterior_hidden_dim": CONTEXT_DIM,
    }
    model = ConditionalVAE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR,
    )
    fixed_noise = torch.randn(SAMPLES_PER_CLASS, LATENT_DIM, device=device)

    with tqdm(
        total=EPOCHS * len(train_loader),
        desc=f"CVAE 1/{EPOCHS}",
        unit="batch",
        mininterval=PROGRESS_INTERVAL,
    ) as progress:
        for epoch in range(1, EPOCHS + 1):
            progress.set_description(f"CVAE {epoch}/{EPOCHS}", refresh=False)
            model.train()
            metrics = MetricAccumulator(
                ("loss", "distortion", "rate"),
                device=device,
            )
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                reconstruction, statistics = model(images, labels)
                loss, terms = conditional_vae_loss(
                    reconstruction,
                    images,
                    statistics,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.add_batch_means(
                    (
                        loss,
                        terms["distortion"],
                        terms["rate"],
                    ),
                    num_examples=images.shape[0],
                )
                progress.set_postfix(
                    loss=f"{metrics.compute_weighted_means()['loss']:.3f}",
                    refresh=False,
                )
                progress.update(1)

            train_metrics = metrics.compute_weighted_means(require_finite=True)
            train_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            history.append(train_metrics)
            scheduler.step()
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
                save_conditional_samples(
                    model,
                    training_dir / f"epoch_{epoch:03d}.png",
                    device=device,
                    samples_per_class=SAMPLES_PER_CLASS,
                    noise=fixed_noise,
                )

    save_training_metrics(
        history,
        out_dir,
        prefix="cvae",
        max_panels=MAX_METRIC_PANELS,
    )
    save_model_weights(
        model,
        out_dir / CHECKPOINT_NAME,
        metadata={
            "model_name": "conditional_vae",
            "model_config": model_config,
            "data_config": glasses_data_config(),
            "training_config": {
                "epochs": EPOCHS,
                "batch_size": train_loader.batch_size,
                "optimizer": "Adam",
                "learning_rate": LR,
                "scheduler": "CosineAnnealingLR",
                "minimum_learning_rate": MIN_LR,
                "betas": [0.9, 0.999],
                "seed": SEED,
            },
        },
    )
    save_conditional_samples(
        model,
        out_dir / "conditional_samples.png",
        device=device,
        samples_per_class=SAMPLES_PER_CLASS,
        noise=fixed_noise,
    )


def main() -> None:
    set_seed(SEED)
    device = try_gpu()
    train_loader = make_train_loader(device)
    train_cvae(
        train_loader=train_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
