"""Conditional VAE: make the training and generation information paths explicit.

The main standard-prior model factorizes

    p(x, z | c) = p(z) p(x | z, c)

and trains the recognition posterior q(z | x, c) with

    distortion + KL[q(z | x, c) || p(z)].

The target image enters q during training. Generation samples z from N(0, I)
and passes only z and the requested label to the decoder. This lesson uses
one standard normal prior and saves final weights for the evaluation script.

Data:
    data/mnist, downloaded automatically by torchvision when absent.

Outputs:
    output/vae/conditional_vae/conditional_vae.pth: final checkpoint
    output/vae/conditional_vae/conditional_samples.png: final class grid
    output/vae/conditional_vae/training/epoch_*.png: saved after each selected epoch's validation
    output/vae/conditional_vae/cvae_metrics.csv: saved after all training epochs
    output/vae/conditional_vae/cvae_metrics_*.png: final metric curves

Training data -- MNIST:
Training images:          60,000
Validation images:        10,000
Batch size:                  128
Samples per epoch:        59,904 (468 full batches; drop_last=True)
Training epochs:              15
Optimizer updates:         7,020
Default variant:          standard-prior
Note: Digit labels condition every variant. The final 96 shuffled training
images are omitted per epoch.

Default dimensions:
Training input:           32x32 grayscale
Generated image:          32x32 grayscale
Latent vector:                 16 values

Model size:
Standard-prior CVAE:       0.971 M parameters (default)

Run this script without arguments; edit the constants below to experiment.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.conditional_vae import (
    ConditionalVAE,
    conditional_vae_loss,
    evaluate_cvae,
    save_conditional_samples,
)
from dl_utils.vae.training_artifacts import save_training_metrics

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "conditional_vae"
CHECKPOINT_NAME = "conditional_vae.pth"
NUM_CLASSES = 10
IMAGE_SIZE = 32
SAMPLES_PER_CLASS = 8
SAMPLE_EVERY = 5  # Save after epochs 1, 5, 10, ... and the final epoch.
ACTIVE_RATE_THRESHOLD = 0.05
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4


# Edit these defaults to explore the lesson.
EPOCHS = 15
BATCH_SIZE = 128
LATENT_DIM = 16
CONDITION_DIM = 32
HIDDEN_CHANNELS = 128
LR = 2e-4
WORKERS = 4
SEED = 42


def make_loaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    train_set = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    validation_set = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": WORKERS > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(validation_set, shuffle=False, drop_last=False, **common),
    )


def train_cvae(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
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
    }
    model = ConditionalVAE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
                ("loss", "distortion", "rate", "num_active_latent_dimensions"),
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
                    active_rate_threshold=ACTIVE_RATE_THRESHOLD,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.update(
                    (
                        loss,
                        terms["distortion"],
                        terms["rate"],
                        terms["num_active_latent_dimensions"],
                    ),
                    num_examples=images.shape[0],
                )
                progress.set_postfix(
                    loss=f"{metrics.compute()['loss']:.3f}",
                    refresh=False,
                )
                progress.update(1)

            train_metrics = metrics.compute()
            validation_metrics = evaluate_cvae(
                model,
                validation_loader,
                device=device,
                active_rate_threshold=ACTIVE_RATE_THRESHOLD,
            )
            history.append(
                {
                    **{f"train_{name}": value for name, value in train_metrics.items()},
                    **{f"val_{name}": value for name, value in validation_metrics.items()},
                }
            )
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
                save_conditional_samples(
                    model, training_dir / f"epoch_{epoch:03d}.png", device=device
                )

    save_training_metrics(
        history,
        out_dir,
        prefix="cvae",
        max_panels=MAX_METRIC_PANELS,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "conditional_vae",
            "model_config": model_config,
        },
        out_dir / CHECKPOINT_NAME,
    )
    save_conditional_samples(model, out_dir / "conditional_samples.png", device=device)


def main() -> None:
    set_seed(SEED)
    device = try_gpu()
    train_loader, validation_loader = make_loaders(device)
    train_cvae(
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
