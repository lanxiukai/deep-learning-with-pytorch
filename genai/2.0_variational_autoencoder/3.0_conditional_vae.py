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
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.conditional_vae import (
    ConditionalVAE,
    save_conditional_samples,
)
from dl_utils.vae.training_artifacts import save_training_metrics
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar

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


def conditional_vae_loss(
    reconstruction: Tensor,
    real_images: Tensor,
    statistics: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return negative conditional ELBO with explicit per-sample reductions."""
    # reconstruction: (B, 1, 32, 32)
    # real_images:    (B, 1, 32, 32)
    distortion = (
        F.binary_cross_entropy(reconstruction, real_images, reduction="none")
        .flatten(1)
        .sum(dim=1)
        .mean()
    )  # mean distortion per sample: scalar ()
    rate_per_dimension = diagonal_gaussian_kl_from_logvar(
        statistics["q_mu"],
        statistics["q_logvar"],
        statistics["p_mu"],
        statistics["p_logvar"],
    )  # (B, latent_dim)
    rate = rate_per_dimension.sum(dim=1).mean()  # mean rate per sample: scalar ()
    return distortion + rate, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "num_active_latent_dimensions": (
            rate_per_dimension.mean(dim=0) > ACTIVE_RATE_THRESHOLD
        )
        .sum()
        .detach(),
    }  # loss, terms


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


def _accumulate(
    total_metrics: dict[str, float], metrics: dict[str, Tensor], batch_size: int
) -> None:
    for name, value in metrics.items():
        total_metrics[name] = total_metrics.get(name, 0.0) + value.item() * batch_size


@torch.inference_mode()
def evaluate_cvae(
    model: ConditionalVAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    # Accumulated metric totals across all batches.
    total_metrics: dict[str, float] = {}
    total_examples = 0  # Total number of examples processed.
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction, statistics = model(images, labels)
        loss, terms = conditional_vae_loss(reconstruction, images, statistics)
        _accumulate(total_metrics, {"loss": loss.detach(), **terms}, images.shape[0])
        total_examples += images.shape[0]
    return {name: value / total_examples for name, value in total_metrics.items()}


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
            total_metrics: dict[str, float] = {}
            total_examples = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                reconstruction, statistics = model(images, labels)
                loss, terms = conditional_vae_loss(reconstruction, images, statistics)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                _accumulate(
                    total_metrics,
                    {"loss": loss.detach(), **terms},
                    images.shape[0],
                )
                total_examples += images.shape[0]
                progress.set_postfix(
                    loss=f"{total_metrics['loss'] / total_examples:.3f}",
                    refresh=False,
                )
                progress.update(1)

            train_metrics = {
                name: value / total_examples for name, value in total_metrics.items()
            }
            validation_metrics = evaluate_cvae(model, validation_loader, device)
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
