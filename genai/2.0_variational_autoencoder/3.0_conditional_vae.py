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
    output/vae/conditional_vae/<variant>/model.pth: final variant checkpoint
    output/vae/conditional_vae/<variant>/conditional_samples.png: class grid

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

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.conditional_vae import (
    ConditionalVAE,
)
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar

PROJECT_ROOT = infer_project_root()


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
    target: Tensor,
    statistics: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return negative conditional ELBO with explicit per-sample reductions."""
    distortion = (
        F.binary_cross_entropy(reconstruction, target, reduction="none")
        .flatten(1)
        .sum(dim=1)
        .mean()
    )
    rate_per_dimension = diagonal_gaussian_kl_from_logvar(
        statistics["q_mu"],
        statistics["q_logvar"],
        statistics["p_mu"],
        statistics["p_logvar"],
    )
    rate = rate_per_dimension.sum(dim=1).mean()
    return distortion + rate, {
        "distortion": distortion.detach(),
        "rate": rate.detach(),
        "active_units": (rate_per_dimension.mean(dim=0) > 0.05).sum().detach(),
    }


def make_loaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transform,
    )
    validation_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
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
    totals: dict[str, float], metrics: dict[str, Tensor], batch_size: int
) -> None:
    for name, value in metrics.items():
        totals[name] = totals.get(name, 0.0) + float(value) * batch_size


@torch.inference_mode()
def evaluate_cvae(
    model: ConditionalVAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reconstruction, statistics = model(images, labels)
        loss, terms = conditional_vae_loss(reconstruction, images, statistics)
        _accumulate(totals, {"loss": loss.detach(), **terms}, images.shape[0])
        examples += images.shape[0]
    return {name: value / examples for name, value in totals.items()}


def _save_conditional_samples(
    model: ConditionalVAE,
    path: Path,
    *,
    device: torch.device,
    samples_per_class: int = 8,
) -> None:
    labels = torch.arange(10, device=device).repeat_interleave(samples_per_class)
    model.eval()
    with torch.inference_mode():
        images = model.generate(labels)
    save_image(images, path, nrow=samples_per_class)


def train_cvae(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    variant = "standard-prior"
    out_dir = PROJECT_ROOT / "output" / "vae" / "conditional_vae" / variant
    reset_dir(str(out_dir))
    model_config = {
        "num_classes": 10,
        "latent_dim": LATENT_DIM,
        "condition_dim": CONDITION_DIM,
        "hidden_channels": HIDDEN_CHANNELS,
    }
    model = ConditionalVAE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        totals: dict[str, float] = {}
        examples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            reconstruction, statistics = model(images, labels)
            loss, terms = conditional_vae_loss(reconstruction, images, statistics)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _accumulate(
                totals,
                {"loss": loss.detach(), **terms},
                images.shape[0],
            )
            examples += images.shape[0]

        train_metrics = {name: value / examples for name, value in totals.items()}
        validation_metrics = evaluate_cvae(model, validation_loader, device)
        print(
            f"{variant} epoch {epoch:03d}: "
            f"train loss={train_metrics['loss']:.3f}, "
            f"D={train_metrics['distortion']:.3f}, "
            f"R={train_metrics['rate']:.3f}; "
            f"validation loss={validation_metrics['loss']:.3f}, "
            f"D={validation_metrics['distortion']:.3f}, "
            f"R={validation_metrics['rate']:.3f}"
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "conditional_vae",
            "model_config": model_config,
        },
        out_dir / "model.pth",
    )
    _save_conditional_samples(model, out_dir / "conditional_samples.png", device=device)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(device)
    train_cvae(
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
