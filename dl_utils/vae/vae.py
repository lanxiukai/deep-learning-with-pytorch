"""Shared 256x256 VAE model, objective terms, and lesson training loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.plot.figures import save_loss_panels
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.metrics import MetricAccumulator, save_metrics_csv
from dl_utils.vae.vae_common import (
    ImageDecoder,
    ImageEncoder,
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
    split_gaussian_parameters,
)

type _VAELossFunction = Callable[
    [Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]

_METRIC_NAMES = ("total", "reconstruction", "kl")


class VAEEncoder(nn.Module):
    """Shared ImageEncoder -> q(z | x) mean and log-variance."""

    def __init__(self, z_dim: int = 100, hidden_channels: int = 256) -> None:
        super().__init__()
        self.net = ImageEncoder(hidden_channels)
        self.posterior = nn.Linear(hidden_channels * 4 * 4, 2 * z_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Encode an image batch into posterior mean and log-variance."""
        return split_gaussian_parameters(self.posterior(self.net(inputs)))


class VAE(nn.Module):
    """Standard/beta-VAE using the shared 256x256 RGB encoder and decoder."""

    backbone = "shared_rgb"

    def __init__(self, z_dim: int = 100, hidden_channels: int = 256) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.hidden_channels = hidden_channels
        self.encoder = VAEEncoder(z_dim, hidden_channels)
        self.decoder = ImageDecoder(z_dim, hidden_channels)

    def encode(self, inputs):
        """Return posterior mean and log-variance for an image batch."""
        return self.encoder(inputs)

    def reconstruct(self, inputs, *, sample=True):
        """Reconstruct from a posterior sample or deterministically from mu."""
        mu, logvar = self.encode(inputs)
        z = reparameterize_logvar(mu, logvar) if sample else mu
        return mu, logvar, self.decoder(z)

    def forward(self, inputs):
        return self.reconstruct(inputs, sample=True)


def reconstruction_and_kl(
    images: Tensor,
    reconstructions: Tensor,
    mu: Tensor,
    logvar: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the shared per-image reconstruction and KL terms."""
    reconstruction_loss = (
        (reconstructions - images).square().flatten(1).sum(dim=1).mean()
    )
    kl_loss = diagonal_gaussian_kl_from_logvar(mu, logvar).sum(dim=1).mean()
    return reconstruction_loss, kl_loss


def weighted_vae_loss(
    images: Tensor,
    reconstructions: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return reconstruction loss plus a beta-weighted KL term."""
    reconstruction_loss, kl_loss = reconstruction_and_kl(
        images, reconstructions, mu, logvar
    )
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


@torch.inference_mode()
def _save_epoch_samples(
    model: VAE,
    fixed_z: Tensor,
    output_path: str | PathLike[str],
    *,
    columns: int,
) -> None:
    """Decode one fixed prior batch for comparable epoch snapshots."""
    was_training = model.training
    model.eval()
    samples = model.decoder(fixed_z).cpu()
    save_image(samples, Path(output_path), nrow=columns)
    model.train(was_training)


def _train_epoch(
    model: VAE,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress_bar: tqdm,
    loss_function: _VAELossFunction,
) -> dict[str, float]:
    """Train one FP32 epoch with a lesson-provided VAE objective."""
    model.train()
    metrics = MetricAccumulator(_METRIC_NAMES, device=device)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        mu, logvar, decoded_images = model(images)
        total_loss, reconstruction_loss, kl_loss = loss_function(
            images, decoded_images, mu, logvar
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        metrics.update(
            (total_loss, reconstruction_loss, kl_loss),
            num_examples=images.shape[0],
        )
        progress_bar.update(1)
    return metrics.compute_finite()


def train_vae(
    *,
    loss_function: _VAELossFunction,
    data_dir: str | PathLike[str],
    out_dir: str | PathLike[str],
    checkpoint_name: str,
    model_name: str,
    model_config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    num_epochs: int = 100,
    batch_size: int = 16,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    num_fixed_samples: int = 18,
    sample_grid_columns: int = 6,
    sample_every_epochs: int = 10,
    seed: int = 42,
) -> None:
    """Run the common data, optimization, sampling, and saving workflow."""
    data_path = Path(data_dir)
    output_path = Path(out_dir)
    training_path = output_path / "training"
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. "
            "Run tool_scripts/download_dataset.py first."
        )
    if num_epochs < 1 or sample_every_epochs < 1:
        raise ValueError("epoch counts must be positive")

    if not output_path.exists():
        reset_dir(str(output_path))
    reset_dir(str(training_path))
    set_seed(seed)
    device = try_gpu()
    loader = image_folder_loader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = VAE(**dict(model_config)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    z_dim = model_config.get("z_dim")
    if isinstance(z_dim, bool) or not isinstance(z_dim, int) or z_dim < 1:
        raise ValueError("model_config must contain a positive integer z_dim")
    fixed_z = torch.randn(num_fixed_samples, z_dim, device=device)
    loss_history = {
        "epoch": [],
        "total": [],
        "reconstruction": [],
        "kl": [],
    }

    _save_epoch_samples(
        model,
        fixed_z,
        training_path / "epoch_000.png",
        columns=sample_grid_columns,
    )
    with tqdm(
        total=num_epochs * len(loader),
        desc=f"Epoch 1/{num_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs}", refresh=False)
            metrics = _train_epoch(
                model,
                loader,
                optimizer,
                device,
                progress_bar,
                loss_function,
            )
            loss_history["epoch"].append(epoch)
            for name in _METRIC_NAMES:
                loss_history[name].append(metrics[name])
            progress_bar.set_postfix(
                total=f"{metrics['total']:.3f}",
                refresh=False,
            )
            if epoch % sample_every_epochs == 0:
                _save_epoch_samples(
                    model,
                    fixed_z,
                    training_path / f"epoch_{epoch:03d}.png",
                    columns=sample_grid_columns,
                )

    save_metrics_csv(loss_history, output_path / f"{model_name}_metrics.csv")
    checkpoint_metadata = {
        "model_name": model_name,
        "model_config": {"z_dim": model.z_dim, "hidden_channels": model.hidden_channels},
        "backbone": model.backbone,
        "dataset": "glasses-256",
        "value_range": [0.0, 1.0],
        "seed": seed,
        **dict(metadata or {}),
    }
    save_model_weights(
        model,
        output_path / checkpoint_name,
        metadata=checkpoint_metadata,
    )
    save_loss_panels(
        loss_history["epoch"],
        {
            "Training objective": {
                "Total loss": loss_history["total"],
            },
            "Reconstruction objective": {
                "Summed-pixel MSE per image": loss_history["reconstruction"],
            },
            "Prior regularization": {
                "Unweighted KL nats per image": loss_history["kl"],
            },
        },
        output_path / "loss_curves.png",
    )


__all__ = [
    "VAE",
    "VAEEncoder",
    "reconstruction_and_kl",
    "train_vae",
    "weighted_vae_loss",
]
