"""Shared 256x256 VAE model, objective terms, and lesson training loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.plot.figures import save_loss_panels
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.metrics import MetricAccumulator

type _VAELossFunction = Callable[
    [Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]

_METRIC_NAMES = ("total", "reconstruction", "kl")


def diagonal_gaussian_kl(mu, std):
    """Return KL[N(mu, std^2) || N(0, 1)] per sample and dimension."""
    if mu.shape != std.shape or mu.ndim != 2:
        raise ValueError("mu and std must have matching [batch, latent] shapes.")
    return 0.5 * (
        mu.square()
        + std.square()
        - 1.0
        - 2.0 * torch.log(std.clamp_min(1e-8))
    )  # (B, z_dim)


def reparameterize(mu, std):
    """Draw a differentiable sample from a diagonal Gaussian posterior."""
    if mu.shape != std.shape:
        raise ValueError("mu and std must have matching shapes.")
    return mu + std * torch.randn_like(mu)


class VAEEncoder(nn.Module):
    """Map 256x256 RGB images to diagonal-Gaussian posterior parameters."""

    # Manual forward (not Sequential) — the network branches into μ and log(σ)
    def __init__(self, z_dim=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)  # (B, 8, 128, 128)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)  # (B, 16, 64, 64)
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)  # (B, 32, 31, 31)
        self.linear1 = nn.Linear(32 * 31 * 31, 1024)  # (B, 1024)
        self.linear2 = nn.Linear(1024, z_dim)  # μ
        self.linear3 = nn.Linear(1024, z_dim)  # log(σ)

    def statistics(self, inputs):
        """Encode an image batch into posterior mean and standard deviation."""
        # inputs: (B, 3, 256, 256)
        hidden = F.relu(self.conv1(inputs))  # (B, 8, 128, 128)
        hidden = F.relu(self.batch2(self.conv2(hidden)))  # (B, 16, 64, 64)
        hidden = F.relu(self.conv3(hidden))  # (B, 32, 31, 31)
        hidden = torch.flatten(hidden, start_dim=1)  # (B, 32 * 31 * 31)
        hidden = F.relu(self.linear1(hidden))  # (B, 1024)
        mu = self.linear2(hidden)  # μ: (B, z_dim)
        # Bound only the exponential's numerical range, not the sampled z.
        log_std = self.linear3(hidden).clamp(-12.0, 12.0)  # log(σ): (B, z_dim)
        std = torch.exp(log_std)  # σ: (B, z_dim)
        return mu, std

class VAEDecoder(nn.Module):
    """Decode latent vectors into 256x256 RGB Gaussian means with fixed scale."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32 * 31 * 31),
            nn.ReLU(inplace=True),
        )
        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=(32, 31, 31),
        )  # (B, 32, 31, 31)
        # padding: symmetric crop from output edges
        # 0 ≤ output_padding < stride:
        # extra rows/cols on right & bottom
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                                output_padding=1),             # (B, 16, 63, 63)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                                padding=1, output_padding=1),  # (B, 8, 128, 128)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,
                                padding=1, output_padding=1))  # (B, 3, 256, 256)

    def forward(self, z):
        # latent vector z: (B, z_dim)
        hidden = self.linear(z)               # (B, 32 * 31 * 31)
        hidden = self.unflatten(hidden)       # (B, 32, 31, 31)
        hidden = self.conv_transpose(hidden)  # (B, 3, 256, 256)
        # Pixel-wise Gaussian mean; the fixed likelihood scale is absorbed by
        # the reconstruction-loss coefficient in the lesson script.
        decoded_images = torch.sigmoid(hidden)
        return decoded_images


class VAE(nn.Module):
    """Standard convolutional VAE for 256x256 RGB images."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.encoder = VAEEncoder(z_dim)
        self.decoder = VAEDecoder(z_dim)

    def reconstruct(self, inputs, *, sample=True):
        """Reconstruct from a posterior sample or deterministically from mu."""
        mu, std = self.encoder.statistics(inputs)
        z = reparameterize(mu, std) if sample else mu
        return mu, std, self.decoder(z)

    def forward(self, inputs):
        return self.reconstruct(inputs, sample=True)


def reconstruction_and_kl(
    images: Tensor,
    reconstructions: Tensor,
    mu: Tensor,
    std: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the shared per-image reconstruction and KL terms."""
    reconstruction_loss = (
        (reconstructions - images).square().flatten(1).sum(dim=1).mean()
    )
    kl_loss = diagonal_gaussian_kl(mu, std).sum(dim=1).mean()
    return reconstruction_loss, kl_loss


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
        mu, std, decoded_images = model(images)
        total_loss, reconstruction_loss, kl_loss = loss_function(
            images, decoded_images, mu, std
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


def train_glasses_vae(
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
            progress_bar.set_description(
                f"Epoch {epoch}/{num_epochs}", refresh=False
            )
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
                reconstruction=f"{metrics['reconstruction']:.3f}",
                kl=f"{metrics['kl']:.3f}",
                refresh=False,
            )
            if epoch % sample_every_epochs == 0:
                _save_epoch_samples(
                    model,
                    fixed_z,
                    training_path / f"epoch_{epoch:03d}.png",
                    columns=sample_grid_columns,
                )

    checkpoint_metadata = {
        "model_name": model_name,
        "model_config": dict(model_config),
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
    "VAEDecoder",
    "VAEEncoder",
    "diagonal_gaussian_kl",
    "reconstruction_and_kl",
    "reparameterize",
    "train_glasses_vae",
]
