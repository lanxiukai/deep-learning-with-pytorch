"""Lightweight shared training loop for directly comparable hierarchy lessons."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.glasses import glasses_data_config
from dl_utils.filesystem.directories import reset_dir
from dl_utils.gan.inference import generate_in_batches
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.hierarchical_vae import (
    HIERARCHY_OBJECTIVE,
    HierarchicalVAE,
    hierarchical_vae_loss,
    model_config,
)
from dl_utils.vae.training_artifacts import save_training_metrics


def warmup_weight(update: int, *, warmup_updates: int) -> float:
    if warmup_updates <= 0:
        return 1.0
    return min(1.0, update / warmup_updates)


@torch.inference_mode()
def save_prior_samples(
    model: HierarchicalVAE,
    path: Path,
    *,
    top_noise: torch.Tensor,
    lower_noise: torch.Tensor,
    batch_size: int,
    columns: int,
) -> None:
    samples = generate_in_batches(
        (top_noise, lower_noise), batch_size, model.generate, module=model
    )
    save_image(samples, path, nrow=columns)


def train_hierarchy(
    model: HierarchicalVAE,
    train_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    minimum_learning_rate: float,
    warmup_epochs: float,
    free_bits: float,
    out_dir: Path,
    model_name: str,
    seed: int,
    sample_count: int,
    sample_grid_columns: int,
    sample_every: int,
    progress_interval: float,
    max_metric_panels: int,
) -> None:
    # Match data order and base noise across the two posterior families.
    set_seed(seed)
    if not out_dir.exists():
        reset_dir(str(out_dir))
    training_dir = out_dir / "training"
    reset_dir(str(training_dir))
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=minimum_learning_rate
    )
    # Reuse both noise tensors so the grids show model changes across epochs.
    top_noise = torch.randn(sample_count, model.z2_dim, device=device)
    lower_noise = torch.randn(sample_count, model.z1_dim, device=device)
    warmup_updates = round(warmup_epochs * len(train_loader))
    update = 0
    with tqdm(
        total=epochs * len(train_loader),
        desc=f"{model_name} 1/{epochs}",
        unit="batch",
        mininterval=progress_interval,
    ) as progress:
        for epoch in range(1, epochs + 1):
            progress.set_description(f"{model_name} {epoch}/{epochs}", refresh=False)
            model.train()
            metrics = MetricAccumulator(
                ("loss", "distortion", "kl_z1", "kl_z2"), device=device
            )
            for images, _ in train_loader:
                update += 1
                images = images.to(device, non_blocking=True)
                reconstruction, latents = model(images)
                kl_weight = warmup_weight(update, warmup_updates=warmup_updates)
                loss, terms = hierarchical_vae_loss(
                    reconstruction,
                    images,
                    latents,
                    kl_weight=kl_weight,
                    free_bits=free_bits,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.update(
                    (
                        loss,
                        terms["distortion"],
                        terms["kl_z1"],
                        terms["kl_z2"],
                    ),
                    num_examples=images.shape[0],
                )
                progress.set_postfix(
                    loss=f"{metrics.compute()['loss']:.3f}",
                    refresh=False,
                )
                progress.update(1)
            values = metrics.compute_finite()
            history.append(
                values
                | {
                    "kl_weight": warmup_weight(update, warmup_updates=warmup_updates),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            scheduler.step()
            if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
                save_prior_samples(
                    model,
                    training_dir / f"epoch_{epoch:03d}.png",
                    top_noise=top_noise,
                    lower_noise=lower_noise,
                    batch_size=train_loader.batch_size,
                    columns=sample_grid_columns,
                )

    save_training_metrics(
        history, out_dir, prefix=model_name, max_panels=max_metric_panels
    )
    save_model_weights(
        model,
        out_dir / f"{model_name}.pth",
        metadata={
            "model_name": model_name,
            "model_config": model_config(model),
            "data_config": glasses_data_config(),
            "objective": HIERARCHY_OBJECTIVE,
            "posterior_family": model.posterior_family,
            "warmup_epochs": warmup_epochs,
            "free_bits_per_group": free_bits,
            "training_config": {
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "scheduler": "CosineAnnealingLR",
                "minimum_learning_rate": minimum_learning_rate,
                "betas": [0.9, 0.999],
                "seed": seed,
            },
        },
    )
    save_prior_samples(
        model,
        out_dir / "prior_samples.png",
        top_noise=top_noise,
        lower_noise=lower_noise,
        batch_size=train_loader.batch_size,
        columns=sample_grid_columns,
    )


__all__ = ["save_prior_samples", "train_hierarchy", "warmup_weight"]
