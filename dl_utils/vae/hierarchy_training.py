"""Lightweight shared training loop for directly comparable hierarchy lessons."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.factor_shapes import FactorShapes32
from dl_utils.filesystem.directories import reset_dir
from dl_utils.vae.training_artifacts import save_training_metrics
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar
from dl_utils.vae.vae_hierarchy import (
    ActiveUnitAccumulator,
    HierarchicalVAE32,
    hierarchical_vae_loss,
    model_config,
)


def make_factor_shape_loaders(
    *,
    batch_size: int,
    workers: int,
    split_seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    train_set = FactorShapes32(split="train", split_seed=split_seed)
    test_set = FactorShapes32(split="test", split_seed=split_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(test_set, shuffle=False, drop_last=False, **common),
    )


def warmup_weight(update: int, *, warmup_updates: int) -> float:
    if warmup_updates <= 0:
        return 1.0
    return min(1.0, update / warmup_updates)


@torch.inference_mode()
def evaluate_hierarchy(
    model: HierarchicalVAE32,
    loader: DataLoader,
    *,
    device: torch.device,
    active_variance_threshold: float,
) -> dict[str, float]:
    model.eval()
    distortion = 0.0
    kl_z1 = 0.0
    kl_z2 = 0.0
    examples = 0
    active = ActiveUnitAccumulator()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        latents = model.infer(images, sample=False)
        reconstruction = model.decode(latents["z1"])
        distortion += float(
            F.binary_cross_entropy(reconstruction, images, reduction="sum")
        )
        kl_z1 += float(
            diagonal_gaussian_kl_from_logvar(
                latents["q1_mu"],
                latents["q1_logvar"],
                latents["p1_mu"],
                latents["p1_logvar"],
            ).sum()
        )
        kl_z2 += float(
            diagonal_gaussian_kl_from_logvar(
                latents["q2_mu"], latents["q2_logvar"]
            ).sum()
        )
        active.update(latents)
        examples += images.shape[0]
    active_z1, active_z2 = active.counts(variance_threshold=active_variance_threshold)
    return {
        "distortion": distortion / examples,
        "kl_z1": kl_z1 / examples,
        "kl_z2": kl_z2 / examples,
        "total_rate": (kl_z1 + kl_z2) / examples,
        "active_z1_corrections": float(active_z1),
        "active_z2_units": float(active_z2),
    }


def train_hierarchy(
    model: HierarchicalVAE32,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    warmup_epochs: float,
    free_bits: float,
    active_variance_threshold: float,
    out_dir: Path,
    model_name: str,
    split_seed: int,
    sample_count: int,
    sample_grid_columns: int,
    sample_every: int,
    progress_interval: float,
    max_metric_panels: int,
) -> None:
    reset_dir(str(out_dir))
    training_dir = out_dir / "training"
    training_dir.mkdir()
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    warmup_updates = round(warmup_epochs * len(train_loader))
    update = 0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = torch.zeros(4, device=device)
        examples = 0
        active = ActiveUnitAccumulator()
        progress = tqdm(
            train_loader,
            desc=f"{model_name} {epoch}/{epochs}",
            mininterval=progress_interval,
        )
        for images, _ in progress:
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
            totals += (
                torch.stack(
                    [
                        loss.detach(),
                        terms["distortion"],
                        terms["kl_z1"],
                        terms["kl_z2"],
                    ]
                )
                * images.shape[0]
            )
            examples += images.shape[0]
            active.update(latents)
            progress.set_postfix(
                loss=f"{float(totals[0]) / examples:.3f}",
                refresh=False,
            )
        values = (totals / examples).tolist()
        active_z1, active_z2 = active.counts(
            variance_threshold=active_variance_threshold
        )
        history.append(
            dict(zip(("loss", "distortion", "kl_z1", "kl_z2"), values))
            | {
                "active_z1": float(active_z1),
                "active_z2": float(active_z2),
                "kl_weight": warmup_weight(update, warmup_updates=warmup_updates),
            }
        )
        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            model.eval()
            with torch.inference_mode():
                samples = model.sample(sample_count, device=device)
            save_image(
                samples,
                training_dir / f"epoch_{epoch:03d}.png",
                nrow=sample_grid_columns,
            )

    validation = evaluate_hierarchy(
        model,
        test_loader,
        device=device,
        active_variance_threshold=active_variance_threshold,
    )
    save_training_metrics(
        history, out_dir, prefix=model_name, max_panels=max_metric_panels
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "model_config": model_config(model),
            "posterior_family": model.posterior_family,
            "warmup_epochs": warmup_epochs,
            "free_bits_per_group": free_bits,
            "split_seed": split_seed,
            "validation": validation,
        },
        out_dir / f"{model_name}.pth",
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(sample_count, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=sample_grid_columns)


__all__ = [
    "evaluate_hierarchy",
    "make_factor_shape_loaders",
    "train_hierarchy",
    "warmup_weight",
]
