"""Lightweight shared training loop for directly comparable hierarchy lessons."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.factor_shapes import FactorShapes32
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
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        latents = model.infer(x, sample=False)
        reconstruction = model.decode(latents["z1"])
        distortion += float(F.binary_cross_entropy(reconstruction, x, reduction="sum"))
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
        examples += x.shape[0]
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
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    warmup_updates = round(warmup_epochs * len(train_loader))
    update = 0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = torch.zeros(4, device=device)
        examples = 0
        active = ActiveUnitAccumulator()
        for x, _ in train_loader:
            update += 1
            x = x.to(device, non_blocking=True)
            reconstruction, latents = model(x)
            kl_weight = warmup_weight(update, warmup_updates=warmup_updates)
            loss, terms = hierarchical_vae_loss(
                reconstruction,
                x,
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
                * x.shape[0]
            )
            examples += x.shape[0]
            active.update(latents)
        values = (totals / examples).tolist()
        active_z1, active_z2 = active.counts(
            variance_threshold=active_variance_threshold
        )
        print(
            f"epoch {epoch:03d}: loss={values[0]:.3f}, "
            f"D={values[1]:.3f}, KL(z1|z2)={values[2]:.3f}, "
            f"KL(z2)={values[3]:.3f}, "
            f"AU(z1 correction)={active_z1}/{model.z1_dim}, "
            f"AU(z2)={active_z2}/{model.z2_dim}, "
            f"warm-up={warmup_weight(update, warmup_updates=warmup_updates):.3f}"
        )

    validation = evaluate_hierarchy(
        model,
        test_loader,
        device=device,
        active_variance_threshold=active_variance_threshold,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "model_config": model_config(model),
            "posterior_family": model.posterior_family,
            "warmup_epochs": warmup_epochs,
            "free_bits_per_group": free_bits,
            "split_seed": split_seed,
        },
        out_dir / "model.pth",
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    print(
        f"validation D={validation['distortion']:.3f}, "
        f"KL(z1|z2)={validation['kl_z1']:.3f}, "
        f"KL(z2)={validation['kl_z2']:.3f}; saved {out_dir}"
    )


__all__ = [
    "evaluate_hierarchy",
    "make_factor_shape_loaders",
    "train_hierarchy",
    "warmup_weight",
]
