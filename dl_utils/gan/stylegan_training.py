"""Latent mixing and R1 regularization shared by style-based GAN lessons."""

import random

import torch


def sample_mixing_latents(
    batch_size: int,
    z_dim: int,
    device: torch.device,
    mixing_probability: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample a primary Z batch and an optional style-mixing batch."""
    if not 0.0 <= mixing_probability <= 1.0:
        raise ValueError("mixing_probability must be within [0, 1].")
    z = torch.randn(batch_size, z_dim, device=device)
    mixing_z = torch.randn_like(z) if random.random() < mixing_probability else None
    # z: (B, z_dim), mixing_z: (B, z_dim) or None
    return z, mixing_z


def r1_penalty(
    real_scores: torch.Tensor,
    real_images: torch.Tensor,
) -> torch.Tensor:
    """Measure squared discriminator gradients at real images."""
    # real_scores: (B,), real_images: (B, 3, H, W)
    # R1 = Eₓ[ ||∇ₓ D(x)||² ]
    gradients = torch.autograd.grad(  # ∂output / ∂input
        real_scores.sum(),  # outputs
        real_images,  # inputs
        create_graph=True,
    )[0]  # (B, 3, H, W)
    return gradients.square().flatten(1).sum(dim=1).mean()


__all__ = ["r1_penalty", "sample_mixing_latents"]
