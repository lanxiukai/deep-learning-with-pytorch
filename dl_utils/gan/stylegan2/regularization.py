"""Path-length regularization for StyleGAN2 synthesis."""

import math

import torch


def path_length_penalty(
    images: torch.Tensor,
    ws: torch.Tensor,
    running_mean: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularize image-space change per unit movement in W."""
    # images: (Bpath, 3, H, W), ws: (Bpath, num_ws, style_dim)
    if running_mean.ndim != 0:
        raise ValueError("running_mean must be a scalar tensor.")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be within [0, 1].")

    noise = torch.randn_like(images) / math.sqrt(images.shape[2] * images.shape[3])
    gradients = torch.autograd.grad(  # ∂output / ∂input
        (images * noise).sum(),  # output
        ws,  # input
        create_graph=True,
    )[0]  # (Bpath, num_ws, style_dim)
    lengths = torch.sqrt(gradients.square().sum(dim=2).mean(dim=1) + 1e-8)  # (B,)
    # running_mean + decay * (lengths.mean() - running_mean)
    updated_mean = running_mean.lerp(lengths.mean().detach(), decay)
    return (lengths - updated_mean).square().mean(), updated_mean


__all__ = ["path_length_penalty"]
