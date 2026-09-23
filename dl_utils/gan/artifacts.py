"""GAN loss layouts and fixed-latent sample artifacts."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from os import PathLike

import torch
from torch import nn

from dl_utils.gan.stylegan_layers import denormalize
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot._backend import pyplot as _plt
from dl_utils.plot.images import save_grid
from dl_utils.training.precision import BF16Precision


def save_loss_curves(
    steps: Sequence[float],
    discriminator_losses: Sequence[float],
    generator_losses: Sequence[float],
    generator_adversarial_losses: Mapping[str, Sequence[float]],
    generator_reconstruction_losses: Mapping[str, Sequence[float]],
    path: str | PathLike[str],
    *,
    xlabel: str = "epoch",
) -> None:
    """Save total and component losses on four independent-y subplots.

    The component mappings supply the plotted curves and their legend labels;
    every series must align with ``steps``.
    """
    x_values = list(steps)
    discriminator_values = list(map(float, discriminator_losses))
    generator_values = list(map(float, generator_losses))
    if len(discriminator_values) != len(x_values):
        raise ValueError(
            "save_loss_curves: discriminator loss length does not match steps."
        )
    if len(generator_values) != len(x_values):
        raise ValueError(
            "save_loss_curves: generator loss length does not match steps."
        )

    def normalize_components(
        name: str,
        curves: Mapping[str, Sequence[float]],
    ) -> dict[str, list[float]]:
        if not curves:
            raise ValueError(f"save_loss_curves: {name} losses are empty.")
        normalized = {}
        for label, values in curves.items():
            normalized_values = list(map(float, values))
            if len(normalized_values) != len(x_values):
                raise ValueError(
                    f"save_loss_curves: {name} loss '{label}' length "
                    "does not match steps."
                )
            normalized[label] = normalized_values
        return normalized

    adversarial_values = normalize_components(
        "generator adversarial",
        generator_adversarial_losses,
    )
    reconstruction_values = normalize_components(
        "generator reconstruction",
        generator_reconstruction_losses,
    )

    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, axes = _plt.subplots(
        4,
        1,
        figsize=(7, 12),
        sharex=True,
        sharey=False,
    )
    axes[0].plot(x_values, discriminator_values, color="tab:blue")
    axes[0].set_title("Total D loss")
    axes[1].plot(x_values, generator_values, color="tab:orange")
    axes[1].set_title("Total G loss")

    component_groups = (
        (axes[2], "G adversarial loss", adversarial_values),
        (axes[3], "G reconstruction loss", reconstruction_values),
    )
    colors = ("tab:blue", "tab:orange")
    line_styles = ("-", "--")
    for axis, title, curves in component_groups:
        for index, (label, values) in enumerate(curves.items()):
            axis.plot(
                x_values,
                values,
                color=colors[index % len(colors)],
                linestyle=line_styles[index % len(line_styles)],
                label=label,
            )
        axis.set_title(title)
        axis.legend()

    for axis in axes:
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path_str, dpi=300)
    _plt.close(fig)


def save_gan_samples(
    generator: nn.Module,
    fixed_z: torch.Tensor,
    output_path: str | PathLike[str],
    *,
    precision: BF16Precision,
    generate: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    nrow: int,
    title: str,
) -> None:
    """Render a fixed-latent EMA sample grid through BF16 inference."""

    def generate_bf16(z_batch: torch.Tensor) -> torch.Tensor:
        with precision.autocast():
            return generate(z_batch).float()

    samples = generate_in_batches(
        fixed_z,
        batch_size,
        generate_bf16,
        module=generator,
    )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("GAN sample generation produced non-finite pixels.")
    save_grid(
        denormalize(samples),
        os.fspath(output_path),
        nrow=nrow,
        title=title,
    )


__all__ = [
    "save_gan_samples",
    "save_loss_curves",
]
