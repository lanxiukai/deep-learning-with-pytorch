"""Conditional hinge-GAN updates, schedules, and EMA buffer calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from dl_utils.gan.sn_gan import (
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.precision import BF16Precision, FP32Precision


@dataclass(frozen=True)
class UpdateRatioSchedule:
    """Plan an exact number of discriminator updates per generator update."""

    discriminator_updates_per_generator: int
    batches_per_epoch: int
    num_epochs: int

    def __post_init__(self):
        if self.discriminator_updates_per_generator < 1:
            raise ValueError("discriminator_updates_per_generator must be positive.")
        if self.batches_per_epoch < 1 or self.num_epochs < 1:
            raise ValueError("batches_per_epoch and num_epochs must be positive.")
        if self.discriminator_updates_per_generator > self.batches_per_epoch:
            raise ValueError(
                "discriminator_updates_per_generator must not exceed batches_per_epoch."
            )
        _, incomplete_cycle = divmod(
            self.total_discriminator_updates,
            self.discriminator_updates_per_generator,
        )
        if incomplete_cycle:
            raise ValueError(
                "the training plan does not end on a complete "
                "discriminator-to-generator update cycle."
            )

    @property
    def total_discriminator_updates(self) -> int:
        return self.num_epochs * self.batches_per_epoch

    @property
    def total_generator_updates(self) -> int:
        return (
            self.total_discriminator_updates // self.discriminator_updates_per_generator
        )

    def completed_discriminator_updates(self, completed_epochs: int) -> int:
        """Return the expected D step count after complete epochs."""
        if not 0 <= completed_epochs <= self.num_epochs:
            raise ValueError(f"completed_epochs must be within [0, {self.num_epochs}].")
        return completed_epochs * self.batches_per_epoch


@dataclass(frozen=True)
class ConditionalHingeStepResult:
    """Detached losses and global phase after one discriminator update."""

    discriminator: torch.Tensor
    generator_total: torch.Tensor | None
    generator_adversarial: torch.Tensor | None
    generator_regularization: torch.Tensor | None
    discriminator_steps: int


@dataclass(frozen=True)
class ConditionalHingeEpochResult:
    """Mean losses and global discriminator phase after one epoch."""

    discriminator: float
    generator_total: float
    generator_adversarial: float
    generator_regularization: float
    discriminator_steps: int


def train_conditional_hinge_step(
    generator: nn.Module,
    discriminator: nn.Module,
    real: torch.Tensor,
    labels: torch.Tensor,
    optimizer_g,
    optimizer_d,
    device: torch.device,
    precision: BF16Precision | FP32Precision,
    *,
    z_dim: int,
    num_classes: int,
    generator_batch_size: int,
    discriminator_steps: int,
    discriminator_updates_per_generator: int,
    generator_regularizer: Callable[[nn.Module], torch.Tensor] | None = None,
    after_generator_step: Callable[[], None] | None = None,
) -> ConditionalHingeStepResult:
    """Run one shared conditional hinge-GAN discriminator phase."""
    if min(z_dim, num_classes, generator_batch_size) < 1:
        raise ValueError("latent, class, and generator batch sizes must be positive.")
    if discriminator_steps < 0:
        raise ValueError("discriminator_steps must be non-negative.")
    if discriminator_updates_per_generator < 1:
        raise ValueError("discriminator update ratio must be positive.")

    real = real.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    batch_size = real.shape[0]
    optimizer_d.zero_grad(set_to_none=True)
    with precision.autocast():
        noise = torch.randn(batch_size, z_dim, device=device)
        # Match class counts so the D update focuses on image differences.
        with torch.no_grad():
            fake = generator(noise, labels)
        loss_d = discriminator_hinge_loss(
            discriminator(real, labels),
            discriminator(fake, labels),
        )
    precision.backward_step(loss_d, optimizer_d)

    discriminator_steps += 1
    if discriminator_steps % discriminator_updates_per_generator:
        return ConditionalHingeStepResult(
            discriminator=loss_d.detach(),
            generator_total=None,
            generator_adversarial=None,
            generator_regularization=None,
            discriminator_steps=discriminator_steps,
        )

    sampled_labels = torch.randint(
        num_classes,
        (generator_batch_size,),
        device=device,
    )
    noise = torch.randn(generator_batch_size, z_dim, device=device)
    discriminator.requires_grad_(False)
    try:
        optimizer_g.zero_grad(set_to_none=True)
        with precision.autocast():
            fake = generator(noise, sampled_labels)
            loss_g_adversarial = generator_hinge_loss(
                discriminator(fake, sampled_labels)
            )
        loss_g_regularization = (
            loss_g_adversarial.new_zeros(())
            if generator_regularizer is None
            else generator_regularizer(generator)
        )
        loss_g_total = loss_g_adversarial.float() + loss_g_regularization
        precision.backward_step(loss_g_total, optimizer_g)
        if after_generator_step is not None:
            after_generator_step()
    finally:
        discriminator.requires_grad_(True)

    return ConditionalHingeStepResult(
        discriminator=loss_d.detach(),
        generator_total=loss_g_total.detach(),
        generator_adversarial=loss_g_adversarial.detach(),
        generator_regularization=loss_g_regularization.detach(),
        discriminator_steps=discriminator_steps,
    )


def train_conditional_hinge_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader,
    optimizer_g,
    optimizer_d,
    device: torch.device,
    precision: BF16Precision | FP32Precision,
    *,
    z_dim: int,
    num_classes: int,
    generator_batch_size: int,
    discriminator_steps: int,
    update_schedule: UpdateRatioSchedule,
    generator_regularizer: Callable[[nn.Module], torch.Tensor] | None = None,
    after_generator_step: Callable[[], None] | None = None,
    progress_bar=None,
) -> ConditionalHingeEpochResult:
    """Train one epoch while keeping each model's update ratio explicit."""
    discriminator_metrics = MetricAccumulator(("loss",), device=device)
    generator_metrics = MetricAccumulator(
        ("total", "adversarial", "regularization"),
        device=device,
    )

    for real, labels in loader:
        result = train_conditional_hinge_step(
            generator,
            discriminator,
            real,
            labels,
            optimizer_g,
            optimizer_d,
            device,
            precision,
            z_dim=z_dim,
            num_classes=num_classes,
            generator_batch_size=generator_batch_size,
            discriminator_steps=discriminator_steps,
            discriminator_updates_per_generator=(
                update_schedule.discriminator_updates_per_generator
            ),
            generator_regularizer=generator_regularizer,
            after_generator_step=after_generator_step,
        )
        discriminator_steps = result.discriminator_steps
        batch_size = real.shape[0]
        discriminator_metrics.add_batch_means(
            (result.discriminator,), num_examples=batch_size
        )
        if result.generator_total is not None:
            assert result.generator_adversarial is not None
            assert result.generator_regularization is not None
            generator_metrics.add_batch_means(
                (
                    result.generator_total,
                    result.generator_adversarial,
                    result.generator_regularization,
                ),
                num_examples=generator_batch_size,
            )

        if progress_bar is not None:
            progress_bar.update(1)

    generator_losses = generator_metrics.compute_weighted_means()
    return ConditionalHingeEpochResult(
        discriminator=discriminator_metrics.compute_weighted_means()["loss"],
        generator_total=generator_losses["total"],
        generator_adversarial=generator_losses["adversarial"],
        generator_regularization=generator_losses["regularization"],
        discriminator_steps=discriminator_steps,
    )


@torch.no_grad()
def refresh_generator_statistics(
    generator: nn.Module,
    *,
    z_dim: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 64,
    num_batches: int = 32,
) -> None:
    """Refresh an EMA generator's BatchNorm and spectral-norm buffers.

    Averaged parameters can disagree with buffers copied from the live model.
    Forward-only calibration updates those buffers without changing parameters
    or consuming the training loop's random-number stream.
    """
    random_generator = torch.Generator(device=device).manual_seed(771)
    was_training = generator.training
    generator.train()
    try:
        for _ in range(num_batches):
            noise = torch.randn(
                batch_size, z_dim, device=device, generator=random_generator
            )
            labels = torch.randint(
                num_classes,
                (batch_size,),
                device=device,
                generator=random_generator,
            )
            generator(noise, labels)
    finally:
        generator.train(was_training)


__all__ = [
    "ConditionalHingeEpochResult",
    "ConditionalHingeStepResult",
    "UpdateRatioSchedule",
    "refresh_generator_statistics",
    "train_conditional_hinge_epoch",
    "train_conditional_hinge_step",
]
