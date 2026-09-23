"""Shared GAN objectives, updates, schedules, runtime, and checkpoint state."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

from dl_utils.data.datasets.celeba import CelebATrainingStream
from dl_utils.data.loading import resolve_num_workers
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import configure_device, try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import TrainingCheckpoint
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.precision import (
    BF16Precision,
    FP32Precision,
    resolve_bf16_precision,
)


@dataclass
class GANRun:
    """Own common paths, BF16 runtime state, and the CelebA stream."""

    output_dir: Path
    training_dir: Path
    checkpoint_dir: Path
    device: torch.device
    precision: BF16Precision
    data: CelebATrainingStream

    @property
    def pipeline(self) -> str:
        return self.data.pipeline

    @property
    def dataset_size(self) -> int:
        return self.data.dataset_size

    def prepare_output(self, resume_from: str | PathLike[str] | None) -> None:
        """Reset transient artifacts for a fresh run or retain them on resume."""
        if resume_from is None:
            reset_dir(str(self.training_dir))
            reset_dir(str(self.checkpoint_dir))
            return
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def prepare_gan_run(
    model_name: str,
    *,
    seed: int,
    data_pipeline: str,
    num_workers: int,
    prefetch_factor: int,
    project_root: str | PathLike[str] | None = None,
) -> GANRun:
    """Validate data and construct the single-GPU BF16 lesson runtime."""
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be one safe path component.")
    root = (
        infer_project_root() if project_root is None else Path(project_root).resolve()
    )
    data_dir = root / "data" / "celeba"
    if not (data_dir / "list_eval_partition.csv").is_file():
        raise FileNotFoundError(
            f"CelebA data not found: {data_dir}. "
            "Run tool_scripts/download_dataset.py first."
        )

    output_root = Path(os.environ.get("DL_OUTPUT_ROOT", str(root / "output" / "gan")))
    output_dir = output_root / model_name
    set_seed(seed)
    device = try_gpu()
    configure_device(device)
    precision = resolve_bf16_precision(device)
    data = CelebATrainingStream(
        data_dir,
        device,
        pipeline=data_pipeline,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    return GANRun(
        output_dir=output_dir,
        training_dir=output_dir / "training",
        checkpoint_dir=output_dir / "checkpoints",
        device=device,
        precision=precision,
        data=data,
    )


def initialize_gan_models[
    GeneratorT: nn.Module,
    DiscriminatorT: nn.Module,
](
    generator: GeneratorT,
    discriminator: DiscriminatorT,
    device: torch.device,
) -> tuple[GeneratorT, DiscriminatorT, GeneratorT]:
    """Move both models to the device, optimize 4D state, and create the EMA."""

    def move(module: nn.Module) -> None:
        module.to(device=device)
        module._apply(
            lambda tensor: (
                tensor.contiguous(memory_format=torch.channels_last)
                if tensor.ndim == 4
                else tensor
            )
        )

    move(generator)
    move(discriminator)
    averaged_generator = deepcopy(generator).eval().requires_grad_(False)
    return generator, discriminator, averaged_generator


def start_gan_checkpoint(
    path: str | PathLike[str],
    *,
    resume_from: str | PathLike[str] | None,
    unit: str,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    metric_names: tuple[str, ...],
    fixed_z: torch.Tensor,
    run_config: Mapping[str, Any],
    extra_state: Mapping[str, Any] | None = None,
) -> tuple[TrainingCheckpoint, int, dict[str, Any]]:
    """Start one compatible latest-checkpoint stream for a GAN lesson."""
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("metric_names must be non-empty and unique.")
    reserved = {"loss_history", "fixed_z", "run_config"}
    extras = dict(extra_state or {})
    overlap = sorted(reserved & set(extras))
    if overlap:
        raise ValueError(f"extra_state uses reserved keys: {overlap}.")

    initial_state = {
        "loss_history": {
            "kimg": [],
            **{name: [] for name in metric_names},
        },
        "fixed_z": fixed_z.detach().cpu(),
        "run_config": dict(run_config),
        **extras,
    }
    checkpoint = TrainingCheckpoint(
        path,
        unit=unit,
        models=models,
        optimizers=optimizers,
    )
    completed_units, state = checkpoint.resume(
        resume_from,
        initial_state=initial_state,
    )
    if set(state) != set(initial_state):
        raise ValueError(
            "Checkpoint training-state keys differ from this lesson; "
            f"saved={sorted(state)}, expected={sorted(initial_state)}."
        )
    if state["run_config"] != initial_state["run_config"]:
        raise ValueError(
            "Checkpoint runtime options differ from this run; reuse the "
            "original batch, budget, regularization, and data-pipeline "
            "options."
        )
    history = state["loss_history"]
    expected_history_keys = {"kimg", *metric_names}
    if not isinstance(history, dict) or set(history) != expected_history_keys:
        raise ValueError("Checkpoint loss history does not match this lesson.")
    if not isinstance(state["fixed_z"], torch.Tensor):
        raise TypeError("Checkpoint fixed_z must be a tensor.")
    return checkpoint, completed_units, state


def append_gan_metrics(
    history: dict[str, list[float]],
    seen_kimg: float,
    metrics: Mapping[str, float],
) -> None:
    """Append one boundary's ordered metrics to a GAN loss history."""
    expected_names = set(history) - {"kimg"}
    if set(metrics) != expected_names:
        raise ValueError("metrics do not match the configured loss history.")
    history["kimg"].append(float(seen_kimg))
    for name, value in metrics.items():
        history[name].append(float(value))


@dataclass(frozen=True)
class ProgressivePhase:
    """One fade-in or stabilization phase at a fixed resolution."""

    resolution: int
    name: str
    batch_size: int
    num_batches: int

    @property
    def num_images(self):
        return self.num_batches * self.batch_size


def build_progressive_schedule(
    *,
    resolutions: Sequence[int],
    batch_sizes: Mapping[int, int],
    phase_kimg: int,
) -> tuple[ProgressivePhase, ...]:
    """Build the fade-in and stabilization phases used by ProGAN."""
    resolutions = tuple(int(value) for value in resolutions)
    phase_kimg = int(phase_kimg)
    if phase_kimg < 1:
        raise ValueError("phase_kimg must be positive.")
    phases = []
    for resolution in resolutions:
        batch_size = int(batch_sizes[resolution])
        target_images = phase_kimg * 1_000
        if batch_size < 1:
            raise ValueError("progressive batch sizes must be positive.")
        names = (
            ("stabilization",)
            if resolution == resolutions[0]
            else ("fade-in", "stabilization")
        )
        phases.extend(
            ProgressivePhase(
                resolution=resolution,
                name=name,
                batch_size=batch_size,
                num_batches=math.ceil(target_images / batch_size),
            )
            for name in names
        )
    return tuple(phases)


def phase_alpha(phase: ProgressivePhase, batch_index: int) -> float:
    """Return the linear fade-in coefficient for one phase batch."""
    if phase.name != "fade-in" or phase.num_batches == 1:
        return 1.0
    return batch_index / (phase.num_batches - 1)


@dataclass(frozen=True)
class ProgressiveGANOptions:
    """Resolved runtime options for a progressive GAN lesson."""

    phase_kimg: int
    batch_sizes: dict[int, int]
    d_reg_every: int
    reg_batch_shrink: int
    num_workers: int
    prefetch_factor: int


def resolve_progressive_gan_options(
    *,
    phase_kimg: int | None,
    batch_scale: int | None,
    d_reg_every: int | None,
    reg_batch_shrink: int | None,
    num_workers: int | None,
    prefetch_factor: int,
    base_batch_sizes: Mapping[int, int],
    default_phase_kimg: int,
    default_d_reg_every: int,
    default_reg_batch_shrink: int,
    default_num_workers: int,
) -> ProgressiveGANOptions:
    """Validate and resolve the shared ProGAN/StyleGAN runtime options."""
    resolved_phase_kimg = default_phase_kimg if phase_kimg is None else phase_kimg
    resolved_batch_scale = 1 if batch_scale is None else batch_scale
    resolved_d_reg_every = default_d_reg_every if d_reg_every is None else d_reg_every
    resolved_reg_batch_shrink = (
        default_reg_batch_shrink if reg_batch_shrink is None else reg_batch_shrink
    )
    if (
        min(
            resolved_phase_kimg,
            resolved_batch_scale,
            resolved_d_reg_every,
            resolved_reg_batch_shrink,
            prefetch_factor,
        )
        < 1
    ):
        raise ValueError("training counts and scales must be positive.")
    batch_sizes = {
        int(resolution): int(batch_size) * resolved_batch_scale
        for resolution, batch_size in base_batch_sizes.items()
    }
    if not batch_sizes or min(batch_sizes.values()) < 1:
        raise ValueError("base_batch_sizes must contain positive values.")
    return ProgressiveGANOptions(
        phase_kimg=resolved_phase_kimg,
        batch_sizes=batch_sizes,
        d_reg_every=resolved_d_reg_every,
        reg_batch_shrink=resolved_reg_batch_shrink,
        num_workers=resolve_num_workers(num_workers, default_num_workers),
        prefetch_factor=prefetch_factor,
    )


def update_discriminator(
    real_samples,
    noise,
    discriminator,
    generator,
    loss_function,
    optimizer,
    real_label: float = 1.0,
):
    """Update the discriminator on one batch of real and generated samples."""
    batch_size = real_samples.shape[0]
    real_labels = torch.full(
        (batch_size,),
        real_label,
        device=real_samples.device,
    )
    generated_labels = torch.zeros(batch_size, device=real_samples.device)

    optimizer.zero_grad()
    real_outputs = discriminator(real_samples)
    generated_samples = generator(noise)
    generated_outputs = discriminator(generated_samples.detach())
    discriminator_loss = (
        loss_function(real_outputs, real_labels.reshape(real_outputs.shape))
        + loss_function(
            generated_outputs,
            generated_labels.reshape(generated_outputs.shape),
        )
    ) / 2
    discriminator_loss.backward()
    optimizer.step()
    return discriminator_loss


def update_generator(noise, discriminator, generator, loss_function, optimizer):
    """Update the generator to make generated samples score as real."""
    batch_size = noise.shape[0]
    real_labels = torch.ones(batch_size, device=noise.device)

    optimizer.zero_grad()
    generated_samples = generator(noise)
    generated_outputs = discriminator(generated_samples)
    generator_loss = loss_function(
        generated_outputs,
        real_labels.reshape(generated_outputs.shape),
    )
    generator_loss.backward()
    optimizer.step()
    return generator_loss


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


def discriminator_hinge_loss(real_scores, fake_scores):
    """Return the discriminator hinge loss."""
    return F.relu(1 - real_scores).mean() + F.relu(1 + fake_scores).mean()


def generator_hinge_loss(fake_scores):
    """Return the generator hinge objective."""
    return -fake_scores.mean()


__all__ = [
    "ConditionalHingeEpochResult",
    "ConditionalHingeStepResult",
    "GANRun",
    "ProgressiveGANOptions",
    "ProgressivePhase",
    "UpdateRatioSchedule",
    "append_gan_metrics",
    "build_progressive_schedule",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "initialize_gan_models",
    "phase_alpha",
    "prepare_gan_run",
    "r1_penalty",
    "refresh_generator_statistics",
    "resolve_progressive_gan_options",
    "sample_mixing_latents",
    "start_gan_checkpoint",
    "train_conditional_hinge_epoch",
    "train_conditional_hinge_step",
    "update_discriminator",
    "update_generator",
]
