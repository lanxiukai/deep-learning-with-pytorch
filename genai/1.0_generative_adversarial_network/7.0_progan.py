"""Train a compact ProGAN on 128x128 aligned CelebA.

The defaults target an NVIDIA B200 or B300: BF16 AMP, twice the teaching batch
size, a 2.2-million-image schedule, lazy full-FP32 gradient penalties, fused
Adam, channels-last tensors, and CUDA JPEG decoding when nvJPEG is available.
They favor useful reference images over paper-level quality or exact
reproduction.

Fresh runs reset ``output/gan/progan/training`` and ``checkpoints``.  A
resumable checkpoint and an EMA sample grid are written at each progressive
phase boundary, matching the reference lesson.

Run:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.0_progan.py

Resume:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.0_progan.py \
        --resume-from output/gan/progan/checkpoints/latest.pth

Default schedule:
    Progressive phases:      11 x 200 kimg
    Real images shown to D:  2.2 million
    Main D/G updates:        24,222 / 24,222

Default image sizes:
    Training input:          4x4 to 128x128 RGB, progressive by phase
    Generated image:         4x4 to 128x128 RGB, progressive by phase

Model size:
    Generator:               4.06 M parameters (plus one EMA copy)
    Discriminator:           4.59 M parameters
"""

import argparse
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm

from dl_utils.gan.progan import ProGANDiscriminator, ProGANGenerator
from dl_utils.gan.stylegan_common import (
    RESOLUTIONS,
    build_progressive_schedule,
    phase_alpha,
)
from dl_utils.gan.training import (
    append_gan_metrics,
    initialize_gan_models,
    prepare_gan_run,
    resolve_progressive_gan_options,
    save_gan_samples,
    start_gan_checkpoint,
)
from dl_utils.plot.figures import save_loss_panels
from dl_utils.training.accelerator import make_fused_adam
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.ema import update_ema_by_images
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.validation import validate_finite_training_state

BATCH_SIZES = {4: 256, 8: 256, 16: 128, 32: 64, 64: 64, 128: 64}
PHASE_KIMG = 200
NUM_WORKERS = 8
Z_DIM = 128
BASE_CHANNELS = 32
LEARNING_RATE = 1e-3
GRADIENT_PENALTY_WEIGHT = 10.0
DRIFT_PENALTY_WEIGHT = 1e-3
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 16
SEED = 42
D_REG_EVERY = 2
REG_BATCH_SHRINK = 2

MODEL_CONFIG = {"z_dim": Z_DIM, "base_channels": BASE_CHANNELS}
DISCRIMINATOR_CONFIG = {"base_channels": BASE_CHANNELS}
METRIC_NAMES = ("loss_d_main", "loss_g_main", "gradient_penalty")


def build_training_schedule(
    dataset_size,
    *,
    batch_sizes=None,
    phase_kimg=PHASE_KIMG,
):
    """Build one fade-in or stabilization entry per progressive phase."""
    batch_sizes = dict(BATCH_SIZES if batch_sizes is None else batch_sizes)
    if dataset_size < max(batch_sizes.values()):
        raise ValueError("CelebA train split is smaller than a training batch.")
    return build_progressive_schedule(
        resolutions=RESOLUTIONS,
        batch_sizes=batch_sizes,
        phase_kimg=phase_kimg,
    )


def gradient_penalty(discriminator, real_images, fake_images, resolution, alpha):
    """Penalize critic gradients away from unit norm."""
    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    interpolated = torch.lerp(fake_images, real_images, epsilon)
    interpolated.requires_grad_(True)
    scores = discriminator(interpolated, resolution=resolution, alpha=alpha)
    gradients = torch.autograd.grad(
        scores.sum(),
        interpolated,
        create_graph=True,
    )[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return (norms - 1.0).square().mean()


def train_phase(
    generator,
    discriminator,
    data,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    phase,
    precision,
    global_step,
    d_reg_every,
    reg_batch_shrink,
    *,
    gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
):
    """Train one progressive phase with BF16 main passes."""
    metrics = MetricAccumulator(METRIC_NAMES, device=data.device)
    progress = tqdm(
        range(phase.num_batches),
        desc=f"ProGAN {phase.resolution}x {phase.name}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for batch_index in progress:
        real_images = data.next_batch(
            phase.resolution,
            phase.batch_size,
        )
        alpha = phase_alpha(phase, batch_index)
        batch_size = len(real_images)

        optimizer_d.zero_grad(set_to_none=True)
        z = torch.randn(batch_size, generator.z_dim, device=data.device)
        with torch.no_grad(), precision.autocast():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
            )
        with precision.autocast():
            real_scores = discriminator(
                real_images,
                resolution=phase.resolution,
                alpha=alpha,
            )
            fake_scores = discriminator(
                fake_images,
                resolution=phase.resolution,
                alpha=alpha,
            )
            loss_d_main = fake_scores.mean() - real_scores.mean()
        weighted_drift = DRIFT_PENALTY_WEIGHT * real_scores.float().square().mean()
        weighted_gp = real_images.new_zeros(())
        if global_step % d_reg_every == 0:
            reg_batch = max(1, batch_size // reg_batch_shrink)
            penalty_gp = gradient_penalty(
                discriminator,
                real_images[:reg_batch].float(),
                fake_images[:reg_batch].detach().float(),
                phase.resolution,
                alpha,
            )
            weighted_gp = gradient_penalty_weight * d_reg_every * penalty_gp
        loss_d = loss_d_main.float() + weighted_drift + weighted_gp
        precision.backward_step(loss_d, optimizer_d)

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, generator.z_dim, device=data.device)
            with precision.autocast():
                fake_images = generator(
                    z,
                    resolution=phase.resolution,
                    alpha=alpha,
                )
                loss_g_main = -discriminator(
                    fake_images,
                    resolution=phase.resolution,
                    alpha=alpha,
                ).mean()
            precision.backward_step(loss_g_main, optimizer_g)
            update_ema_by_images(
                averaged_generator,
                generator,
                batch_size,
                EMA_HALF_LIFE_KIMG,
            )
        finally:
            discriminator.requires_grad_(True)

        metrics.add_batch_means(
            (loss_d_main, loss_g_main, weighted_gp),
            num_examples=batch_size,
        )
        global_step += 1

    return metrics.compute_weighted_means(require_finite=True), global_step


def main(args):
    if args.refine_from is not None or args.refine_resume is not None:
        from dl_utils.gan.refinement import refine_gan

        if args.resume_from is not None:
            raise ValueError("Use only one progressive or refinement resume mode.")
        refine_gan(args, model_name="progan", lesson=globals())
        return
    options = resolve_progressive_gan_options(
        phase_kimg=args.phase_kimg,
        batch_scale=args.batch_scale,
        d_reg_every=args.d_reg_every,
        reg_batch_shrink=args.reg_batch_shrink,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        base_batch_sizes=BATCH_SIZES,
        default_phase_kimg=PHASE_KIMG,
        default_d_reg_every=D_REG_EVERY,
        default_reg_batch_shrink=REG_BATCH_SHRINK,
        default_num_workers=NUM_WORKERS,
    )
    run = prepare_gan_run(
        "progan",
        seed=SEED,
        data_pipeline=args.data_pipeline,
        num_workers=options.num_workers,
        prefetch_factor=options.prefetch_factor,
    )
    training_schedule = build_training_schedule(
        run.dataset_size,
        batch_sizes=options.batch_sizes,
        phase_kimg=options.phase_kimg,
    )
    total_phases = len(training_schedule)
    run.prepare_output(args.resume_from)

    generator, discriminator, averaged_generator = initialize_gan_models(
        ProGANGenerator(**MODEL_CONFIG),
        ProGANDiscriminator(**DISCRIMINATOR_CONFIG),
        run.device,
    )
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=run.device,
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=run.device,
        lr=LEARNING_RATE,
        betas=(0.0, 0.99),
    )
    print(
        f"precision={run.precision.name} pipeline={run.pipeline} "
        f"workers={options.num_workers} "
        f"phase_kimg={options.phase_kimg} "
        f"D_reg_every={options.d_reg_every}"
    )

    models = {
        "online_generator": generator,
        "progan_generator": averaged_generator,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    run_config = {
        "precision": run.precision.name,
        "data_pipeline": run.pipeline,
        "phase_kimg": options.phase_kimg,
        "batch_sizes": options.batch_sizes,
        "d_reg_every": options.d_reg_every,
        "reg_batch_shrink": options.reg_batch_shrink,
    }
    checkpoint, completed_phases, state = start_gan_checkpoint(
        run.checkpoint_dir / "latest.pth",
        resume_from=args.resume_from,
        unit="progressive-phase-main-metrics-v2",
        models=models,
        optimizers=optimizers,
        metric_names=METRIC_NAMES,
        fixed_z=torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=run.device),
        run_config=run_config,
        extra_state={"global_step": 0},
    )
    if completed_phases > total_phases:
        raise ValueError("Checkpoint phase exceeds the training schedule.")
    if args.resume_from is not None:
        print(f"Resumed after phase {completed_phases}: {args.resume_from}")

    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(run.device)
    global_step = int(state["global_step"])
    completed_schedule = training_schedule[:completed_phases]
    seen_images = sum(phase.num_images for phase in completed_schedule)
    expected_steps = sum(phase.num_batches for phase in completed_schedule)
    if global_step != expected_steps:
        raise ValueError("Checkpoint global step does not match its phase.")

    for phase_index, phase in enumerate(
        training_schedule[completed_phases:],
        start=completed_phases + 1,
    ):
        print(
            f"Phase {phase_index}/{total_phases}: "
            f"{phase.resolution}x{phase.resolution} {phase.name}"
        )
        metrics, global_step = train_phase(
            generator,
            discriminator,
            run.data,
            optimizer_g,
            optimizer_d,
            averaged_generator,
            phase,
            run.precision,
            global_step,
            options.d_reg_every,
            options.reg_batch_shrink,
        )
        validate_finite_training_state(models, optimizers)
        seen_images += phase.num_images
        seen_kimg = seen_images / 1_000
        append_gan_metrics(loss_history, seen_kimg, metrics)

        state["global_step"] = global_step
        checkpoint.save(phase_index, state)
        save_gan_samples(
            averaged_generator,
            fixed_z,
            run.training_dir / f"kimg_{round(seen_kimg):05d}.png",
            precision=run.precision,
            generate=partial(
                averaged_generator,
                resolution=phase.resolution,
                alpha=phase_alpha(phase, phase.num_batches - 1),
            ),
            batch_size=SAMPLE_BATCH_SIZE,
            nrow=SAMPLE_GRID_COLUMNS,
            title=(
                f"ProGAN EMA fixed-z - {phase.resolution}x{phase.resolution} "
                f"{phase.name} - {seen_kimg:.0f} kimg"
            ),
        )

    save_model_weights(
        averaged_generator,
        run.output_dir / "progan_generator.pth",
        metadata={"model_name": "progan", "model_config": MODEL_CONFIG},
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D Wasserstein": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G Wasserstein": loss_history["loss_g_main"],
            },
            "Gradient penalty": {
                "Expected weighted contribution": loss_history["gradient_penalty"],
            },
        },
        run.output_dir / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive 128x128 CelebA ProGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    from dl_utils.gan.refinement import add_refinement_arguments

    add_refinement_arguments(parser, model_name="progan")
    parser.add_argument("--phase-kimg", type=int)
    parser.add_argument("--batch-scale", type=int)
    parser.add_argument("--d-reg-every", type=int)
    parser.add_argument("--reg-batch-shrink", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--data-pipeline",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
