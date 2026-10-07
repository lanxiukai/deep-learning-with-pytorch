"""Train a compact progressive StyleGAN on 128x128 aligned CelebA.

The defaults target an NVIDIA B200 or B300: BF16 AMP, twice the teaching batch
size, a 2.2-million-image schedule, lazy full-FP32 R1, channels-last tensors,
fused Adam, and automatic CUDA JPEG preprocessing.  They aim for recognizable
reference samples rather than paper-level fidelity.

Fresh runs reset ``output/gan/stylegan/training`` and ``checkpoints``.  The
latest full-state checkpoint is replaced atomically at each progressive phase
boundary, matching the reference lesson, together with a fixed-latent/noise EMA
sample grid.

Run:
    uv run --locked --no-sync python \
        visual_generation/1.0_generative_adversarial_network/7.1_stylegan.py

Resume:
    uv run --locked --no-sync python \
        visual_generation/1.0_generative_adversarial_network/7.1_stylegan.py \
        --resume-from output/gan/stylegan/checkpoints/latest.pth

Default schedule:
    Progressive phases:      11 x 200 kimg
    Real images shown to D:  2.2 million
    Main D/G updates:        24,222 / 24,222

Default image sizes:
    Training input:          4x4 to 128x128 RGB, progressive by phase
    Generated image:         4x4 to 128x128 RGB, progressive by phase

Model size:
    Generator:               4.18 M parameters (plus one EMA copy)
    Discriminator:           4.59 M parameters
"""

import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.gan.artifacts import save_gan_samples
from dl_utils.gan.continuation import continue_gan
from dl_utils.gan.stylegan import (
    StyleGANDiscriminator,
    StyleGANGenerator,
    add_refinement_arguments,
    make_continuation_plan,
)
from dl_utils.gan.stylegan_layers import RESOLUTIONS
from dl_utils.gan.training import (
    append_gan_metrics,
    build_progressive_schedule,
    initialize_gan_models,
    phase_alpha,
    prepare_gan_run,
    r1_penalty,
    resolve_progressive_gan_options,
    sample_mixing_latents,
    start_gan_checkpoint,
)
from dl_utils.plot.curves import save_loss_panels
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.ema import update_ema_by_images
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.optimization import make_fused_adam
from dl_utils.training.validation import validate_finite_training_state

BATCH_SIZES = {4: 256, 8: 256, 16: 128, 32: 64, 64: 64, 128: 64}
PHASE_KIMG = 200
NUM_WORKERS = 8
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 16
SEED = 42
D_REG_EVERY = 16
REG_BATCH_SHRINK = 2

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
}
DISCRIMINATOR_CONFIG = {"base_channels": BASE_CHANNELS}
METRIC_NAMES = ("loss_d_main", "loss_g_main", "r1_penalty")


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
    r1_gamma=R1_GAMMA,
):
    """Train one progressive phase with BF16 main passes and FP32 R1."""
    metrics = MetricAccumulator(METRIC_NAMES, device=data.device)
    progress = tqdm(
        range(phase.num_batches),
        desc=f"StyleGAN {phase.resolution}x {phase.name}",
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
        z, mixing_z = sample_mixing_latents(
            batch_size,
            generator.z_dim,
            data.device,
            STYLE_MIXING_PROBABILITY,
        )
        with torch.no_grad(), precision.autocast():
            fake_images = generator(
                z,
                resolution=phase.resolution,
                alpha=alpha,
                mixing_z=mixing_z,
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
            loss_d_main = (
                F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()
            )
        weighted_r1 = real_images.new_zeros(())
        if global_step % d_reg_every == 0:
            reg_batch = max(1, batch_size // reg_batch_shrink)
            real_for_r1 = real_images[:reg_batch].detach().float().requires_grad_(True)
            penalty_r1 = r1_penalty(
                discriminator(
                    real_for_r1,
                    resolution=phase.resolution,
                    alpha=alpha,
                ),
                real_for_r1,
            )
            weighted_r1 = 0.5 * r1_gamma * d_reg_every * penalty_r1
        precision.backward_step(
            loss_d_main.float() + weighted_r1,
            optimizer_d,
        )

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z, mixing_z = sample_mixing_latents(
                batch_size,
                generator.z_dim,
                data.device,
                STYLE_MIXING_PROBABILITY,
            )
            with precision.autocast():
                fake_images = generator(
                    z,
                    resolution=phase.resolution,
                    alpha=alpha,
                    mixing_z=mixing_z,
                    update_w_avg=True,
                )
                loss_g_main = F.softplus(
                    -discriminator(
                        fake_images,
                        resolution=phase.resolution,
                        alpha=alpha,
                    )
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
            (loss_d_main, loss_g_main, weighted_r1),
            num_examples=batch_size,
        )
        global_step += 1

    return metrics.compute_weighted_means(require_finite=True), global_step


def main(args):
    if args.refine_from is not None or args.refine_resume is not None:
        if args.resume_from is not None:
            raise ValueError("Use only one progressive or refinement resume mode.")
        continue_gan(
            args,
            plan=make_continuation_plan(
                args,
                model_config=MODEL_CONFIG,
                discriminator_config=DISCRIMINATOR_CONFIG,
                train_phase=train_phase,
                d_reg_every=D_REG_EVERY,
            ),
        )
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
        "stylegan",
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
        StyleGANGenerator(**MODEL_CONFIG),
        StyleGANDiscriminator(**DISCRIMINATOR_CONFIG),
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
        "stylegan_generator": averaged_generator,
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
                noise_mode="fixed",
            ),
            batch_size=SAMPLE_BATCH_SIZE,
            nrow=SAMPLE_GRID_COLUMNS,
            title=(
                f"StyleGAN EMA fixed samples - {phase.resolution}x"
                f"{phase.resolution} {phase.name} - {seen_kimg:.0f} kimg"
            ),
        )

    save_model_weights(
        averaged_generator,
        run.output_dir / "stylegan_generator.pth",
        metadata={"model_name": "stylegan", "model_config": MODEL_CONFIG},
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D logistic": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G non-saturating": loss_history["loss_g_main"],
            },
            "R1 regularization": {
                "Expected weighted contribution": loss_history["r1_penalty"],
            },
        },
        run.output_dir / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the progressive 128x128 CelebA StyleGAN."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    add_refinement_arguments(parser)
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
