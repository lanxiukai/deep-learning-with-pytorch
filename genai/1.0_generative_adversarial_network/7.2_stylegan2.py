"""Train a compact 128x128 StyleGAN2 on aligned CelebA.

The defaults target an NVIDIA B200 or B300: a 2,500-kimg budget, BF16 AMP, and
twice the teaching batch size while retaining lazy R1 and path-length
regularization.  Higher-order penalties stay in FP32 on reduced regularization
batches.  Fused Adam, channels-last tensors, and batched CUDA JPEG decoding
avoid requiring custom CUDA kernels.

Fresh runs reset ``output/gan/stylegan2/training`` and ``checkpoints``.  A
latest full-state checkpoint is written at every data-epoch boundary, matching
the reference lesson, together with a fixed-latent/noise EMA sample grid.

Run:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.2_stylegan2.py

Resume:
    uv run --locked --no-sync python \
        genai/1.0_generative_adversarial_network/7.2_stylegan2.py \
        --resume-from output/gan/stylegan2/checkpoints/latest.pth

Default schedule:
    Real images shown to D:  2.5 million
    Main D/G updates:        39,063 / 39,063

Default image sizes:
    Training input:          128x128 RGB
    Generated image:         128x128 RGB

Model size:
    Generator:               4.05 M parameters (plus one EMA copy)
    Discriminator:           4.76 M parameters
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.gan.artifacts import save_gan_samples
from dl_utils.gan.continuation import continue_gan
from dl_utils.gan.stylegan2 import (
    StyleDiscriminator,
    StyleGenerator,
    add_refinement_arguments,
    build_training_schedule,
    make_continuation_plan,
    path_length_penalty,
    resolve_fixed_resolution_gan_options,
)
from dl_utils.gan.training import (
    append_gan_metrics,
    initialize_gan_models,
    prepare_gan_run,
    r1_penalty,
    sample_mixing_latents,
    start_gan_checkpoint,
)
from dl_utils.plot.curves import save_loss_panels
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.ema import update_ema_by_images
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.optimization import make_fused_adam
from dl_utils.training.validation import validate_finite_training_state

BATCH_SIZE = 64
TOTAL_KIMG = 2_500
NUM_WORKERS = 8
Z_DIM = 128
STYLE_DIM = 128
BASE_CHANNELS = 32
MAPPING_LAYERS = 8
W_AVG_BETA = 0.995
LEARNING_RATE = 2e-3
STYLE_MIXING_PROBABILITY = 0.9
R1_GAMMA = 10.0
D_REG_EVERY = 16
PATH_WEIGHT = 2.0
G_REG_EVERY = 8
R1_BATCH_SHRINK = 2
PATH_BATCH_SHRINK = 4
EMA_HALF_LIFE_KIMG = 10.0
SAMPLES_TO_DISPLAY = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_BATCH_SIZE = 16
SEED = 42

MODEL_CONFIG = {
    "z_dim": Z_DIM,
    "style_dim": STYLE_DIM,
    "base_channels": BASE_CHANNELS,
    "mapping_layers": MAPPING_LAYERS,
    "w_avg_beta": W_AVG_BETA,
}
DISCRIMINATOR_CONFIG = {"base_channels": BASE_CHANNELS}
METRIC_NAMES = (
    "loss_d_main",
    "loss_g_main",
    "r1_penalty",
    "path_penalty",
)


def train_epoch(
    generator,
    discriminator,
    data,
    optimizer_g,
    optimizer_d,
    averaged_generator,
    path_mean,
    global_step,
    epoch,
    batch_size,
    precision,
    r1_batch_shrink,
    path_batch_shrink,
    *,
    r1_gamma=R1_GAMMA,
    d_reg_every=D_REG_EVERY,
    g_reg_every=G_REG_EVERY,
    path_weight=PATH_WEIGHT,
):
    """Train one data epoch with separate lazy penalties."""
    metrics = MetricAccumulator(METRIC_NAMES, device=data.device)
    progress = tqdm(
        range(epoch.num_batches),
        desc="StyleGAN2 128x",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for batch_index in progress:
        real = data.next_batch(
            128,
            batch_size,
            limit=epoch.batch_size_at(batch_index, batch_size),
        )
        current_batch = len(real)

        optimizer_d.zero_grad(set_to_none=True)
        z, mixing_z = sample_mixing_latents(
            current_batch,
            generator.z_dim,
            data.device,
            STYLE_MIXING_PROBABILITY,
        )
        with torch.no_grad(), precision.autocast():
            fake = generator(z, mixing_z=mixing_z)
        with precision.autocast():
            real_scores = discriminator(real)
            fake_scores = discriminator(fake)
            loss_d_main = (
                F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()
            )
        precision.backward_step(loss_d_main, optimizer_d)

        weighted_r1 = real.new_zeros(())
        if global_step % d_reg_every == 0:
            optimizer_d.zero_grad(set_to_none=True)
            reg_batch = max(1, current_batch // r1_batch_shrink)
            real_for_r1 = real[:reg_batch].detach().float().requires_grad_(True)
            penalty_r1 = r1_penalty(
                discriminator(real_for_r1),
                real_for_r1,
            )
            weighted_r1 = 0.5 * r1_gamma * d_reg_every * penalty_r1
            precision.backward_step(weighted_r1, optimizer_d)

        discriminator.requires_grad_(False)
        try:
            optimizer_g.zero_grad(set_to_none=True)
            z, mixing_z = sample_mixing_latents(
                current_batch,
                generator.z_dim,
                data.device,
                STYLE_MIXING_PROBABILITY,
            )
            with precision.autocast():
                fake = generator(
                    z,
                    mixing_z=mixing_z,
                    update_w_avg=True,
                )
                loss_g_main = F.softplus(-discriminator(fake)).mean()
            precision.backward_step(loss_g_main, optimizer_g)

            weighted_path = real.new_zeros(())
            if global_step % g_reg_every == 0:
                optimizer_g.zero_grad(set_to_none=True)
                path_batch = max(1, current_batch // path_batch_shrink)
                path_z = torch.randn(
                    path_batch,
                    generator.z_dim,
                    device=data.device,
                )
                path_images, path_ws = generator(path_z, return_ws=True)
                penalty_path, path_mean = path_length_penalty(
                    path_images,
                    path_ws,
                    path_mean,
                )
                weighted_path = path_weight * g_reg_every * penalty_path
                precision.backward_step(weighted_path, optimizer_g)
            update_ema_by_images(
                averaged_generator,
                generator,
                current_batch,
                EMA_HALF_LIFE_KIMG,
            )
        finally:
            discriminator.requires_grad_(True)

        metrics.add_batch_means(
            (loss_d_main, loss_g_main, weighted_r1, weighted_path),
            num_examples=current_batch,
        )
        global_step += 1

    return metrics.compute_weighted_means(require_finite=True), path_mean, global_step


def main(args):
    if args.refine_from is not None or args.refine_resume is not None:
        if args.resume_from is not None:
            raise ValueError("Use only one fixed-resolution or refinement resume mode.")
        continue_gan(
            args,
            plan=make_continuation_plan(
                args,
                model_config=MODEL_CONFIG,
                discriminator_config=DISCRIMINATOR_CONFIG,
                train_epoch=train_epoch,
                d_reg_every=D_REG_EVERY,
                g_reg_every=G_REG_EVERY,
            ),
        )
        return
    options = resolve_fixed_resolution_gan_options(
        total_kimg=args.total_kimg,
        batch_scale=args.batch_scale,
        r1_batch_shrink=args.r1_batch_shrink,
        path_batch_shrink=args.path_batch_shrink,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        base_batch_size=BATCH_SIZE,
        default_total_kimg=TOTAL_KIMG,
        default_r1_batch_shrink=R1_BATCH_SHRINK,
        default_path_batch_shrink=PATH_BATCH_SHRINK,
        default_num_workers=NUM_WORKERS,
    )
    run = prepare_gan_run(
        "stylegan2",
        seed=SEED,
        data_pipeline=args.data_pipeline,
        num_workers=options.num_workers,
        prefetch_factor=options.prefetch_factor,
    )
    if run.dataset_size < options.batch_size:
        raise ValueError("CelebA train split is smaller than a training batch.")
    training_schedule = build_training_schedule(
        options.total_kimg,
        options.batch_size,
        run.dataset_size,
    )
    total_epochs = len(training_schedule)
    total_batches = sum(epoch.num_batches for epoch in training_schedule)
    total_images = sum(epoch.num_images for epoch in training_schedule)
    run.prepare_output(args.resume_from)

    generator, discriminator, averaged_generator = initialize_gan_models(
        StyleGenerator(**MODEL_CONFIG),
        StyleDiscriminator(**DISCRIMINATOR_CONFIG),
        run.device,
    )

    g_ratio = G_REG_EVERY / (G_REG_EVERY + 1)
    d_ratio = D_REG_EVERY / (D_REG_EVERY + 1)
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=run.device,
        lr=LEARNING_RATE * g_ratio,
        betas=(0.0, 0.99**g_ratio),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=run.device,
        lr=LEARNING_RATE * d_ratio,
        betas=(0.0, 0.99**d_ratio),
    )
    print(
        f"precision={run.precision.name} pipeline={run.pipeline} "
        f"workers={options.num_workers} "
        f"total_kimg={options.total_kimg} "
        f"batch={options.batch_size}"
    )

    models = {
        "online_generator": generator,
        "stylegan2_generator": averaged_generator,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    run_config = {
        "precision": run.precision.name,
        "data_pipeline": run.pipeline,
        "dataset_size": run.dataset_size,
        "total_kimg": options.total_kimg,
        "batch_size": options.batch_size,
        "r1_batch_shrink": options.r1_batch_shrink,
        "path_batch_shrink": options.path_batch_shrink,
    }
    checkpoint, completed_epochs, state = start_gan_checkpoint(
        run.checkpoint_dir / "latest.pth",
        resume_from=args.resume_from,
        unit="fixed-resolution-epoch-main-metrics-v2",
        models=models,
        optimizers=optimizers,
        metric_names=METRIC_NAMES,
        fixed_z=torch.randn(SAMPLES_TO_DISPLAY, Z_DIM, device=run.device),
        run_config=run_config,
        extra_state={
            "path_mean": torch.zeros(()),
            "global_step": 0,
            "seen_images": 0,
        },
    )
    if completed_epochs > total_epochs:
        raise ValueError("Checkpoint epoch exceeds the training schedule.")
    if args.resume_from is not None:
        print(f"Resumed after epoch {completed_epochs}: {args.resume_from}")

    loss_history = state["loss_history"]
    fixed_z = state["fixed_z"].to(run.device)
    path_mean = state["path_mean"].to(run.device)
    global_step = int(state["global_step"])
    seen_images = int(state["seen_images"])
    completed_schedule = training_schedule[:completed_epochs]
    if global_step != sum(epoch.num_batches for epoch in completed_schedule):
        raise ValueError("Checkpoint global step does not match its epoch.")
    if seen_images != sum(epoch.num_images for epoch in completed_schedule):
        raise ValueError("Checkpoint image count does not match its epoch.")

    for epoch_index, epoch in enumerate(
        training_schedule[completed_epochs:],
        start=completed_epochs + 1,
    ):
        print(f"Epoch {epoch_index}/{total_epochs}")
        metrics, path_mean, global_step = train_epoch(
            generator,
            discriminator,
            run.data,
            optimizer_g,
            optimizer_d,
            averaged_generator,
            path_mean,
            global_step,
            epoch,
            options.batch_size,
            run.precision,
            options.r1_batch_shrink,
            options.path_batch_shrink,
        )
        validate_finite_training_state(
            models,
            optimizers,
            extra_tensors={"path_mean": path_mean},
        )
        seen_images += epoch.num_images
        seen_kimg = seen_images / 1_000
        append_gan_metrics(loss_history, seen_kimg, metrics)

        state["path_mean"] = path_mean.detach().cpu()
        state["global_step"] = global_step
        state["seen_images"] = seen_images
        checkpoint.save(epoch_index, state)
        save_gan_samples(
            averaged_generator,
            fixed_z,
            run.training_dir / f"kimg_{round(seen_kimg):05d}.png",
            precision=run.precision,
            generate=lambda z: averaged_generator(z, noise_mode="fixed"),
            batch_size=SAMPLE_BATCH_SIZE,
            nrow=SAMPLE_GRID_COLUMNS,
            title=f"StyleGAN2 EMA fixed samples - {seen_kimg:.0f} kimg",
        )

    if global_step != total_batches or seen_images != total_images:
        raise RuntimeError("Training ended with an unexpected image budget.")
    save_model_weights(
        averaged_generator,
        run.output_dir / "stylegan2_generator.pth",
        metadata={"model_name": "stylegan2", "model_config": MODEL_CONFIG},
    )
    save_loss_panels(
        loss_history["kimg"],
        {
            "Discriminator main objective": {
                "D logistic loss": loss_history["loss_d_main"],
            },
            "Generator main objective": {
                "G non-saturating loss": loss_history["loss_g_main"],
            },
            "R1 regularization": {
                "Expected weighted contribution": loss_history["r1_penalty"],
            },
            "Path length regularization": {
                "Expected weighted contribution": loss_history["path_penalty"],
            },
        },
        run.output_dir / "loss_curves.png",
        xlabel="thousands of real images shown to the discriminator (kimg)",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the full-resolution 128x128 CelebA StyleGAN2."
    )
    parser.add_argument("--resume-from", type=Path, metavar="CHECKPOINT")
    add_refinement_arguments(parser)
    parser.add_argument("--total-kimg", type=int)
    parser.add_argument("--batch-scale", type=int)
    parser.add_argument("--r1-batch-shrink", type=int)
    parser.add_argument("--path-batch-shrink", type=int)
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
