"""Train a perceptual VQ tokenizer, then its frozen-token Transformer prior.

Stage one combines pixel L1, frozen LPIPS, VQ commitment loss,
and a delayed PatchGAN hinge objective. After the delay, an adaptive
last-decoder-layer gradient ratio balances reconstruction and adversarial loss.
The codebook uses EMA updates independently of the optimizer.
Stage two fits a class-conditional causal Transformer to cached frozen tokens.
The tokenizer reconstructs without labels; the prior uses G=0 / NoG=1 labels.

This teaching recipe uses 256px glasses-256 faces
and downsamples four times to the same 16x16 token grid and 512-entry
vocabulary as VQ-VAE/FSQ: 256 tokens, or 2,304 fixed-length bits per image.
Batch size 16 and 30 epochs per stage remain a separate training budget.
Different conditioning, backbones, objectives and priors prevent a strict ablation.
Prior preview columns alternate G (with glasses) and NoG (without glasses).

Edit RESUME and TRAIN_TOKENIZER below for recovery or prior-only training.
Each stage saves latest full state before validation, best/last weights, and
per-epoch CSV metrics. The selected tokenizer/prior snapshots are bound by ID.
A recorded seeded training subset monitors progress and selects snapshots;
the stored validation / val_* fields are training-set diagnostics.
Run 7.1 for reconstruction and generation diagnostics on the training images.

Outputs under output/vae/vqgan/:
    tokenizer/{latest,best,last}.pth and metrics.csv
    prior/{latest,best,last}.pth and metrics.csv
    vqgan.pth and transformer_prior.pth: selected evaluation weights
    token_cache_*.pth and training/*.png
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.training import discriminator_hinge_loss, generator_hinge_loss
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.artifacts import save_training_metrics
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_workflow import (
    TokenizerStage,
    TokenUsageAccumulator,
    cached_token_loader,
    glasses_loader,
    image_contract,
    load_tokenizer_weights,
    save_tokenizer_preview,
    seed_epoch_loader,
)
from dl_utils.vae.perceptual_autoencoder import (
    LPIPSPerceptualLoss,
    PatchDiscriminator,
    VQPerceptualAutoencoder,
    adaptive_adversarial_weight,
)
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS
from dl_utils.vae.transformer_prior import CausalTransformerPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vqgan"
TOKENIZER_CHECKPOINT_NAME = "vqgan.pth"
PRIOR_CHECKPOINT_NAME = "transformer_prior.pth"
RECONSTRUCTION_SAMPLES = 8
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4
ADAM_BETAS = (0.5, 0.9)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
IMAGE_SIZE = 256
NUM_CLASSES = len(GLASSES_CLASS_NAMES)
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS
LATENT_GRID_SIZE = IMAGE_SIZE // (2**DOWNSAMPLE_STEPS)
TOKENS_PER_IMAGE = LATENT_GRID_SIZE**2
NUM_FIXED_SAMPLES = 8
SAMPLE_GRID_COLUMNS = 4


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
TRAIN_TOKENIZER = True
RESUME = True
LOG_EVERY = 100
MONITOR_SEED = 123
TOKENIZER_EPOCHS = 30
PRIOR_EPOCHS = 30
BATCH_SIZE = 16
HIDDEN_CHANNELS = 128
LATENT_CHANNELS = 64
CODEBOOK_SIZE = 512
COMMITMENT = 0.25
EMA_DECAY = 0.99
EMA_EPSILON = 1e-5
DISCRIMINATOR_CHANNELS = 64
PERCEPTUAL_WEIGHT = 1.0
VQ_WEIGHT = 1.0
DISCRIMINATOR_WEIGHT = 1.0
DISCRIMINATOR_START = 1000
PRIOR_DIM = 256
PRIOR_HEADS = 8
PRIOR_LAYERS = 4
PRIOR_DROPOUT = 0.0
LR = 2e-4
DISCRIMINATOR_LR = 2e-4
PRIOR_LR = 3e-4
TEMPERATURE = 1.0
SAMPLE_EVERY = 5
MONITOR_EXAMPLES = 1_024
WORKERS = 4
SEED = 42


def vqgan_autoencoder_step(
    model: VQPerceptualAutoencoder,
    discriminator: PatchDiscriminator,
    perceptual: nn.Module,
    images: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
    perceptual_weight: float,
    vq_weight: float,
    discriminator_weight: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    reconstruction, indices, vq_loss, _ = model(images)
    pixel_l1 = F.l1_loss(reconstruction, images)
    perceptual_loss = perceptual(reconstruction, images)
    reconstruction_objective = pixel_l1 + float(perceptual_weight) * perceptual_loss

    discriminator.requires_grad_(False)
    adversarial = images.new_zeros(())
    adversarial_scale = images.new_zeros(())
    if step >= discriminator_start:
        adversarial = generator_hinge_loss(discriminator(reconstruction))
        adversarial_scale = adaptive_adversarial_weight(
            reconstruction_objective,
            adversarial,
            model.decoder.last_layer,
            scale=discriminator_weight,
        )
    loss = (
        reconstruction_objective + vq_weight * vq_loss + adversarial_scale * adversarial
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    discriminator.requires_grad_(True)
    return (
        reconstruction.detach(),
        indices.detach(),
        {
            "autoencoder": loss.detach(),
            "pixel_l1": pixel_l1.detach(),
            "perceptual": perceptual_loss.detach(),
            "vq": vq_loss.detach(),
            "generator_adversarial": adversarial.detach(),
            "adversarial_scale": adversarial_scale.detach(),
        },
    )


def vqgan_discriminator_step(
    discriminator: PatchDiscriminator,
    images: Tensor,
    reconstruction: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    discriminator_start: int,
) -> dict[str, Tensor]:
    if step < discriminator_start:
        zero = images.new_zeros(())
        return {"discriminator": zero, "real_logit": zero, "fake_logit": zero}
    real_logits = discriminator(images)
    fake_logits = discriminator(reconstruction.detach())
    loss = 0.5 * discriminator_hinge_loss(real_logits, fake_logits)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "discriminator": loss.detach(),
        "real_logit": real_logits.mean().detach(),
        "fake_logit": fake_logits.mean().detach(),
    }


@torch.inference_mode()
def validate_tokenizer(
    model: VQPerceptualAutoencoder,
    discriminator: PatchDiscriminator,
    perceptual: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    discriminator.eval()
    perceptual.eval()
    vocabulary_size = model.quantizer.codebook_size
    metrics = MetricAccumulator(
        (
            "pixel_l1",
            "perceptual_distance",
            "quantization_mse",
            "real_logit",
            "reconstruction_logit",
        ),
        device=device,
    )
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(images)
        metrics.add_batch_means(
            (
                F.l1_loss(reconstruction, images),
                perceptual(reconstruction, images),
                diagnostics["quantization_mse"],
                discriminator(images).mean(),
                discriminator(reconstruction).mean(),
            ),
            num_examples=images.shape[0],
        )
        usage.update(indices)
        examples += images.shape[0]
    if examples == 0:
        raise ValueError("tokenizer validation observed no examples")
    values = metrics.compute_weighted_means()
    return {
        "examples": float(examples),
        **values,
        **usage.rate_metrics(TOKENS_PER_IMAGE),
    }


@torch.inference_mode()
def validate_prior(prior, loader, *, device) -> dict[str, float]:
    prior.eval()
    metrics = MetricAccumulator(("nll",), device=device)
    for indices, labels in loader:
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, targets = prior.teacher_forcing(indices, labels)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        metrics.add_batch_means((loss,), num_examples=indices.shape[0])
    nll = metrics.compute_weighted_means(require_finite=True)["nll"]
    return {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": TOKENS_PER_IMAGE * nll / math.log(2),
    }


def train_tokenizer(train_loader, monitor_loader, device, out_dir, monitor_protocol):
    config = {
        "latent_channels": LATENT_CHANNELS,
        "codebook_size": CODEBOOK_SIZE,
        "hidden_channels": HIDDEN_CHANNELS,
        "commitment": COMMITMENT,
        "ema_decay": EMA_DECAY,
        "ema_epsilon": EMA_EPSILON,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = VQPerceptualAutoencoder(**config).to(device)
    discriminator = PatchDiscriminator(DISCRIMINATOR_CHANNELS).to(device)
    perceptual = LPIPSPerceptualLoss().to(device)
    ae_optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=ADAM_BETAS)
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=ADAM_BETAS
    )
    stage = TokenizerStage(
        out_dir / "tokenizer",
        models={"model": model, "discriminator": discriminator},
        optimizers={"model": ae_optimizer, "discriminator": d_optimizer},
        metadata={
            **image_contract(IMAGE_SIZE, conditional=True),
            "model_name": "vqgan_tokenizer",
            "model_config": config,
            "discriminator_config": {"base_channels": DISCRIMINATOR_CHANNELS},
            "perceptual_loss": "lpips-v0.1-vgg",
            "monitor_protocol": monitor_protocol,
            "selection_metric": "training_subset_pixel_l1_plus_weighted_lpips",
        },
        recipe={
            "lr": LR,
            "d_lr": DISCRIMINATOR_LR,
            "betas": ADAM_BETAS,
            "perceptual_weight": PERCEPTUAL_WEIGHT,
            "vq_weight": VQ_WEIGHT,
            "discriminator_weight": DISCRIMINATOR_WEIGHT,
            "discriminator_start": DISCRIMINATOR_START,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        resume=RESUME,
    )
    global_step = stage.state.get("global_step", 0)
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for epoch in stage.epochs(TOKENIZER_EPOCHS):
        if stage.needs_training(epoch):
            model.train()
            discriminator.train()
            seed_epoch_loader(train_loader, SEED, epoch)
            names = (
                "autoencoder",
                "pixel_l1",
                "perceptual",
                "vq",
                "generator_adversarial",
                "adversarial_scale",
                "discriminator",
                "real_logit",
                "fake_logit",
            )
            metrics_accumulator = MetricAccumulator(names, device=device)
            usage = TokenUsageAccumulator(CODEBOOK_SIZE)
            with tqdm(
                train_loader,
                desc=f"vqgan {epoch}/{TOKENIZER_EPOCHS}",
                unit="batch",
                mininterval=PROGRESS_INTERVAL,
            ) as progress:
                for batch_index, (images, _) in enumerate(progress, 1):
                    images = images.to(device, non_blocking=True)
                    reconstruction, indices, metrics = vqgan_autoencoder_step(
                        model,
                        discriminator,
                        perceptual,
                        images,
                        ae_optimizer,
                        step=global_step,
                        discriminator_start=DISCRIMINATOR_START,
                        perceptual_weight=PERCEPTUAL_WEIGHT,
                        vq_weight=VQ_WEIGHT,
                        discriminator_weight=DISCRIMINATOR_WEIGHT,
                    )
                    d_metrics = vqgan_discriminator_step(
                        discriminator,
                        images,
                        reconstruction,
                        d_optimizer,
                        step=global_step,
                        discriminator_start=DISCRIMINATOR_START,
                    )
                    values = metrics | d_metrics
                    metrics_accumulator.add_batch_means(
                        tuple(values[name] for name in names),
                        num_examples=images.shape[0],
                    )
                    usage.update(indices)
                    global_step += 1
                    if batch_index % LOG_EVERY == 0:
                        running = metrics_accumulator.compute_weighted_means(
                            require_finite=True
                        )
                        progress.set_postfix(
                            ae=f"{running['autoencoder']:.4f}",
                            d=f"{running['discriminator']:.4f}",
                            refresh=False,
                        )
            stage.state["global_step"] = global_step
            stage.record_training(
                epoch,
                {
                    **metrics_accumulator.compute_weighted_means(require_finite=True),
                    **usage.training_metrics(),
                },
            )
        validation = validate_tokenizer(
            model,
            discriminator,
            perceptual,
            monitor_loader,
            device=device,
        )
        score = (
            validation["pixel_l1"]
            + PERCEPTUAL_WEIGHT * validation["perceptual_distance"]
        )
        stage.record_validation(epoch, validation, score=score)
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
            save_tokenizer_preview(
                model,
                monitor_loader,
                training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                count=RECONSTRUCTION_SAMPLES,
                device=device,
            )
    payload = stage.export_best(out_dir / TOKENIZER_CHECKPOINT_NAME)
    save_training_metrics(
        stage.history, out_dir, prefix="tokenizer", max_panels=MAX_METRIC_PANELS
    )
    return model.eval().requires_grad_(False), payload


def train_prior(
    tokenizer,
    tokenizer_payload,
    train_loader,
    monitor_loader,
    device,
    out_dir,
    monitor_protocol,
):
    tokenizer.eval().requires_grad_(False)
    tokenizer_grid_size = IMAGE_SIZE // (2**tokenizer.downsample_steps)
    if tokenizer_grid_size != LATENT_GRID_SIZE:
        raise ValueError(
            "The selected tokenizer grid must match the current training configuration."
        )
    tokenizer_id = tokenizer_payload["snapshot_id"]
    cache_metadata = {
        **image_contract(IMAGE_SIZE, conditional=True),
        "source_root": str(DATA_DIR.resolve()),
    }
    tokens = cached_token_loader(
        tokenizer,
        train_loader,
        out_dir / "token_cache_train.pth",
        tokenizer_id=tokenizer_id,
        metadata={**cache_metadata, "split": "train"},
        device=device,
        shuffle=True,
    )
    monitor_tokens = cached_token_loader(
        tokenizer,
        monitor_loader,
        out_dir / "token_cache_monitor.pth",
        tokenizer_id=tokenizer_id,
        metadata={**cache_metadata, "protocol": monitor_protocol},
        device=device,
        shuffle=False,
    )
    config = {
        "vocabulary_size": tokenizer.quantizer.codebook_size,
        "sequence_length": TOKENS_PER_IMAGE,
        "model_dim": PRIOR_DIM,
        "heads": PRIOR_HEADS,
        "layers": PRIOR_LAYERS,
        "dropout": PRIOR_DROPOUT,
        "num_classes": NUM_CLASSES,
    }
    prior = CausalTransformerPrior(**config).to(device)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=PRIOR_LR)
    stage = TokenizerStage(
        out_dir / "prior",
        models={"model": prior},
        optimizers={"model": optimizer},
        metadata={
            **image_contract(IMAGE_SIZE, conditional=True),
            "model_name": "vqgan_transformer_prior",
            "model_config": config,
            "tokenizer_id": tokenizer_id,
            "monitor_protocol": monitor_protocol,
            "selection_metric": "training_subset_nll",
        },
        recipe={"lr": PRIOR_LR, "batch_size": BATCH_SIZE, "seed": SEED},
        resume=RESUME,
    )
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for epoch in stage.epochs(PRIOR_EPOCHS):
        if stage.needs_training(epoch):
            prior.train()
            seed_epoch_loader(tokens, SEED, epoch)
            metrics = MetricAccumulator(("nll",), device=device)
            with tqdm(
                tokens,
                desc=f"Transformer {epoch}/{PRIOR_EPOCHS}",
                unit="batch",
                mininterval=PROGRESS_INTERVAL,
            ) as progress:
                for batch_index, (indices, labels) in enumerate(progress, 1):
                    indices = indices.to(
                        device=device, dtype=torch.long, non_blocking=True
                    )
                    labels = labels.to(device, non_blocking=True)
                    logits, targets = prior.teacher_forcing(indices, labels)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    metrics.add_batch_means((loss,), num_examples=indices.shape[0])
                    if batch_index % LOG_EVERY == 0:
                        nll = metrics.compute_weighted_means(require_finite=True)["nll"]
                        progress.set_postfix(
                            nll=f"{nll:.4f}",
                            bpt=f"{nll / math.log(2):.3f}",
                            refresh=False,
                        )
            nll = metrics.compute_weighted_means(require_finite=True)["nll"]
            stage.record_training(
                epoch, {"nll": nll, "bits_per_token": nll / math.log(2)}
            )
        validation = validate_prior(prior, monitor_tokens, device=device)
        stage.record_validation(
            epoch, validation, score=validation["nll_nats_per_token"]
        )
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == PRIOR_EPOCHS:
            with torch.random.fork_rng(), torch.inference_mode():
                torch.manual_seed(SEED)
                prior.eval()
                labels = torch.arange(NUM_FIXED_SAMPLES, device=device).remainder(
                    NUM_CLASSES
                )
                indices = prior.sample(
                    NUM_FIXED_SAMPLES,
                    device=device,
                    labels=labels,
                    temperature=TEMPERATURE,
                )
                samples = tokenizer.decode_indices(
                    indices.reshape(
                        NUM_FIXED_SAMPLES, LATENT_GRID_SIZE, LATENT_GRID_SIZE
                    )
                )
            save_image(
                samples.mul(0.5).add(0.5),
                training_dir / f"prior_epoch_{epoch:03d}.png",
                nrow=SAMPLE_GRID_COLUMNS,
            )
    stage.export_best(out_dir / PRIOR_CHECKPOINT_NAME)
    save_training_metrics(
        stage.history, out_dir, prefix="prior", max_panels=MAX_METRIC_PANELS
    )


def train() -> None:
    device = try_gpu()
    train_loader, _ = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        shuffle=True,
        seed=SEED,
        num_workers=WORKERS,
        conditional=True,
    )
    monitor_loader, protocol = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        max_examples=MONITOR_EXAMPLES,
        seed=MONITOR_SEED,
        num_workers=WORKERS,
        conditional=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_TOKENIZER:
        tokenizer, payload = train_tokenizer(
            train_loader, monitor_loader, device, OUTPUT_DIR, protocol
        )
    else:
        tokenizer, payload = load_tokenizer_weights(
            OUTPUT_DIR / TOKENIZER_CHECKPOINT_NAME,
            VQPerceptualAutoencoder,
            name="vqgan_tokenizer",
            image_size=IMAGE_SIZE,
            device=device,
            downsample_steps=DOWNSAMPLE_STEPS,
            conditional=True,
        )
    train_prior(
        tokenizer,
        payload,
        train_loader,
        monitor_loader,
        device,
        OUTPUT_DIR,
        protocol,
    )


def main() -> None:
    set_seed(SEED)
    train()


if __name__ == "__main__":
    main()
