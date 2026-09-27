"""FSQ: train a 256px tokenizer, then an unconditional PixelCNN.

Use the same 4,500 glasses-256 faces as the standard Gaussian VAE.
Both stages ignore folder labels. VQ-VAE and FSQ share four downsampling
steps, a 16x16 token grid, 512 possible codes, MSE reconstruction, and the
same prior budget. Latent quantization MSE is a within-model diagnostic.

Edit the constants below. RESUME continues an epoch-boundary checkpoint;
set TRAIN_TOKENIZER=False to train only the prior from selected weights.
A fixed 256-image training subset monitors progress and selects snapshots;
the stored val_* fields are training-set diagnostics, not held-out results.
A prior is bound to its exact tokenizer snapshot.
Run 6.2 for simple VAE comparisons.

Outputs under output/vae/fsq/:
    tokenizer/{latest,best,last}.pth and metrics.csv
    prior/{latest,best,last}.pth and metrics.csv
    fsq.pth and pixelcnn_prior.pth: selected, paired evaluation weights
    token_cache_*.pth: compact frozen tokens; training/*.png: fixed previews
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.artifacts import save_training_metrics
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.quantization import (
    FSQAutoencoder,
    TokenUsageAccumulator,
)
from dl_utils.vae.token_prior import (
    PixelCNNPrior,
    evaluate_pixelcnn_prior,
    sample_pixelcnn_prior_images,
    train_pixelcnn_prior_epoch,
)
from dl_utils.vae.tokenizer_workflow import (
    TokenizerStage,
    cached_token_loader,
    glasses_loader,
    image_contract,
    load_tokenizer_weights,
    seed_epoch_loader,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "fsq"
TOKENIZER_CHECKPOINT_NAME = "fsq.pth"
PRIOR_CHECKPOINT_NAME = "pixelcnn_prior.pth"
IMAGE_SIZE = 256
DOWNSAMPLE_STEPS = 4
LATENT_GRID_SIZE = IMAGE_SIZE // (2**DOWNSAMPLE_STEPS)
TOKENS_PER_IMAGE = LATENT_GRID_SIZE**2

# Edit these defaults to explore the lesson.
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
TRAIN_TOKENIZER = True
RESUME = True  # Continue from the latest epoch when a checkpoint exists.
TOKENIZER_EPOCHS = 100
PRIOR_EPOCHS = 100
HIDDEN_CHANNELS = 128
LEVELS = (8, 8, 8)  # 512 combinations, matching VQ-VAE exactly.
PRIOR_HIDDEN_CHANNELS = 64
PRIOR_LAYERS = 16  # The two masked streams cover the complete 16x16 past.
BATCH_SIZE = 16
LR = 2e-4
PRIOR_LR = 2e-4
TEMPERATURE = 1.0
SAMPLE_EVERY = 10
LOG_EVERY = 100
MONITOR_EXAMPLES = 256
MONITOR_SEED = 123
WORKERS = 4
SEED = 42
RECONSTRUCTION_SAMPLES = 8
NUM_FIXED_SAMPLES = 18
SAMPLE_GRID_COLUMNS = 6
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4


@torch.inference_mode()
def evaluate_tokenizer(model, loader, *, device) -> dict[str, float]:
    model.eval()
    vocabulary_size = model.quantizer.codebook_size
    metrics = MetricAccumulator(
        ("mse", "mean_psnr_db", "quantization_mse"), device=device
    )
    usage = TokenUsageAccumulator(vocabulary_size)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction, indices, diagnostics = model(images)
        per_image_mse = (reconstruction - images).square().flatten(1).mean(1)
        metrics.add_batch_means(
            (
                per_image_mse.mean(),
                (10 * torch.log10(4 / per_image_mse.clamp_min(1e-12))).mean(),
                diagnostics["quantization_mse"],
            ),
            num_examples=images.shape[0],
        )
        usage.update(indices)
    means = metrics.compute_weighted_means(require_finite=True)
    statistics = usage.statistics()
    entropy_bits = statistics["token_entropy_nats"].item() / math.log(2)
    return {
        **means,
        "psnr_from_pooled_mse_db": 10 * math.log10(4 / max(means["mse"], 1e-12)),
        "perplexity": statistics["perplexity"].item(),
        "active_codes": statistics["active_codes"].item(),
        "usage_fraction": statistics["usage_fraction"].item(),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": TOKENS_PER_IMAGE * entropy_bits,
        "fixed_length_bits_per_image": TOKENS_PER_IMAGE
        * math.ceil(math.log2(vocabulary_size)),
    }


def train_tokenizer(train_loader, monitor_loader, device, out_dir, monitor_protocol):
    config = {
        "levels": LEVELS,
        "hidden_channels": HIDDEN_CHANNELS,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = FSQAutoencoder(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    stage = TokenizerStage(
        out_dir / "tokenizer",
        models={"model": model},
        optimizers={"model": optimizer},
        metadata={
            **image_contract(IMAGE_SIZE, dataset="glasses-256"),
            "model_name": "fsq_tokenizer",
            "model_config": config,
            "monitor_protocol": monitor_protocol,
            "selection_metric": "training_subset_mse",
        },
        recipe={"lr": LR, "batch_size": BATCH_SIZE, "seed": SEED},
        resume=RESUME,
    )
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for epoch in stage.epochs(TOKENIZER_EPOCHS):
        if stage.needs_training(epoch):
            model.train()
            seed_epoch_loader(train_loader, SEED, epoch)
            metrics = MetricAccumulator(("mse", "quantization_mse"), device=device)
            usage = TokenUsageAccumulator(model.quantizer.codebook_size)
            with tqdm(
                train_loader,
                desc=f"fsq {epoch}/{TOKENIZER_EPOCHS}",
                unit="batch",
                mininterval=PROGRESS_INTERVAL,
            ) as progress:
                for batch_index, (images, _) in enumerate(progress, 1):
                    images = images.to(device, non_blocking=True)
                    reconstruction, indices, diagnostics = model(images)
                    loss = F.mse_loss(reconstruction, images)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    metrics.add_batch_means(
                        (loss, diagnostics["quantization_mse"]),
                        num_examples=images.shape[0],
                    )
                    usage.update(indices)
                    if batch_index % LOG_EVERY == 0:
                        progress.set_postfix(
                            metrics.compute_weighted_means(require_finite=True),
                            refresh=False,
                        )
            means = metrics.compute_weighted_means(require_finite=True)
            statistics = usage.statistics()
            stage.record_training(
                epoch,
                {
                    **means,
                    "perplexity": statistics["perplexity"].item(),
                    "active_codes": statistics["active_codes"].item(),
                    "entropy_bits": statistics["token_entropy_nats"].item()
                    / math.log(2),
                },
            )
        # latest.pth is already durable if monitoring or plotting fails.
        validation = evaluate_tokenizer(model, monitor_loader, device=device)
        stage.record_validation(epoch, validation, score=validation["mse"])
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
            with torch.inference_mode():
                images = next(iter(monitor_loader))[0][:RECONSTRUCTION_SAMPLES].to(
                    device
                )
                reconstruction = model(images)[0]
                save_image(
                    torch.cat((images, reconstruction)).mul(0.5).add(0.5),
                    training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                    nrow=len(images),
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
    tokenizer_id = tokenizer_payload["snapshot_id"]
    cache_metadata = {
        **image_contract(IMAGE_SIZE, dataset="glasses-256"),
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
        "hidden_channels": PRIOR_HIDDEN_CHANNELS,
        "layers": PRIOR_LAYERS,
        "num_classes": 0,
    }
    prior = PixelCNNPrior(**config).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=PRIOR_LR)
    stage = TokenizerStage(
        out_dir / "prior",
        models={"model": prior},
        optimizers={"model": optimizer},
        metadata={
            **image_contract(IMAGE_SIZE, dataset="glasses-256"),
            "model_name": "fsq_pixelcnn_prior",
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
            seed_epoch_loader(tokens, SEED, epoch)
            with tqdm(
                total=len(tokens),
                desc=f"PixelCNN {epoch}/{PRIOR_EPOCHS}",
                unit="batch",
                mininterval=PROGRESS_INTERVAL,
            ) as progress:
                nll = train_pixelcnn_prior_epoch(
                    prior,
                    tokens,
                    optimizer,
                    device,
                    progress=progress,
                    log_every=LOG_EVERY,
                )
            stage.record_training(
                epoch, {"nll": nll, "bits_per_token": nll / math.log(2)}
            )
        validation = evaluate_pixelcnn_prior(
            prior,
            monitor_tokens,
            tokens_per_image=TOKENS_PER_IMAGE,
            device=device,
        )
        stage.record_validation(
            epoch, validation, score=validation["nll_nats_per_token"]
        )
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == PRIOR_EPOCHS:
            # Preview randomness must not change the resumed optimization stream.
            with torch.random.fork_rng():
                torch.manual_seed(SEED)
                prior.eval()
                samples = sample_pixelcnn_prior_images(
                    tokenizer,
                    prior,
                    NUM_FIXED_SAMPLES,
                    grid_size=LATENT_GRID_SIZE,
                    device=device,
                    temperature=TEMPERATURE,
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
    )
    monitor_loader, protocol = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        max_examples=MONITOR_EXAMPLES,
        seed=MONITOR_SEED,
        num_workers=WORKERS,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_TOKENIZER:
        tokenizer, payload = train_tokenizer(
            train_loader, monitor_loader, device, OUTPUT_DIR, protocol
        )
    else:
        tokenizer, payload = load_tokenizer_weights(
            OUTPUT_DIR / TOKENIZER_CHECKPOINT_NAME,
            FSQAutoencoder,
            name="fsq_tokenizer",
            image_size=IMAGE_SIZE,
            device=device,
            dataset="glasses-256",
            downsample_steps=DOWNSAMPLE_STEPS,
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
