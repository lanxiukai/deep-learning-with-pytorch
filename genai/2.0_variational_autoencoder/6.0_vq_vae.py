"""VQ-VAE: learn a discrete tokenizer, then learn its token prior.

Stage 1 is the original three-way VQ objective:

    reconstruction + codebook loss + commitment loss

The decoder sees a nearest code in the forward pass, while the straight-
through estimator copies its gradient to the encoder. Stage 2 freezes the
entire tokenizer and trains a class-conditional causal PixelCNN on the 16x16
integer grid. Thus reconstruction and generation remain visibly different
paths. Under the stage-one fixed uniform prior, every position's KL is the
parameter-independent constant ``log(codebook_size)`` and is omitted.

Three stride-2 encoder blocks compress each 128x128 image to 16x16 tokens.
This teaching-scale compromise keeps ancestral PixelCNN sampling at 256
positions without widening the original channel and codebook defaults.

Training runs the tokenizer first, then freezes it and trains the prior.
Both final weight files are saved in the same directory for evaluation.

Data:
    data/celeba (official train and validation splits), prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Outputs:
    output/vae/vq_vae/tokenizer_*.png: reconstruction comparisons
    output/vae/vq_vae/prior_*.png: one PixelCNN sample per Smiling label
    output/vae/vq_vae/vq_vae.pth: final tokenizer checkpoint
    output/vae/vq_vae/pixelcnn_prior.pth: final token-prior checkpoint

Training data -- CelebA-128:
Training images:             162,770
Validation images:            19,867
Batch size:                       64
Samples per epoch:           162,752 (2,543 full batches; drop_last=True)
Tokenizer epochs:                 20
Prior epochs:                     20
Optimizer updates:            50,860 tokenizer / 50,860 prior
The tokenizer remains label-free; the frozen-token PixelCNN uses the binary
Smiling attribute. The final 18 shuffled images are omitted per epoch.
Both splits center-crop aligned faces to 178x178, resize to 128x128, and use
no random horizontal flips.

Default dimensions:
Training input:             128x128 RGB
Generated image:            128x128 RGB
Latent token grid:            16x16 indices (512-entry codebook)

Model size:
VQ-VAE tokenizer:              1.71 M parameters
Conditional PixelCNN prior:    1.84 M parameters (tokenizer frozen)
Stored total:                  3.55 M parameters

Run this script without arguments to train both stages in order.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.celeba import (
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    make_aligned_celeba_train_validation_loaders,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.artifacts import save_training_metrics
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.quantization import VQVAE, TokenUsageAccumulator
from dl_utils.vae.token_prior import (
    PixelCNNPrior,
    evaluate_pixelcnn_prior,
    make_fixed_class_labels,
    sample_pixelcnn_prior_images,
    train_pixelcnn_prior_epoch,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vq_vae"
TOKENIZER_CHECKPOINT_NAME = "vq_vae.pth"
PRIOR_CHECKPOINT_NAME = "pixelcnn_prior.pth"
RECONSTRUCTION_SAMPLES = 8
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGE_SIZE = 128
NUM_CLASSES = len(CELEBA_SMILING_CLASSES)
DOWNSAMPLE_STEPS = 3
LATENT_GRID_SIZE = IMAGE_SIZE // (2**DOWNSAMPLE_STEPS)
TOKENS_PER_IMAGE = LATENT_GRID_SIZE**2
SAMPLES_PER_CLASS = 1


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
TOKENIZER_EPOCHS = 20
PRIOR_EPOCHS = 20
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 64
CODEBOOK_SIZE = 512
COMMITMENT = 0.25
PRIOR_HIDDEN_CHANNELS = 128
PRIOR_LAYERS = 7
BATCH_SIZE = 64
LR = 2e-4
PRIOR_LR = 2e-4
TEMPERATURE = 1.0
SAMPLE_EVERY = 5
WORKERS = 4
SEED = 42


def tokenizer_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    quantizer_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distortion = F.mse_loss(reconstruction, images)
    return distortion + quantizer_loss, distortion.detach()


@torch.inference_mode()
def evaluate_tokenizer(
    model: VQVAE,
    loader: DataLoader,
    *,
    vocabulary_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator(("mse", "quantization_mse"), device=device)
    usage = TokenUsageAccumulator(vocabulary_size)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(images)
        metrics.add_batch_means(
            (F.mse_loss(reconstruction, images), diagnostics["quantization_mse"]),
            num_examples=images.shape[0],
        )
        usage.update(indices)
    means = metrics.compute_weighted_means()
    statistics = usage.statistics()
    entropy_bits = statistics["token_entropy_nats"].item() / math.log(2)
    return {
        **means,
        "perplexity": statistics["perplexity"].item(),
        "active_codes": statistics["active_codes"].item(),
        "usage_fraction": statistics["usage_fraction"].item(),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": TOKENS_PER_IMAGE * entropy_bits,
        "fixed_length_bits_per_image": (
            TOKENS_PER_IMAGE * math.ceil(math.log2(vocabulary_size))
        ),
    }


def train_tokenizer(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> VQVAE:
    config = {
        "hidden_channels": HIDDEN_CHANNELS,
        "embedding_dim": EMBEDDING_DIM,
        "codebook_size": CODEBOOK_SIZE,
        "commitment": COMMITMENT,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = VQVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    training_dir = out_dir / "training"
    if not training_dir.exists():
        reset_dir(str(training_dir))
    history = []
    with tqdm(
        total=TOKENIZER_EPOCHS * len(train_loader),
        desc=f"vq_vae 1/{TOKENIZER_EPOCHS}",
        unit="batch",
        mininterval=PROGRESS_INTERVAL,
    ) as progress:
        for epoch in range(1, TOKENIZER_EPOCHS + 1):
            progress.set_description(f"vq_vae {epoch}/{TOKENIZER_EPOCHS}", refresh=False)
            model.train()
            metrics = MetricAccumulator(
                ("loss", "mse", "codebook_loss", "commitment_loss"), device=device
            )
            usage = TokenUsageAccumulator(CODEBOOK_SIZE)
            preview = None
            for images, _ in train_loader:
                images = images.to(device, non_blocking=True)
                reconstruction, indices, quantizer_loss, diagnostics = model(images)
                loss, distortion = tokenizer_loss(reconstruction, images, quantizer_loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics.add_batch_means(
                    (
                        loss,
                        distortion,
                        diagnostics["codebook_loss"],
                        diagnostics["commitment_loss"],
                    ),
                    num_examples=images.shape[0],
                )
                usage.update(indices)
                progress.set_postfix(
                    loss=f"{metrics.compute_weighted_means()['loss']:.4f}",
                    refresh=False,
                )
                progress.update(1)
                preview = (
                    images[:RECONSTRUCTION_SAMPLES].detach(),
                    reconstruction[:RECONSTRUCTION_SAMPLES].detach(),
                )
            if preview is None:
                raise ValueError("training loader produced no batches; reduce batch size")
            means = metrics.compute_weighted_means()
            epoch_usage = usage.statistics()
            history.append(
                means
                | {
                    "perplexity": epoch_usage["perplexity"].item(),
                    "active_codes": epoch_usage["active_codes"].item(),
                    "entropy_bits": epoch_usage["token_entropy_nats"].item()
                    / math.log(2),
                }
            )
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == TOKENIZER_EPOCHS:
                save_image(
                    torch.cat(preview).mul(0.5).add(0.5),
                    training_dir / f"tokenizer_epoch_{epoch:03d}.png",
                    nrow=RECONSTRUCTION_SAMPLES,
                )
    validation = evaluate_tokenizer(
        model,
        validation_loader,
        vocabulary_size=CODEBOOK_SIZE,
        device=device,
    )
    save_training_metrics(
        history, out_dir, prefix="tokenizer", max_panels=MAX_METRIC_PANELS
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "vq_vae_tokenizer",
            "model_config": config,
            "validation": validation,
            "dataset": "celeba",
            "image_size": IMAGE_SIZE,
        },
        out_dir / TOKENIZER_CHECKPOINT_NAME,
    )
    return model.eval().requires_grad_(False)


def train_prior(
    tokenizer: VQVAE,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    tokenizer.eval().requires_grad_(False)
    vocabulary_size = tokenizer.quantizer.codebook_size
    prior = PixelCNNPrior(
        vocabulary_size,
        hidden_channels=PRIOR_HIDDEN_CHANNELS,
        layers=PRIOR_LAYERS,
        num_classes=NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=PRIOR_LR)
    training_dir = out_dir / "training"
    if not training_dir.exists():
        reset_dir(str(training_dir))
    history = []
    with tqdm(
        total=PRIOR_EPOCHS * len(train_loader),
        desc=f"PixelCNN 1/{PRIOR_EPOCHS}",
        unit="batch",
        mininterval=PROGRESS_INTERVAL,
    ) as progress:
        for epoch in range(1, PRIOR_EPOCHS + 1):
            progress.set_description(f"PixelCNN {epoch}/{PRIOR_EPOCHS}", refresh=False)
            nll = train_pixelcnn_prior_epoch(
                tokenizer,
                prior,
                train_loader,
                optimizer,
                device,
                progress=progress,
            )
            history.append({"nll": nll, "bits_per_token": nll / math.log(2)})
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == PRIOR_EPOCHS:
                prior.eval()
                labels = make_fixed_class_labels(NUM_CLASSES, SAMPLES_PER_CLASS, device)
                samples = sample_pixelcnn_prior_images(
                    tokenizer,
                    prior,
                    labels,
                    grid_size=LATENT_GRID_SIZE,
                    device=device,
                    temperature=TEMPERATURE,
                )
                save_image(
                    samples.mul(0.5).add(0.5),
                    training_dir / f"prior_epoch_{epoch:03d}.png",
                    nrow=NUM_CLASSES,
                )
                if epoch == PRIOR_EPOCHS:
                    save_image(
                        samples.mul(0.5).add(0.5),
                        out_dir / "prior_samples.png",
                        nrow=NUM_CLASSES,
                    )
    validation = evaluate_pixelcnn_prior(
        tokenizer,
        prior,
        validation_loader,
        tokens_per_image=TOKENS_PER_IMAGE,
        device=device,
    )
    prior_config = {
        "vocabulary_size": vocabulary_size,
        "hidden_channels": PRIOR_HIDDEN_CHANNELS,
        "layers": PRIOR_LAYERS,
        "num_classes": NUM_CLASSES,
    }
    save_training_metrics(
        history, out_dir, prefix="prior", max_panels=MAX_METRIC_PANELS
    )
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "model_name": "vq_vae_pixelcnn_prior",
            "model_config": prior_config,
            "validation": validation,
            "dataset": "celeba",
            "image_size": IMAGE_SIZE,
            "conditioning": "class_conditional",
            "attribute": CELEBA_SMILING_ATTRIBUTE,
            "class_names": list(CELEBA_SMILING_CLASSES),
        },
        out_dir / PRIOR_CHECKPOINT_NAME,
    )


def train() -> None:
    device = try_gpu()
    out_dir = OUTPUT_DIR
    train_loader, validation_loader = make_aligned_celeba_train_validation_loaders(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        num_workers=WORKERS,
        attribute=CELEBA_SMILING_ATTRIBUTE,
    )
    if not out_dir.exists():
        reset_dir(str(out_dir))
    reset_dir(str(out_dir / "training"))
    tokenizer = train_tokenizer(train_loader, validation_loader, device, out_dir)
    train_prior(
        tokenizer,
        train_loader,
        validation_loader,
        device,
        out_dir,
    )


def main() -> None:
    set_seed(SEED)
    train()


if __name__ == "__main__":
    main()
