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
    output/vae/vq_vae/tokenizer.pth: final tokenizer checkpoint
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

from dl_utils.data.celeba import (
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    make_aligned_celeba_train_validation_loaders,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.quantization import VQVAE, TokenUsageAccumulator
from dl_utils.vae.token_prior import (
    PixelCNNPrior,
    evaluate_pixelcnn_prior,
    make_fixed_class_labels,
    sample_pixelcnn_prior_images,
    train_pixelcnn_prior_epoch,
)

PROJECT_ROOT = infer_project_root()
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
    x: torch.Tensor,
    quantizer_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distortion = F.mse_loss(reconstruction, x)
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
    distortion = 0.0
    quantization_mse = 0.0
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, indices, _, diagnostics = model(x)
        distortion += float(F.mse_loss(reconstruction, x)) * x.shape[0]
        quantization_mse += float(diagnostics["quantization_mse"]) * x.shape[0]
        usage.update(indices)
        examples += x.shape[0]
    statistics = usage.statistics()
    entropy_bits = float(statistics["token_entropy_nats"]) / math.log(2)
    return {
        "mse": distortion / examples,
        "quantization_mse": quantization_mse / examples,
        "perplexity": float(statistics["perplexity"]),
        "active_codes": float(statistics["active_codes"]),
        "usage_fraction": float(statistics["usage_fraction"]),
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
    fixed_bits = TOKENS_PER_IMAGE * math.ceil(math.log2(CODEBOOK_SIZE))
    print(f"fixed-length upper bound={fixed_bits} bits/image")
    for epoch in range(1, TOKENIZER_EPOCHS + 1):
        model.train()
        sums = torch.zeros(4, device=device)
        examples = 0
        usage = TokenUsageAccumulator(CODEBOOK_SIZE)
        preview = None
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, quantizer_loss, diagnostics = model(x)
            loss, distortion = tokenizer_loss(reconstruction, x, quantizer_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                distortion,
                diagnostics["codebook_loss"],
                diagnostics["commitment_loss"],
            ]
            sums += torch.stack(values) * x.shape[0]
            usage.update(indices)
            examples += x.shape[0]
            preview = (x[:8].detach(), reconstruction[:8].detach())
        if preview is None:
            raise ValueError("training loader produced no batches; reduce batch size")
        means = (sums / examples).tolist()
        epoch_usage = usage.statistics()
        print(
            f"tokenizer {epoch:03d}: loss={means[0]:.4f}, D={means[1]:.4f}, "
            f"code={means[2]:.4f}, commit={means[3]:.4f}, "
            f"PPL={epoch_usage['perplexity'].item():.1f}, "
            f"active={epoch_usage['active_codes'].item()}/{CODEBOOK_SIZE}"
        )
        if epoch == 1 or epoch % SAMPLE_EVERY == 0:
            save_image(
                torch.cat(preview).mul(0.5).add(0.5),
                out_dir / f"tokenizer_{epoch:03d}.png",
                nrow=8,
            )
    validation = evaluate_tokenizer(
        model,
        validation_loader,
        vocabulary_size=CODEBOOK_SIZE,
        device=device,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "vq_vae_tokenizer",
            "model_config": config,
            "dataset": "celeba",
            "image_size": IMAGE_SIZE,
        },
        out_dir / "tokenizer.pth",
    )
    print(
        f"validation tokenizer: MSE={validation['mse']:.4f}, "
        f"entropy={validation['marginal_entropy_bits_per_token']:.3f} "
        f"bits/token, active={validation['active_codes']:.0f}/"
        f"{CODEBOOK_SIZE}"
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
    for epoch in range(1, PRIOR_EPOCHS + 1):
        nll = train_pixelcnn_prior_epoch(
            tokenizer,
            prior,
            train_loader,
            optimizer,
            device,
        )
        print(
            f"prior {epoch:03d}: train nll={nll:.4f}, "
            f"train bits/token={nll / math.log(2):.3f}"
        )
        if epoch == 1 or epoch % SAMPLE_EVERY == 0:
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
                out_dir / f"prior_{epoch:03d}.png",
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
    torch.save(
        {
            "state_dict": prior.state_dict(),
            "model_name": "vq_vae_pixelcnn_prior",
            "model_config": prior_config,
            "dataset": "celeba",
            "image_size": IMAGE_SIZE,
            "conditioning": "class_conditional",
            "attribute": CELEBA_SMILING_ATTRIBUTE,
            "class_names": list(CELEBA_SMILING_CLASSES),
        },
        out_dir / "pixelcnn_prior.pth",
    )
    print(
        f"validation prior: {validation['bits_per_token']:.3f} "
        f"bits/token, {validation['bits_per_image']:.1f} bits/image"
    )


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "output" / "vae" / "vq_vae"
    train_loader, validation_loader = make_aligned_celeba_train_validation_loaders(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        num_workers=WORKERS,
        attribute=CELEBA_SMILING_ATTRIBUTE,
    )
    reset_dir(str(out_dir))
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
