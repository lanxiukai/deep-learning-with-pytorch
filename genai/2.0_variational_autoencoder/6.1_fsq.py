"""FSQ: replace learned vector lookup with fixed scalar quantization.

This is a matched tokenizer branch of VQ-VAE. Relative to
``6.0_vq_vae.py``, the encoder, decoder, straight-through gradient, frozen
tokenizer boundary, and class-conditional second-stage PixelCNN remain. The
quantizer changes:

* each low-dimensional channel is bounded and rounded to fixed levels;
* mixed-radix packing maps the scalar tuple to one integer token;
* no learned codebook, codebook loss, or commitment loss exists.

Removing dead *entries* does not guarantee that the encoder visits every
scalar combination, so usage entropy and reconstruction are still logged.
The 128x128 image path and 16x16 token grid exactly match ``6.0_vq_vae.py`` so
the quantizer remains the controlled algorithmic difference.

Training runs the tokenizer first, then freezes it and trains the prior.
Both final weight files are saved in the same directory for evaluation.

Data:
    data/celeba (official train and validation splits), prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Outputs:
    output/vae/fsq/tokenizer_*.png: reconstruction comparisons
    output/vae/fsq/prior_*.png: one PixelCNN sample per Smiling label
    output/vae/fsq/tokenizer.pth: final FSQ tokenizer checkpoint
    output/vae/fsq/pixelcnn_prior.pth: final token-prior checkpoint

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
Latent token grid:            16x16 indices (1,600 fixed combinations)

Model size:
FSQ tokenizer:                 1.60 M parameters
Conditional PixelCNN prior:    2.12 M parameters (tokenizer frozen)
Stored total:                  3.72 M parameters
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
from dl_utils.vae.quantization import FSQAutoencoder, TokenUsageAccumulator
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
LEVELS = (8, 8, 5, 5)
TOKENIZER_EPOCHS = 20
PRIOR_EPOCHS = 20
HIDDEN_CHANNELS = 128
PRIOR_HIDDEN_CHANNELS = 128
PRIOR_LAYERS = 7
BATCH_SIZE = 64
LR = 2e-4
PRIOR_LR = 2e-4
TEMPERATURE = 1.0
SAMPLE_EVERY = 5
WORKERS = 4
SEED = 42


@torch.inference_mode()
def evaluate_tokenizer(
    model: FSQAutoencoder,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    vocabulary_size = model.quantizer.codebook_size
    distortion = 0.0
    quantization_mse = 0.0
    examples = 0
    usage = TokenUsageAccumulator(vocabulary_size)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        reconstruction, indices, diagnostics = model(x)
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
) -> FSQAutoencoder:
    config = {
        "levels": LEVELS,
        "hidden_channels": HIDDEN_CHANNELS,
        "downsample_steps": DOWNSAMPLE_STEPS,
    }
    model = FSQAutoencoder(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    vocabulary_size = model.quantizer.codebook_size
    fixed_bits = TOKENS_PER_IMAGE * math.ceil(math.log2(vocabulary_size))
    print(
        f"FSQ levels={LEVELS}, vocabulary={vocabulary_size}, "
        f"fixed-length upper bound={fixed_bits} bits/image"
    )
    for epoch in range(1, TOKENIZER_EPOCHS + 1):
        model.train()
        sums = torch.zeros(2, device=device)
        examples = 0
        usage = TokenUsageAccumulator(vocabulary_size)
        preview = None
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            reconstruction, indices, diagnostics = model(x)
            loss = F.mse_loss(reconstruction, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values = [
                loss.detach(),
                diagnostics["quantization_mse"],
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
            f"tokenizer {epoch:03d}: D={means[0]:.4f}, quant={means[1]:.4f}, "
            f"PPL={epoch_usage['perplexity'].item():.1f}, "
            f"active={epoch_usage['active_codes'].item()}/{vocabulary_size}, "
            f"epoch marginal entropy="
            f"{epoch_usage['token_entropy_nats'].item() / math.log(2):.2f} bits/token"
        )
        if epoch == 1 or epoch % SAMPLE_EVERY == 0:
            save_image(
                torch.cat(preview).mul(0.5).add(0.5),
                out_dir / f"tokenizer_{epoch:03d}.png",
                nrow=8,
            )
    validation = evaluate_tokenizer(model, validation_loader, device=device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "fsq_tokenizer",
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
        f"{vocabulary_size}"
    )
    return model.eval().requires_grad_(False)


def train_prior(
    tokenizer: FSQAutoencoder,
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
            "model_name": "fsq_pixelcnn_prior",
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
    out_dir = PROJECT_ROOT / "output" / "vae" / "fsq"
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
