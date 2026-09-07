"""Compare VQ-VAE and FSQ with their frozen priors on CelebA-128.

Tokenizer evidence:

* MSE/PSNR and reconstruction feature-distribution distance;
* quantization error, active tokens, marginal entropy, and fixed capacity.

System evidence:

* held-out prior cross-entropy and effective bits per image;
* prior-sampled feature-distribution distance and conditional sample grids.

The feature metric uses torchvision Inception-v3 with a fixed 256-dimensional random
projection.  It is a consistent Fréchet proxy for this comparison, not the
canonical TensorFlow FID implementation.

Data:
    data/celeba (official validation split), prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Checkpoints:
    output/vae/vq_vae/{tokenizer.pth,pixelcnn_prior.pth}: VQ-VAE system
    output/vae/fsq/{tokenizer.pth,pixelcnn_prior.pth}: FSQ system
    Run 6.0 and 6.1 first to produce both complete systems.

Outputs:
    output/vae/discrete_tokenizer_evaluation/metrics.json: system comparison
    output/vae/discrete_tokenizer_evaluation/<system>_real_and_reconstruction.png
    output/vae/discrete_tokenizer_evaluation/<system>_prior_samples.png

Evaluation data -- CelebA validation:
Available images:                      19,867
Batch size:                                64
Reconstruction examples:                1,024
Generated examples:                       100
Generation batch size:                     10
Sampling temperature:                     1.0
Projected Inception feature dimensions:    256

Default dimensions:
Evaluation input:                     128x128 RGB
Generated image:                      128x128 RGB
Latent token grid:                      16x16 indices

Model size:
VQ-VAE tokenizer / prior:               1.71 M / 1.84 M parameters
VQ-VAE system total:                    3.55 M parameters
FSQ tokenizer / prior:                  1.60 M / 2.12 M parameters
FSQ system total:                       3.72 M parameters
Frozen Inception-v3 evaluator:         25.11 M parameters
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.celeba import (
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    make_aligned_celeba_loader,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.image_quality import (
    FeatureMoments,
    TorchvisionInceptionFeatures,
    collect_reference_feature_moments,
    evaluate_conditional_generation,
    frechet_distance,
)
from dl_utils.vae.quantization import (
    VQVAE,
    FSQAutoencoder,
    TokenUsageAccumulator,
)
from dl_utils.vae.token_prior import PixelCNNPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGE_SIZE = 128
NUM_CLASSES = len(CELEBA_SMILING_CLASSES)


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
BATCH_SIZE = 64
MAX_EXAMPLES = 1_024
GENERATION_EXAMPLES = 100
GENERATION_BATCH_SIZE = 10
TEMPERATURE = 1.0
INCEPTION_PROJECTION_DIM = 256
FEATURE_SEED = 2026
WORKERS = 4
SEED = 123
VQ_VAE_TOKENIZER = OUTPUT_ROOT / "vq_vae" / "tokenizer.pth"
VQ_VAE_PRIOR = OUTPUT_ROOT / "vq_vae" / "pixelcnn_prior.pth"
FSQ_TOKENIZER = OUTPUT_ROOT / "fsq" / "tokenizer.pth"
FSQ_PRIOR = OUTPUT_ROOT / "fsq" / "pixelcnn_prior.pth"


@dataclass
class DiscreteSystem:
    name: str
    tokenizer: VQVAE | FSQAutoencoder
    prior: PixelCNNPrior
    image_size: int

    @property
    def latent_grid_size(self) -> int:
        return self.image_size // (2**self.tokenizer.downsample_steps)

    def reconstruct_and_tokens(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if isinstance(self.tokenizer, VQVAE):
            reconstruction, indices, _, diagnostics = self.tokenizer(x)
        else:
            reconstruction, indices, diagnostics = self.tokenizer(x)
        return reconstruction, indices, diagnostics

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor,
        temperature: float,
    ) -> Tensor:
        indices = self.prior.sample(
            count,
            self.latent_grid_size,
            self.latent_grid_size,
            device=device,
            labels=labels,
            temperature=temperature,
        )
        return self.tokenizer.decode_indices(indices)


def make_validation_loader(device: torch.device) -> DataLoader:
    return make_aligned_celeba_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        split="validation",
        attribute=CELEBA_SMILING_ATTRIBUTE,
        horizontal_flip=False,
        num_workers=WORKERS,
        shuffle=False,
        drop_last=False,
    )


def load_single_level_system(
    *,
    name: str,
    tokenizer_path: Path,
    prior_path: Path,
    device: torch.device,
) -> DiscreteSystem:
    checkpoint = torch.load(tokenizer_path, map_location=device, weights_only=True)
    model_class = VQVAE if name == "vq_vae" else FSQAutoencoder
    tokenizer = model_class(**checkpoint["model_config"]).to(device)
    tokenizer.load_state_dict(checkpoint["state_dict"])
    checkpoint = torch.load(prior_path, map_location=device, weights_only=True)
    prior = PixelCNNPrior(**checkpoint["model_config"]).to(device)
    prior.load_state_dict(checkpoint["state_dict"])
    return DiscreteSystem(
        name,
        tokenizer.eval().requires_grad_(False),
        prior.eval().requires_grad_(False),
        IMAGE_SIZE,
    )


def load_systems(device: torch.device) -> list[DiscreteSystem]:
    systems = [
        load_single_level_system(
            name="vq_vae",
            tokenizer_path=VQ_VAE_TOKENIZER,
            prior_path=VQ_VAE_PRIOR,
            device=device,
        ),
        load_single_level_system(
            name="fsq",
            tokenizer_path=FSQ_TOKENIZER,
            prior_path=FSQ_PRIOR,
            device=device,
        ),
    ]
    return systems


@torch.inference_mode()
def evaluate_tokenizer(
    system: DiscreteSystem,
    loader: DataLoader,
    feature_extractor: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    reconstruction_moments = FeatureMoments(feature_dim)
    vocabulary_size = system.tokenizer.quantizer.codebook_size
    positions = system.latent_grid_size**2
    usage = TokenUsageAccumulator(vocabulary_size)
    squared_error = 0.0
    elements = 0
    quantization_sum = 0.0
    prior_nll_sum = 0.0
    examples = 0
    comparison = None
    for x, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        reconstruction, indices, diagnostics = system.reconstruct_and_tokens(x)
        if comparison is None:
            comparison = torch.cat((x[:16], reconstruction[:16])).cpu()
        reconstruction_moments.update(feature_extractor(reconstruction))
        squared_error += float((reconstruction - x).square().sum())
        elements += x.numel()
        quantization_sum += float(diagnostics["quantization_mse"]) * x.shape[0]
        prior_nll_sum += (
            float(
                F.cross_entropy(
                    system.prior(indices, labels=labels),
                    indices,
                )
            )
            * x.shape[0]
        )
        usage.update(indices)
        examples += x.shape[0]

    mse = squared_error / elements
    statistics = usage.statistics()
    entropy_bits = float(statistics["token_entropy_nats"]) / math.log(2)
    prior_bits = prior_nll_sum / examples / math.log(2)
    return {
        "examples": examples,
        "mse": mse,
        "psnr_for_minus_one_to_one_range": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
        "projected_inception_reconstruction_frechet": frechet_distance(
            real_moments,
            reconstruction_moments,
        ),
        "quantization_mse": quantization_sum / examples,
        "vocabulary_size": vocabulary_size,
        "active_codes": int(statistics["active_codes"]),
        "perplexity": float(statistics["perplexity"]),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": positions * entropy_bits,
        "fixed_length_bits_per_image": positions
        * math.ceil(math.log2(vocabulary_size)),
        "prior_bits_per_token": prior_bits,
        "prior_bits_per_image": positions * prior_bits,
    }, comparison


def evaluate() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_validation_loader(device)
    systems = load_systems(device)
    projection_dim = INCEPTION_PROJECTION_DIM
    feature_extractor = TorchvisionInceptionFeatures(
        projection_dim=projection_dim,
        projection_seed=FEATURE_SEED,
    ).to(device)
    feature_dim = feature_extractor.feature_dim
    real_reconstruction_moments, real_generation_moments = (
        collect_reference_feature_moments(
            loader,
            feature_extractor,
            feature_dim=feature_dim,
            reconstruction_examples=MAX_EXAMPLES,
            generation_examples=GENERATION_EXAMPLES,
            device=device,
        )
    )
    out_dir = OUTPUT_ROOT / "discrete_tokenizer_evaluation"
    reset_dir(str(out_dir))
    model_results: dict[str, object] = {}
    results: dict[str, object] = {
        "protocol": {
            "dataset": "celeba",
            "split": "validation",
            "image_size": IMAGE_SIZE,
            "conditioning": "class_conditional",
            "attribute": CELEBA_SMILING_ATTRIBUTE,
            "class_names": list(CELEBA_SMILING_CLASSES),
            "max_reconstruction_examples": MAX_EXAMPLES,
            "generation_examples": GENERATION_EXAMPLES,
            "feature_extractor": "torchvision Inception-v3 pool features",
            "fixed_random_projection_dimension": projection_dim,
            "fid_warning": (
                "These numbers use torchvision preprocessing and are not "
                "canonical TensorFlow FID values."
            ),
        },
        "models": model_results,
    }
    for system in systems:
        metrics, comparison = evaluate_tokenizer(
            system,
            loader,
            feature_extractor,
            real_reconstruction_moments,
            feature_dim=feature_dim,
            max_examples=MAX_EXAMPLES,
            device=device,
        )
        save_image(
            comparison.mul(0.5).add(0.5),
            out_dir / f"{system.name}_real_and_reconstruction.png",
            nrow=16,
        )
        generation, images = evaluate_conditional_generation(
            system,
            feature_extractor,
            real_generation_moments,
            examples=GENERATION_EXAMPLES,
            batch_size=GENERATION_BATCH_SIZE,
            num_classes=NUM_CLASSES,
            temperature=TEMPERATURE,
            device=device,
        )
        metrics["generation"] = generation
        save_image(
            images.mul(0.5).add(0.5),
            out_dir / f"{system.name}_prior_samples.png",
            nrow=8,
        )
        model_results[system.name] = metrics
        print(
            f"{system.name}: MSE={metrics['mse']:.4f}, "
            f"rFID-proxy="
            f"{metrics['projected_inception_reconstruction_frechet']:.2f}, "
            f"entropy={metrics['marginal_entropy_bits_per_image']:.1f} "
            "bits/image"
        )
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved evaluation to {out_dir}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
