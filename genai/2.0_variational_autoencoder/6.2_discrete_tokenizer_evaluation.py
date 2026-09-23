"""Compare VQ-VAE and FSQ with their frozen priors on CelebA-128.

Tokenizer evidence:

* reconstruction MSE/PSNR;
* quantization error, active tokens, marginal entropy, and fixed capacity.

System evidence:

* held-out prior cross-entropy and effective bits per image;
* conditional prior sample grids.

Data:
    data/celeba (official validation split), prepared by
    tool_scripts/download_dataset.py --dataset celeba.

Checkpoints:
    output/vae/vq_vae/{vq_vae.pth,pixelcnn_prior.pth}: VQ-VAE system
    output/vae/fsq/{fsq.pth,pixelcnn_prior.pth}: FSQ system
    Run 6.0 and 6.1 first to produce both complete systems.

Outputs:
    output/vae/evaluation/discrete_tokenizer/metrics.json: system comparison
    output/vae/evaluation/discrete_tokenizer/<system>_real_and_reconstruction.png
    output/vae/evaluation/discrete_tokenizer/<system>_prior_samples.png
    output/vae/evaluation/discrete_tokenizer/metric_comparison.png

Evaluation data -- CelebA validation:
Available images:                      19,867
Batch size:                                64
Reconstruction examples:                1,024
Generated examples saved:                  64
Generation batch size:                     10
Sampling temperature:                     1.0

Default dimensions:
Evaluation input:                     128x128 RGB
Generated image:                      128x128 RGB
Latent token grid:                      16x16 indices

Model size:
VQ-VAE tokenizer / prior:               1.71 M / 1.84 M parameters
VQ-VAE system total:                    3.55 M parameters
FSQ tokenizer / prior:                  1.60 M / 2.12 M parameters
FSQ system total:                       3.72 M parameters
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.celeba import (
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    make_aligned_celeba_loader,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.inference import generate_in_batches
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.quantization import (
    VQVAE,
    FSQAutoencoder,
    TokenUsageAccumulator,
)
from dl_utils.vae.token_prior import PixelCNNPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
OUTPUT_DIR = OUTPUT_ROOT / "evaluation" / "discrete_tokenizer"
RECONSTRUCTION_SAMPLES = 16
SAVED_GENERATION_SAMPLES = 64
SAMPLE_GRID_COLUMNS = 8
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGE_SIZE = 128
NUM_CLASSES = len(CELEBA_SMILING_CLASSES)


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
BATCH_SIZE = 64
MAX_EXAMPLES = 1_024
GENERATION_BATCH_SIZE = 10
TEMPERATURE = 1.0
WORKERS = 4
SEED = 123
VQ_VAE_TOKENIZER = OUTPUT_ROOT / "vq_vae" / "vq_vae.pth"
VQ_VAE_PRIOR = OUTPUT_ROOT / "vq_vae" / "pixelcnn_prior.pth"
FSQ_TOKENIZER = OUTPUT_ROOT / "fsq" / "fsq.pth"
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
        self, images: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if isinstance(self.tokenizer, VQVAE):
            reconstruction, indices, _, diagnostics = self.tokenizer(images)
        else:
            reconstruction, indices, diagnostics = self.tokenizer(images)
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
    *,
    max_examples: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    vocabulary_size = system.tokenizer.quantizer.codebook_size
    positions = system.latent_grid_size**2
    usage = TokenUsageAccumulator(vocabulary_size)
    squared_error = 0.0
    elements = 0
    quantization_sum = 0.0
    prior_nll_sum = 0.0
    examples = 0
    comparison = None
    for images, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        reconstruction, indices, diagnostics = system.reconstruct_and_tokens(images)
        if comparison is None:
            comparison = torch.cat(
                (
                    images[:RECONSTRUCTION_SAMPLES],
                    reconstruction[:RECONSTRUCTION_SAMPLES],
                )
            ).cpu()
        squared_error += (reconstruction - images).square().sum().item()
        elements += images.numel()
        quantization_sum += diagnostics["quantization_mse"].item() * images.shape[0]
        prior_nll_sum += (
            F.cross_entropy(
                system.prior(indices, labels=labels),
                indices,
            ).item()
            * images.shape[0]
        )
        usage.update(indices)
        examples += images.shape[0]

    mse = squared_error / elements
    statistics = usage.statistics()
    entropy_bits = statistics["token_entropy_nats"].item() / math.log(2)
    prior_bits = prior_nll_sum / examples / math.log(2)
    return {
        "examples": examples,
        "mse": mse,
        "psnr_for_minus_one_to_one_range": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
        "quantization_mse": quantization_sum / examples,
        "vocabulary_size": vocabulary_size,
        "active_codes": int(statistics["active_codes"]),
        "perplexity": statistics["perplexity"].item(),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": positions * entropy_bits,
        "fixed_length_bits_per_image": positions
        * math.ceil(math.log2(vocabulary_size)),
        "prior_bits_per_token": prior_bits,
        "prior_bits_per_image": positions * prior_bits,
    }, comparison


def save_metric_comparison(
    model_results: dict[str, object], output_path) -> None:
    """Compare fidelity, token capacity, and prior coding efficiency."""
    names = list(model_results)
    metrics = (
        ("Reconstruction MSE", "mse"),
        ("Reconstruction PSNR", "psnr_for_minus_one_to_one_range"),
        ("Marginal bits / image", "marginal_entropy_bits_per_image"),
        ("Prior bits / image", "prior_bits_per_image"),
    )
    with plt.ioff():
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for axis, (title, key) in zip(axes.flat, metrics):
            axis.bar(
                names,
                [model_results[name][key] for name in names],
                color=("#4c78a8", "#f58518"),
            )
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    loader = make_validation_loader(device)
    systems = load_systems(device)
    out_dir = OUTPUT_DIR
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
            "saved_generation_examples": SAVED_GENERATION_SAMPLES,
            "sampling_temperature": TEMPERATURE,
        },
        "models": model_results,
    }
    for system in systems:
        metrics, comparison = evaluate_tokenizer(
            system,
            loader,
            max_examples=MAX_EXAMPLES,
            device=device,
        )
        save_image(
            comparison.mul(0.5).add(0.5),
            out_dir / f"{system.name}_real_and_reconstruction.png",
            nrow=min(RECONSTRUCTION_SAMPLES, BATCH_SIZE, MAX_EXAMPLES),
        )
        images = generate_in_batches(
            torch.arange(SAVED_GENERATION_SAMPLES, device=device).remainder(
                NUM_CLASSES
            ),
            GENERATION_BATCH_SIZE,
            lambda labels, system=system: system.sample(
                len(labels), device=device, labels=labels, temperature=TEMPERATURE
            ),
        )
        save_image(
            images.mul(0.5).add(0.5),
            out_dir / f"{system.name}_prior_samples.png",
            nrow=SAMPLE_GRID_COLUMNS,
        )
        model_results[system.name] = metrics
    save_metric_comparison(model_results, out_dir / "metric_comparison.png")
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
