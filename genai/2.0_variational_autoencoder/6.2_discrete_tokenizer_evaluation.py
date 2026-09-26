"""Compare the selected VQ-VAE and FSQ systems on held-out CelebA-128.

The default tokenizers have equal 16x16 grids and 512-entry vocabularies.
Report paired MSE, both PSNR aggregations, token utilization, nominal rate,
and conditional prior coding cost. Latent quantization MSE is diagnostic
within a model; it is not a common distortion scale for VQ and FSQ.

Run 6.0 and 6.1 first. Loading validates preprocessing, labels, architecture
version and the exact tokenizer snapshot used to train each prior.
By default all 19,962 official test images are evaluated. Set MAX_EXAMPLES
for a reproducible sampled subset; metrics.json records its image IDs.
Generation remains a 64-image conditional preview, not a distribution metric.
Outputs remain in output/vae/evaluation/discrete_tokenizer/.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.datasets.celeba import (
    CELEBA_SMILING_CLASSES,
)
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.quantization import (
    VQVAE,
    FSQAutoencoder,
    TokenUsageAccumulator,
)
from dl_utils.vae.token_prior import PixelCNNPrior
from dl_utils.vae.tokenizer_workflow import (
    heldout_loader,
    load_prior_weights,
    load_tokenizer_weights,
)

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
EVALUATION_SPLIT = "test"
MAX_EXAMPLES: int | None = None  # None evaluates the complete split.
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


def make_evaluation_loader(device: torch.device):
    return heldout_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        split=EVALUATION_SPLIT,
        max_examples=MAX_EXAMPLES,
        seed=SEED,
        num_workers=WORKERS,
    )


def load_single_level_system(
    *,
    name: str,
    tokenizer_path: Path,
    prior_path: Path,
    device: torch.device,
) -> DiscreteSystem:
    model_class = VQVAE if name == "vq_vae" else FSQAutoencoder
    tokenizer, payload = load_tokenizer_weights(
        tokenizer_path,
        model_class,
        name=f"{name}_tokenizer",
        image_size=IMAGE_SIZE,
        device=device,
    )
    prior = load_prior_weights(
        prior_path,
        PixelCNNPrior,
        name=f"{name}_pixelcnn_prior",
        image_size=IMAGE_SIZE,
        tokenizer=tokenizer,
        tokenizer_payload=payload,
        device=device,
    )
    return DiscreteSystem(name, tokenizer, prior, IMAGE_SIZE)


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
    max_examples: int | None,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    vocabulary_size = system.tokenizer.quantizer.codebook_size
    positions = system.latent_grid_size**2
    usage = TokenUsageAccumulator(vocabulary_size)
    squared_error = 0.0
    psnr_sum = 0.0
    elements = 0
    quantization_sum = 0.0
    prior_nll_sum = 0.0
    examples = 0
    comparison = None
    for images, labels in loader:
        remaining = images.shape[0] if max_examples is None else max_examples - examples
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
        per_image_mse = (reconstruction - images).square().flatten(1).mean(1)
        psnr_sum += (10 * torch.log10(4 / per_image_mse.clamp_min(1e-12))).sum().item()
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

    if comparison is None or examples == 0:
        raise ValueError("Evaluation requires at least one held-out example.")
    mse = squared_error / elements
    statistics = usage.statistics()
    entropy_bits = statistics["token_entropy_nats"].item() / math.log(2)
    prior_bits = prior_nll_sum / examples / math.log(2)
    return {
        "examples": examples,
        "mse": mse,
        "mean_psnr_db": psnr_sum / examples,
        "psnr_from_pooled_mse_db": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
        "quantization_mse_within_model_only": quantization_sum / examples,
        "positions": positions,
        "tokenizer_parameters": sum(p.numel() for p in system.tokenizer.parameters()),
        "prior_parameters": sum(p.numel() for p in system.prior.parameters()),
        "vocabulary_size": vocabulary_size,
        "active_codes": int(statistics["active_codes"]),
        "perplexity": statistics["perplexity"].item(),
        "marginal_entropy_bits_per_token": entropy_bits,
        "marginal_entropy_bits_per_image": positions * entropy_bits,
        "fixed_length_bits_per_image": positions
        * math.ceil(math.log2(vocabulary_size)),
        "prior_bits_per_token": prior_bits,
        "prior_bits_per_image": positions * prior_bits,
        "prior_bits_per_pixel": positions * prior_bits / system.image_size**2,
    }, comparison


def save_metric_comparison(model_results: dict[str, Any], output_path) -> None:
    """Compare fidelity, token capacity, and prior coding efficiency."""
    names = list(model_results)
    metrics = (
        ("Reconstruction MSE", "mse"),
        ("Mean per-image PSNR", "mean_psnr_db"),
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
    loader, protocol = make_evaluation_loader(device)
    systems = load_systems(device)
    out_dir = OUTPUT_DIR
    reset_dir(str(out_dir))
    model_results: dict[str, Any] = {}
    results: dict[str, object] = {
        "protocol": {
            **protocol,
            "conditioning": "class_conditional",
            "saved_generation_examples": SAVED_GENERATION_SAMPLES,
            "sampling_temperature": TEMPERATURE,
            "sampling_seed": SEED,
            "generation_batch_size": GENERATION_BATCH_SIZE,
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
            nrow=comparison.shape[0] // 2,
        )
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
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
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
