"""Evaluate selected VQGAN, PatchGAN, and Transformer weights on glasses-256.

The complete 4,500-image training set at 256x256 is the default. MAX_EXAMPLES
selects and records a seeded random subset for a shorter check. These are
training-set diagnostics, not held-out results. Report paired L1,
MSE, both PSNR aggregations, LPIPS, token rates, class-conditional prior NLL,
and PatchGAN diagnostics. The 64 generated images remain a visual preview;
columns alternate G (with glasses) and NoG (without glasses).

Loading requires matching tokenizer/prior snapshot IDs, image preprocessing,
conditioning and the current four-step spatial compression. Run 7.0 first.
The default 16x16 grid and 512-entry vocabulary match 6.0/6.1. Different
conditioning, backbones, objectives, priors and training budgets prevent a controlled
algorithm ranking.
Outputs remain in output/vae/vqgan/evaluation/.
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

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.perceptual_autoencoder import (
    LPIPSPerceptualLoss,
    PatchDiscriminator,
    VQPerceptualAutoencoder,
)
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS, TokenUsageAccumulator
from dl_utils.vae.tokenizer_workflow import (
    glasses_loader,
    load_prior_weights,
    load_tokenizer_weights,
)
from dl_utils.vae.transformer_prior import CausalTransformerPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
IMAGE_SIZE = 256
NUM_CLASSES = len(GLASSES_CLASS_NAMES)
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
BATCH_SIZE = 16
MAX_EXAMPLES: int | None = None  # None evaluates all training images.
GENERATION_BATCH_SIZE = 10
TEMPERATURE = 1.0
WORKERS = 4
SEED = 123
VQGAN_TOKENIZER = OUTPUT_ROOT / "vqgan" / "vqgan.pth"
OUTPUT_DIR = VQGAN_TOKENIZER.parent / "evaluation"
PRIOR_CHECKPOINT_NAME = "transformer_prior.pth"
RECONSTRUCTION_SAMPLES = 16
SAVED_GENERATION_SAMPLES = 64
SAMPLE_GRID_COLUMNS = 8


@dataclass
class EvaluatedSystem:
    tokenizer: VQPerceptualAutoencoder
    prior: CausalTransformerPrior
    discriminator: PatchDiscriminator

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.quantizer.codebook_size

    def reconstruct_and_tokens(
        self, images: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        reconstruction, indices, _, diagnostics = self.tokenizer(images)
        return reconstruction, indices, diagnostics

    def prior_loss(self, indices: Tensor, labels: Tensor) -> Tensor:
        logits, targets = self.prior.teacher_forcing(indices, labels)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor,
        temperature: float,
    ) -> Tensor:
        side = IMAGE_SIZE // (2**self.tokenizer.downsample_steps)
        indices = self.prior.sample(
            count,
            device=device,
            labels=labels,
            temperature=temperature,
        ).reshape(count, side, side)
        return self.tokenizer.decode_indices(indices)


def make_evaluation_loader(device: torch.device):
    return glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        device,
        max_examples=MAX_EXAMPLES,
        seed=SEED,
        num_workers=WORKERS,
        conditional=True,
    )


def load_vqgan_system(tokenizer_path: Path, device: torch.device) -> EvaluatedSystem:
    tokenizer, payload = load_tokenizer_weights(
        tokenizer_path,
        VQPerceptualAutoencoder,
        name="vqgan_tokenizer",
        image_size=IMAGE_SIZE,
        device=device,
        downsample_steps=DOWNSAMPLE_STEPS,
        conditional=True,
    )
    discriminator = PatchDiscriminator(**payload["discriminator_config"]).to(device)
    discriminator.load_state_dict(payload["discriminator_state_dict"])
    prior = load_prior_weights(
        tokenizer_path.with_name(PRIOR_CHECKPOINT_NAME),
        CausalTransformerPrior,
        name="vqgan_transformer_prior",
        image_size=IMAGE_SIZE,
        tokenizer=tokenizer,
        tokenizer_payload=payload,
        device=device,
        conditional=True,
    )
    return EvaluatedSystem(tokenizer, prior, discriminator.eval().requires_grad_(False))


@torch.inference_mode()
def evaluate_reconstruction(
    system: EvaluatedSystem,
    loader: DataLoader,
    perceptual: nn.Module,
    *,
    max_examples: int | None,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    usage = TokenUsageAccumulator(system.vocabulary_size)
    paired_totals = torch.zeros(3, device=device)
    discriminator_totals = torch.zeros(2, device=device)
    squared_error = 0.0
    psnr_sum = 0.0
    element_count = 0
    prior_nll = 0.0
    prior_examples = 0
    examples = 0
    positions = 0
    comparison: Tensor | None = None
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
        positions = indices.shape[1] * indices.shape[2]
        per_image_mse = (reconstruction - images).square().flatten(1).mean(1)
        psnr_sum += (10 * torch.log10(4 / per_image_mse.clamp_min(1e-12))).sum().item()
        squared_error += (reconstruction - images).square().sum().item()
        element_count += images.numel()
        paired_totals += (
            torch.stack(
                [
                    F.l1_loss(reconstruction, images),
                    perceptual(reconstruction, images),
                    diagnostics["quantization_mse"],
                ]
            )
            * images.shape[0]
        )
        usage.update(indices)
        loss = system.prior_loss(indices, labels)
        prior_nll += loss.item() * images.shape[0]
        prior_examples += images.shape[0]
        discriminator_totals += (
            torch.stack(
                [
                    system.discriminator(images).mean(),
                    system.discriminator(reconstruction).mean(),
                ]
            )
            * images.shape[0]
        )
        examples += images.shape[0]

    if comparison is None or examples == 0:
        raise ValueError("evaluation needs at least one training example")
    paired = (paired_totals / examples).tolist()
    mse = squared_error / element_count
    token_statistics = usage.statistics()
    entropy_bits = token_statistics["token_entropy_nats"].item() / math.log(2)
    nll = prior_nll / prior_examples
    prior_metrics = {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": positions * nll / math.log(2),
        "bits_per_pixel": positions * nll / math.log(2) / IMAGE_SIZE**2,
        "parameter_count": sum(
            parameter.numel() for parameter in system.prior.parameters()
        ),
    }
    logits = (discriminator_totals / examples).tolist()
    discriminator_metrics = {
        "real_logit": logits[0],
        "reconstruction_logit": logits[1],
        "real_minus_reconstruction_logit": logits[0] - logits[1],
    }
    return {
        "examples": examples,
        "paired_fidelity": {
            "pixel_l1": paired[0],
            "mse": mse,
            "mean_psnr_db": psnr_sum / examples,
            "psnr_from_pooled_mse_db": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
            "lpips_v0_1_vgg": paired[1],
        },
        "quantization": {
            "mse_within_model_only": paired[2],
            "vocabulary_size": system.vocabulary_size,
            "positions": positions,
            "active_codes": int(token_statistics["active_codes"]),
            "usage_fraction": token_statistics["usage_fraction"].item(),
            "perplexity": token_statistics["perplexity"].item(),
            "marginal_entropy_bits_per_token": entropy_bits,
            "marginal_entropy_bits_per_image": positions * entropy_bits,
            "fixed_length_bits_per_image": positions
            * math.ceil(math.log2(system.vocabulary_size)),
        },
        "patch_discriminator": discriminator_metrics,
        "prior": prior_metrics,
    }, comparison


def save_metric_summary(metrics: dict[str, object], output_path) -> None:
    """Visualize training-set fidelity, code use, and prior fit."""
    fidelity = metrics["paired_fidelity"]
    quantization = metrics["quantization"]
    prior = metrics["prior"]
    assert isinstance(fidelity, dict)
    assert isinstance(quantization, dict)
    assert isinstance(prior, dict)
    with plt.ioff():
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].bar(
            ("L1", "LPIPS"),
            (
                fidelity["pixel_l1"],
                fidelity["lpips_v0_1_vgg"],
            ),
            color=("#4c78a8", "#f58518"),
        )
        axes[0].set_title("Paired reconstruction error")
        axes[1].bar(
            ("Active codes", "Perplexity"),
            (quantization["active_codes"], quantization["perplexity"]),
            color=("#e45756", "#72b7b2"),
        )
        axes[1].set_title("Token utilization")
        axes[2].bar(
            ("Prior bits / token", "Prior bits / image"),
            (prior["bits_per_token"], prior["bits_per_image"]),
            color=("#b279a2", "#ff9da6"),
        )
        axes[2].set_title("Prior coding cost")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    system = load_vqgan_system(VQGAN_TOKENIZER, device)
    loader, protocol = make_evaluation_loader(device)
    perceptual = LPIPSPerceptualLoss().to(device)
    metrics, comparison = evaluate_reconstruction(
        system,
        loader,
        perceptual,
        max_examples=MAX_EXAMPLES,
        device=device,
    )
    sample_labels = torch.arange(SAVED_GENERATION_SAMPLES, device=device).remainder(
        NUM_CLASSES
    )
    images = generate_in_batches(
        sample_labels,
        GENERATION_BATCH_SIZE,
        lambda labels: system.sample(
            len(labels), device=device, labels=labels, temperature=TEMPERATURE
        ),
    )
    metrics["protocol"] = {
        **protocol,
        "sampling_seed": SEED,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "saved_generation_examples": SAVED_GENERATION_SAMPLES,
        "sampling_temperature": TEMPERATURE,
        "generation_class_ids": sample_labels.cpu().tolist(),
    }
    reset_dir(str(OUTPUT_DIR))
    save_image(
        comparison.mul(0.5).add(0.5),
        OUTPUT_DIR / "vqgan_real_and_reconstruction.png",
        nrow=comparison.shape[0] // 2,
    )
    save_image(
        images.mul(0.5).add(0.5),
        OUTPUT_DIR / "vqgan_prior_samples.png",
        nrow=SAMPLE_GRID_COLUMNS,
    )
    save_metric_summary(metrics, OUTPUT_DIR / "metric_summary.png")
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
