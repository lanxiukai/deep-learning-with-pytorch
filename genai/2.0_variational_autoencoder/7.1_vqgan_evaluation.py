"""Evaluate the default VQGAN tokenizer, PatchGAN, and Transformer prior.

Load the final weights from 7.0 and evaluate on aligned CelebA-128 validation
images. Report paired reconstruction fidelity (L1, PSNR, SSIM, LPIPS), token
usage and entropy, prior cross-entropy, PatchGAN scores, and generated images.
The projected Inception distance is a teaching proxy, not canonical FID.

Inputs:
    data/celeba: official validation split.
    output/vae/vqgan/tokenizer.pth: tokenizer and discriminator weights.
    output/vae/vqgan/transformer_prior.pth: frozen-token prior weights.

Outputs:
    output/vae/vqgan_evaluation/metrics.json
    output/vae/vqgan_evaluation/vqgan_real_and_reconstruction.png
    output/vae/vqgan_evaluation/vqgan_prior_samples.png

Defaults: 1,024 reconstruction examples, 100 generated images, batch size 16,
256 projected Inception features, and an 8x8 latent token grid.
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
    structural_similarity_index,
)
from dl_utils.vae.perceptual_autoencoder import (
    LPIPSPerceptualLoss,
    PatchDiscriminator,
    VQPerceptualAutoencoder,
)
from dl_utils.vae.quantization import TokenUsageAccumulator
from dl_utils.vae.token_prior import CausalTransformerPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "celeba"
IMAGE_SIZE = 128
NUM_CLASSES = len(CELEBA_SMILING_CLASSES)


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
BATCH_SIZE = 16
MAX_EXAMPLES = 1_024
GENERATION_EXAMPLES = 100
GENERATION_BATCH_SIZE = 10
TEMPERATURE = 1.0
INCEPTION_PROJECTION_DIM = 256
FEATURE_SEED = 2026
WORKERS = 4
SEED = 123
VQGAN_TOKENIZER = OUTPUT_ROOT / "vqgan" / "tokenizer.pth"
OUTPUT_DIR = OUTPUT_ROOT / "vqgan_evaluation"


@dataclass
class EvaluatedSystem:
    tokenizer: VQPerceptualAutoencoder
    prior: CausalTransformerPrior
    discriminator: PatchDiscriminator

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.quantizer.codebook_size

    def reconstruct_and_tokens(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        reconstruction, indices, _, diagnostics = self.tokenizer(x)
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
        side = math.isqrt(self.prior.sequence_length)
        indices = self.prior.sample(
            count,
            device=device,
            labels=labels,
            temperature=temperature,
        ).reshape(count, side, side)
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


def load_vqgan_system(tokenizer_path: Path, device: torch.device) -> EvaluatedSystem:
    checkpoint = torch.load(tokenizer_path, map_location=device, weights_only=True)
    tokenizer = VQPerceptualAutoencoder(**checkpoint["model_config"]).to(device)
    tokenizer.load_state_dict(checkpoint["state_dict"])
    discriminator = PatchDiscriminator(**checkpoint["discriminator_config"]).to(device)
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    checkpoint = torch.load(
        tokenizer_path.with_name("transformer_prior.pth"),
        map_location=device,
        weights_only=True,
    )
    prior = CausalTransformerPrior(**checkpoint["model_config"]).to(device)
    prior.load_state_dict(checkpoint["state_dict"])
    return EvaluatedSystem(
        tokenizer.eval().requires_grad_(False),
        prior.eval().requires_grad_(False),
        discriminator.eval().requires_grad_(False),
    )


@torch.inference_mode()
def evaluate_reconstruction(
    system: EvaluatedSystem,
    loader: DataLoader,
    feature_extractor: nn.Module,
    perceptual: nn.Module,
    real_moments: FeatureMoments,
    *,
    feature_dim: int,
    max_examples: int,
    device: torch.device,
) -> tuple[dict[str, object], Tensor]:
    reconstruction_moments = FeatureMoments(feature_dim)
    usage = TokenUsageAccumulator(system.vocabulary_size)
    paired_totals = torch.zeros(4, device=device)
    discriminator_totals = torch.zeros(2, device=device)
    squared_error = 0.0
    element_count = 0
    prior_nll = 0.0
    prior_examples = 0
    examples = 0
    positions = 0
    comparison: Tensor | None = None
    for x, labels in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        x = x[:remaining].to(device, non_blocking=True)
        labels = labels[:remaining].to(device, non_blocking=True)
        reconstruction, indices, diagnostics = system.reconstruct_and_tokens(x)
        if comparison is None:
            comparison = torch.cat((x[:16], reconstruction[:16])).cpu()
        positions = indices.shape[1] * indices.shape[2]
        reconstruction_moments.update(feature_extractor(reconstruction))
        squared_error += float((reconstruction - x).square().sum())
        element_count += x.numel()
        paired_totals += (
            torch.stack(
                [
                    F.l1_loss(reconstruction, x),
                    perceptual(reconstruction, x),
                    structural_similarity_index(reconstruction, x),
                    diagnostics["quantization_mse"],
                ]
            )
            * x.shape[0]
        )
        usage.update(indices)
        loss = system.prior_loss(indices, labels)
        prior_nll += float(loss) * x.shape[0]
        prior_examples += x.shape[0]
        discriminator_totals += (
            torch.stack(
                [
                    system.discriminator(x).mean(),
                    system.discriminator(reconstruction).mean(),
                ]
            )
            * x.shape[0]
        )
        examples += x.shape[0]

    if comparison is None or examples < 2:
        raise ValueError("evaluation needs at least two held-out examples")
    paired = (paired_totals / examples).tolist()
    mse = squared_error / element_count
    token_statistics = usage.statistics()
    entropy_bits = float(token_statistics["token_entropy_nats"]) / math.log(2)
    nll = prior_nll / prior_examples
    prior_metrics = {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": positions * nll / math.log(2),
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
            "psnr_for_minus_one_to_one_range": 10.0 * math.log10(4.0 / max(mse, 1e-12)),
            "ssim": paired[2],
            "lpips_v0_1_vgg": paired[1],
        },
        "reconstruction_distribution": {
            "projected_inception_frechet": frechet_distance(
                real_moments, reconstruction_moments
            )
        },
        "quantization": {
            "mse": paired[3],
            "vocabulary_size": system.vocabulary_size,
            "positions": positions,
            "active_codes": int(token_statistics["active_codes"]),
            "usage_fraction": float(token_statistics["usage_fraction"]),
            "perplexity": float(token_statistics["perplexity"]),
            "marginal_entropy_bits_per_token": entropy_bits,
            "marginal_entropy_bits_per_image": positions * entropy_bits,
            "fixed_length_bits_per_image": positions
            * math.ceil(math.log2(system.vocabulary_size)),
        },
        "patch_discriminator": discriminator_metrics,
        "prior": prior_metrics,
    }, comparison


def evaluate() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = load_vqgan_system(VQGAN_TOKENIZER, device)
    loader = make_validation_loader(device)
    feature_extractor = TorchvisionInceptionFeatures(
        projection_dim=INCEPTION_PROJECTION_DIM,
        projection_seed=FEATURE_SEED,
    ).to(device)
    perceptual = LPIPSPerceptualLoss().to(device)
    feature_dim = feature_extractor.feature_dim
    real_reconstruction, real_generation = collect_reference_feature_moments(
        loader,
        feature_extractor,
        feature_dim=feature_dim,
        reconstruction_examples=MAX_EXAMPLES,
        generation_examples=GENERATION_EXAMPLES,
        device=device,
    )
    metrics, comparison = evaluate_reconstruction(
        system,
        loader,
        feature_extractor,
        perceptual,
        real_reconstruction,
        feature_dim=feature_dim,
        max_examples=MAX_EXAMPLES,
        device=device,
    )
    generation, images = evaluate_conditional_generation(
        system,
        feature_extractor,
        real_generation,
        examples=GENERATION_EXAMPLES,
        batch_size=GENERATION_BATCH_SIZE,
        num_classes=NUM_CLASSES,
        temperature=TEMPERATURE,
        device=device,
    )
    metrics["generation"] = generation
    metrics["protocol"] = {
        "dataset": "CelebA validation",
        "image_size": IMAGE_SIZE,
        "class_names": list(CELEBA_SMILING_CLASSES),
        "feature_projection_dimension": INCEPTION_PROJECTION_DIM,
        "fid_warning": "Projected torchvision features give a teaching proxy, not canonical FID.",
    }
    reset_dir(str(OUTPUT_DIR))
    save_image(
        comparison.mul(0.5).add(0.5),
        OUTPUT_DIR / "vqgan_real_and_reconstruction.png",
        nrow=16,
    )
    save_image(images.mul(0.5).add(0.5), OUTPUT_DIR / "vqgan_prior_samples.png", nrow=8)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    paired = metrics["paired_fidelity"]
    print(
        f"VQGAN: L1={paired['pixel_l1']:.4f}, LPIPS={paired['lpips_v0_1_vgg']:.4f}; "
        f"saved evaluation to {OUTPUT_DIR}"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
