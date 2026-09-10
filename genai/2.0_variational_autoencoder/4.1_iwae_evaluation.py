"""Compare IWAE particle counts under one held-out protocol.

Both default K=1 and K=5 checkpoints is re-evaluated with the same large K,
particle chunk size, test examples, and repeated random estimates. This keeps
the training objective separate from the evaluation estimator. Posterior-mean
reconstruction, Monte Carlo rate, active units, ESS/K, and prior samples are
reported alongside the bound; none substitutes for the others.

Data:
    data/mnist, downloaded automatically by torchvision when absent. Digit
    labels are ignored by the IWAE comparison.

Checkpoints:
    output/vae/iwae/k1/model.pth: default K=1 model
    output/vae/iwae/k5/model.pth: default K=5 model
    Run 4.0_iwae.py first to produce both checkpoints.

Outputs:
    output/vae/iwae_evaluation/metrics.json: shared-protocol comparison
    output/vae/iwae_evaluation/<model>_real_reconstruction_prior.png

Evaluation data -- MNIST test:
Available images:                  10,000
Batch size:                            64
Evaluated images per repeat:         2,048
Evaluation particles:                 128
Particle chunk size:                    8
Repeated estimates:                     3
Prior samples in each grid:             16

Default dimensions:
Evaluation input:                  32x32 grayscale
Generated image:                   32x32 grayscale
Latent vector:                         16 values

Model size (each checkpoint):
Encoder/posterior:                  0.431 M parameters
Decoder:                            0.199 M parameters
Total:                              0.631 M parameters
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.inference import (
    GaussianVAE32,
    importance_log_weights,
    log_mean_exp,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"


# Edit these defaults to explore the lesson.
EVALUATION_PARTICLES = 128
PARTICLE_CHUNK_SIZE = 8
REPEATS = 3
MAX_EXAMPLES = 2_048
BATCH_SIZE = 64
ACTIVE_VARIANCE_THRESHOLD = 1e-2
WORKERS = 4
SEED = 123
IWAE_K1_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k1" / "model.pth"
IWAE_K5_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k5" / "model.pth"


def make_test_loader(device: torch.device) -> DataLoader:
    dataset = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        ),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=WORKERS > 0,
    )


def load_model(path: Path, device: torch.device) -> GaussianVAE32:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "iwae":
        raise ValueError(f"{path} is not an IWAE checkpoint")
    model = GaussianVAE32(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_model(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    particles: int,
    particle_chunk_size: int,
    repeats: int,
    max_examples: int,
    active_variance_threshold: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    repeat_bounds = []
    all_ess_fractions = []
    all_weight_ranges = []
    all_rates = []
    posterior_codes = []
    distortion_total = 0.0
    distortion_examples = 0

    for repeat in range(repeats):
        set_seed(seed + repeat)
        bound_total = 0.0
        examples = 0
        for images, _ in loader:
            remaining = max_examples - examples
            if remaining <= 0:
                break
            images = images[:remaining].to(device, non_blocking=True)
            log_weights, terms = importance_log_weights(
                model,
                images,
                particles=particles,
                particle_chunk_size=particle_chunk_size,
            )
            per_example_bound = log_mean_exp(log_weights)
            normalized = torch.softmax(log_weights, dim=1)
            ess_fraction = normalized.square().sum(dim=1).reciprocal() / particles
            weight_range = log_weights.max(dim=1).values - log_weights.min(dim=1).values
            rate_samples = terms["log_q"] - terms["log_pz"]
            bound_total += float(per_example_bound.sum())
            all_ess_fractions.append(ess_fraction.cpu())
            all_weight_ranges.append(weight_range.cpu())
            all_rates.append(rate_samples.mean(dim=1).cpu())
            if repeat == 0:
                posterior_mean, _ = model.encode(images)
                reconstruction = model.decode(posterior_mean)
                distortion_total += float(
                    F.binary_cross_entropy(reconstruction, images, reduction="sum")
                )
                distortion_examples += images.shape[0]
                posterior_codes.append(posterior_mean.cpu())
            examples += images.shape[0]
        repeat_bounds.append(bound_total / examples)

    ess = torch.cat(all_ess_fractions)
    ranges = torch.cat(all_weight_ranges)
    rates = torch.cat(all_rates)
    codes = torch.cat(posterior_codes)
    code_variance = codes.var(dim=0, unbiased=False)
    bounds = torch.tensor(repeat_bounds, dtype=torch.float64)
    return {
        "evaluation_particles": particles,
        "repeats": repeats,
        "examples_per_repeat": min(max_examples, len(loader.dataset)),
        "bound_mean": float(bounds.mean()),
        "bound_repeat_standard_deviation": float(bounds.std(unbiased=False)),
        "repeat_bounds": repeat_bounds,
        "posterior_mean_distortion": (distortion_total / distortion_examples),
        "monte_carlo_rate": float(rates.mean()),
        "active_units": int((code_variance > active_variance_threshold).sum()),
        "ess_fraction_mean": float(ess.mean()),
        "ess_fraction_quantiles": [
            float(value) for value in torch.quantile(ess, torch.tensor([0.1, 0.5, 0.9]))
        ],
        "log_weight_range_mean": float(ranges.mean()),
    }


@torch.inference_mode()
def save_model_comparison(
    model: GaussianVAE32,
    loader: DataLoader,
    path: Path,
    *,
    device: torch.device,
) -> None:
    real, _ = next(iter(loader))
    real = real[:16].to(device)
    reconstructions = model.reconstruct(real)
    samples = model.sample(16, device=device)
    save_image(
        torch.cat((real, reconstructions, samples)),
        path,
        nrow=16,
    )


def checkpoint_paths() -> dict[str, Path]:
    return {
        "vae_elbo_k1": IWAE_K1_CHECKPOINT,
        "iwae_k5": IWAE_K5_CHECKPOINT,
    }


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_test_loader(device)
    paths = checkpoint_paths()
    out_dir = OUTPUT_ROOT / "iwae_evaluation"
    reset_dir(str(out_dir))
    results: dict[str, object] = {
        "protocol": {
            "dataset": "MNIST test",
            "image_size": 32,
            "observation": "independent Bernoulli mean",
            "evaluation_particles": EVALUATION_PARTICLES,
            "particle_chunk_size": PARTICLE_CHUNK_SIZE,
            "repeats": REPEATS,
            "max_examples": MAX_EXAMPLES,
        },
        "models": {},
    }
    for name, path in paths.items():
        model = load_model(path, device)
        metrics = evaluate_model(
            model,
            loader,
            particles=EVALUATION_PARTICLES,
            particle_chunk_size=PARTICLE_CHUNK_SIZE,
            repeats=REPEATS,
            max_examples=MAX_EXAMPLES,
            active_variance_threshold=ACTIVE_VARIANCE_THRESHOLD,
            seed=SEED,
            device=device,
        )
        results["models"][name] = metrics
        save_model_comparison(
            model,
            loader,
            out_dir / f"{name}_real_reconstruction_prior.png",
            device=device,
        )
        print(
            f"{name}: bound={metrics['bound_mean']:.3f} +/- "
            f"{metrics['bound_repeat_standard_deviation']:.3f}, "
            f"ESS/K={metrics['ess_fraction_mean']:.3f}, "
            f"rate={metrics['monte_carlo_rate']:.3f}"
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved comparison to {out_dir}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
