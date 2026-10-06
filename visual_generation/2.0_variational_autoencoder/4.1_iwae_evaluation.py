"""Compare IWAE particle counts under one held-out protocol.

Data:
    data/mnist, downloaded automatically by torchvision when absent. Digit
    labels are ignored by the IWAE comparison.

Checkpoints:
    output/vae/iwae/k1/iwae.pth: default K=1 model
    output/vae/iwae/k4/iwae.pth: default K=4 model
    output/vae/iwae/k8/iwae.pth: default K=8 model
    output/vae/iwae/k16/iwae.pth: default K=16 model
    output/vae/iwae/k32/iwae.pth: default K=32 model
    Run 4.0_iwae.py first to produce all checkpoints.

Outputs:
    output/vae/iwae/evaluation/metrics.json: shared-protocol means, repeated-
        estimate uncertainty, and changes relative to K=1
    output/vae/iwae/evaluation/metrics_by_training_particles.png: metric
        means and repeated-estimate uncertainty across training K values
    output/vae/iwae/evaluation/<model>_real_reconstruction_prior.png

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
from statistics import stdev

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.iwae import (
    GaussianVAE,
    evaluate_iwae,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
OUTPUT_DIR = OUTPUT_ROOT / "iwae" / "evaluation"
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
IMAGE_SIZE = 32
DISPLAY_SAMPLES = 16


# Edit these defaults to explore the lesson.
EVALUATION_PARTICLES = 128
PARTICLE_CHUNK_SIZE = 8
REPEATS = 3
MAX_EXAMPLES = 2_048
BATCH_SIZE = 64
WORKERS = 4
SEED = 123
IWAE_K1_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k1" / "iwae.pth"
IWAE_K4_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k4" / "iwae.pth"
IWAE_K8_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k8" / "iwae.pth"
IWAE_K16_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k16" / "iwae.pth"
IWAE_K32_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k32" / "iwae.pth"
TRAINING_PARTICLES = {
    "vae_elbo_k1": 1,
    "iwae_k4": 4,
    "iwae_k8": 8,
    "iwae_k16": 16,
    "iwae_k32": 32,
}


def make_test_loader(device: torch.device) -> DataLoader:
    dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
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


def load_model(path: Path, device: torch.device) -> GaussianVAE:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_name") != "iwae":
        raise ValueError(f"{path} is not an IWAE checkpoint")
    model = GaussianVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def evaluate_model(
    model: GaussianVAE,
    loader: DataLoader,
    *,
    particles: int,
    particle_chunk_size: int,
    repeats: int,
    max_examples: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    estimates: dict[str, list[float]] = {}

    for repeat in range(repeats):
        set_seed(seed + repeat)
        metrics = evaluate_iwae(
            model,
            loader,
            particles=particles,
            particle_chunk_size=particle_chunk_size,
            max_examples=max_examples,
            device=device,
        )
        for name, value in metrics.items():
            estimates.setdefault(name, []).append(value)

    return {
        "mean": {
            name: sum(values) / len(values) for name, values in estimates.items()
        },
        "sample_stddev": {
            name: stdev(values) if len(values) > 1 else 0.0
            for name, values in estimates.items()
        },
    }


def changes_from_baseline(
    metrics: dict[str, float], baseline: dict[str, float]
) -> dict[str, dict[str, float | None]]:
    """Return absolute and percentage changes relative to one metric baseline."""
    changes: dict[str, dict[str, float | None]] = {}
    for name, value in metrics.items():
        baseline_value = baseline[name]
        absolute = value - baseline_value
        changes[name] = {
            "absolute": absolute,
            "percent": None
            if baseline_value == 0.0
            else 100.0 * absolute / baseline_value,
        }
    return changes


def save_metric_comparison(
    model_results: dict[str, dict[str, object]], path: Path
) -> None:
    """Plot held-out metrics by training-particle count with repeat variability."""
    ordered_names = sorted(
        model_results, key=lambda name: TRAINING_PARTICLES[name]
    )
    particles = [TRAINING_PARTICLES[name] for name in ordered_names]
    panels = (
        ("loss", "Negative IWAE bound"),
        ("reconstruction_loss", "Reconstruction loss"),
        ("kl_loss", "KL loss"),
        ("ess_fraction", "ESS / evaluation K"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), squeeze=False)
    for axis, (metric, title) in zip(axes.flat, panels):
        means = []
        stddevs = []
        for name in ordered_names:
            mean = model_results[name]["mean"]
            stddev = model_results[name]["sample_stddev"]
            assert isinstance(mean, dict)
            assert isinstance(stddev, dict)
            means.append(mean[metric])
            stddevs.append(stddev[metric])
        axis.errorbar(
            particles,
            means,
            yerr=stddevs,
            color="#1f77b4",
            marker="o",
            capsize=4,
            linewidth=2,
        )
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_xticks(particles, [f"K={value}" for value in particles])
        axis.set_xlabel("Training particles (log2 scale)")
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle(
        "Held-out IWAE metrics by training particle count\n"
        "Evaluation K=128; markers show means and bars show sample standard deviation"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=200)
    plt.close(figure)


@torch.inference_mode()
def save_model_comparison(
    model: GaussianVAE,
    loader: DataLoader,
    path: Path,
    *,
    device: torch.device,
) -> None:
    real, _ = next(iter(loader))
    real = real[:DISPLAY_SAMPLES].to(device)
    reconstructions = model.reconstruct(real)
    samples = model.sample(real.shape[0], device=device)
    save_image(
        torch.cat((real, reconstructions, samples)),
        path,
        nrow=real.shape[0],
    )


def checkpoint_paths() -> dict[str, Path]:
    return {
        "vae_elbo_k1": IWAE_K1_CHECKPOINT,
        "iwae_k4": IWAE_K4_CHECKPOINT,
        "iwae_k8": IWAE_K8_CHECKPOINT,
        "iwae_k16": IWAE_K16_CHECKPOINT,
        "iwae_k32": IWAE_K32_CHECKPOINT,
    }


def require_checkpoints(paths: dict[str, Path]) -> None:
    """Fail before replacing evaluation artifacts when training is incomplete."""
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        formatted_paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing IWAE checkpoints. Run 4.0_iwae.py to train every default "
            f"K value before evaluating:\n{formatted_paths}"
        )


def evaluate() -> None:
    paths = checkpoint_paths()
    require_checkpoints(paths)
    device = try_gpu()
    loader = make_test_loader(device)
    out_dir = OUTPUT_DIR
    reset_dir(str(out_dir))
    results: dict[str, object] = {
        "protocol": {
            "dataset": "MNIST test",
            "image_size": IMAGE_SIZE,
            "observation": "independent Bernoulli mean",
            "evaluation_particles": EVALUATION_PARTICLES,
            "particle_chunk_size": PARTICLE_CHUNK_SIZE,
            "repeats": REPEATS,
            "max_examples": MAX_EXAMPLES,
        },
        "models": {},
    }
    model_results: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        model = load_model(path, device)
        summary = evaluate_model(
            model,
            loader,
            particles=EVALUATION_PARTICLES,
            particle_chunk_size=PARTICLE_CHUNK_SIZE,
            repeats=REPEATS,
            max_examples=MAX_EXAMPLES,
            seed=SEED,
            device=device,
        )
        model_results[name] = {
            "training_particles": TRAINING_PARTICLES[name],
            **summary,
        }
        save_model_comparison(
            model,
            loader,
            out_dir / f"{name}_real_reconstruction_prior.png",
            device=device,
        )
    baseline = model_results["vae_elbo_k1"]["mean"]
    assert isinstance(baseline, dict)
    for name, model_result in model_results.items():
        mean = model_result["mean"]
        assert isinstance(mean, dict)
        model_result["change_vs_k1"] = changes_from_baseline(mean, baseline)
        results["models"][name] = model_result
    save_metric_comparison(
        model_results, out_dir / "metrics_by_training_particles.png"
    )

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
