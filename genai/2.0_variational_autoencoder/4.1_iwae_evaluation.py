"""Compare IWAE particle counts under one held-out protocol.

Both default K=1 and K=5 checkpoints are re-evaluated with the same large K,
particle chunk size, test examples, and repeated random estimates. This keeps
the training objective separate from the evaluation estimator. Loss,
reconstruction loss, KL loss, and ESS/K are reported for each model.

Data:
    data/mnist, downloaded automatically by torchvision when absent. Digit
    labels are ignored by the IWAE comparison.

Checkpoints:
    output/vae/iwae/k1/iwae.pth: default K=1 model
    output/vae/iwae/k5/iwae.pth: default K=5 model
    Run 4.0_iwae.py first to produce both checkpoints.

Outputs:
    output/vae/iwae/evaluation/metrics.json: shared-protocol comparison
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

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
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
IWAE_K5_CHECKPOINT = OUTPUT_ROOT / "iwae" / "k5" / "iwae.pth"


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
) -> dict[str, float]:
    totals: dict[str, float] = {}

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
            totals[name] = totals.get(name, 0.0) + value

    return {name: value / repeats for name, value in totals.items()}


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
        "iwae_k5": IWAE_K5_CHECKPOINT,
    }


def evaluate() -> None:
    device = try_gpu()
    loader = make_test_loader(device)
    paths = checkpoint_paths()
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
    for name, path in paths.items():
        model = load_model(path, device)
        metrics = evaluate_model(
            model,
            loader,
            particles=EVALUATION_PARTICLES,
            particle_chunk_size=PARTICLE_CHUNK_SIZE,
            repeats=REPEATS,
            max_examples=MAX_EXAMPLES,
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
            f"{name}: loss={metrics['loss']:.3f}, "
            f"reconstruction={metrics['reconstruction_loss']:.3f}, "
            f"KL={metrics['kl_loss']:.3f}, "
            f"ESS/K={metrics['ess_fraction']:.3f}"
        )

    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved comparison to {out_dir}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
