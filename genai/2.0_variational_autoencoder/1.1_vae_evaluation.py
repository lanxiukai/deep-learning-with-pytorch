"""Evaluate the frozen standard-VAE checkpoint without retraining it.

The script keeps three image paths separate:

* posterior-mean reconstruction is deterministic and reads a real image;
* posterior-sample reconstruction also reads a real image;
* prior generation decodes an independent ``N(0, I)`` draw.

Latent interpolation is a local decoder diagnostic, not evidence of
independent semantic factors or unconditional generation.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py. Folder
    labels are ignored by the standard VAE evaluation.

Checkpoint:
    output/vae/vae/vae.pth: frozen checkpoint from 1.0_vae.py

Outputs:
    output/vae/vae/evaluation/metrics.json: reconstruction and posterior metrics
    output/vae/vae/evaluation/real_mean_and_sample_reconstruction.png
    output/vae/vae/evaluation/standard_normal_prior_samples.png
    output/vae/vae/evaluation/posterior_mean_interpolation_not_generation.png

Evaluation data -- glasses-256:
Available images:           4,500
Batch size:                    16
Maximum batches:              100
Evaluated images:           1,600
Comparison images:              8
Prior samples:                  18
Interpolation steps:             7

Default dimensions:
Evaluation input:           256x256 RGB
Generated image:            256x256 RGB
Latent vector:                  100 values

Model size:
Frozen standard VAE:         63.33 M parameters
"""

import json
import math
from itertools import islice

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.data.vision import image_folder_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights
from dl_utils.vae.vae import VAE, diagonal_gaussian_kl, reparameterize

PROJECT_ROOT = infer_project_root()
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "vae" / "vae.pth"
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "vae" / "evaluation"

BATCH_SIZE = 16
MAXIMUM_BATCHES = 100
ACTIVE_VARIANCE_THRESHOLD = 1e-2
NUM_WORKERS = 0
SEED = 42
NUM_COMPARISON_IMAGES = 8
NUM_PRIOR_SAMPLES = 18
PRIOR_GRID_COLUMNS = 6
NUM_INTERPOLATION_STEPS = 7
MINIMUM_PSNR_ERROR = 1e-12


@torch.inference_mode()
def evaluate(
    model,
    loader,
    *,
    z_dim,
    maximum_batches,
    active_variance_threshold,
    device,
):
    """Evaluate posterior paths, prior samples, and a local interpolation."""
    if maximum_batches < 1:
        raise ValueError("maximum_batches must be positive")

    model.eval()
    mu_total = torch.zeros(z_dim, dtype=torch.float64)
    mu_square_total = torch.zeros_like(mu_total)
    kl_total = 0.0
    examples = 0
    squared_error_total = 0.0
    evaluated_elements = 0
    comparison = None
    interpolation = None
    limited_loader = islice(loader, maximum_batches)
    progress = tqdm(
        limited_loader,
        total=min(len(loader), maximum_batches),
        desc="Analyze VAE",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        mu, std = model.encoder.statistics(images)
        mean_reconstructions = model.decoder(mu)
        sample_reconstructions = model.decoder(reparameterize(mu, std))
        squared_error_total += float((mean_reconstructions - images).square().sum())
        evaluated_elements += images.numel()
        mu_total += mu.detach().double().sum(dim=0).cpu()
        mu_square_total += mu.detach().double().square().sum(dim=0).cpu()
        kl_total += float(diagonal_gaussian_kl(mu, std).sum())
        examples += images.shape[0]

        if comparison is None:
            comparison_count = min(NUM_COMPARISON_IMAGES, images.shape[0])
            comparison = torch.cat(
                (
                    images[:comparison_count],
                    mean_reconstructions[:comparison_count],
                    sample_reconstructions[:comparison_count],
                )
            ).cpu()
            if comparison_count >= 2:
                interpolation_weights = torch.linspace(
                    0,
                    1,
                    NUM_INTERPOLATION_STEPS,
                    device=device,
                )[:, None]
                latent_path = (1.0 - interpolation_weights) * mu[
                    0
                ] + interpolation_weights * mu[1]
                interpolation = model.decoder(latent_path).cpu()

    if evaluated_elements == 0 or comparison is None:
        raise ValueError("cannot analyze an empty loader")
    if interpolation is None:
        raise ValueError("at least two images are required for interpolation")

    mean_squared_error = squared_error_total / evaluated_elements
    posterior_mean = mu_total / examples
    variance_of_mu = (
        mu_square_total / examples - posterior_mean.square()
    ).clamp_min(0.0)
    metrics = {
        "posterior_mean_reconstruction": {
            "pixel_mse": mean_squared_error,
            "psnr_for_zero_to_one_range": -10.0
            * math.log10(max(mean_squared_error, MINIMUM_PSNR_ERROR)),
        },
        "posterior": {
            "examples": examples,
            "kl_nats_per_image": kl_total / examples,
            "active_variance_threshold": active_variance_threshold,
            "active_dimensions": int(
                (variance_of_mu > active_variance_threshold).sum()
            ),
        },
    }
    prior_samples = model.decoder(
        torch.randn(NUM_PRIOR_SAMPLES, z_dim, device=device)
    ).cpu()
    return metrics, comparison, prior_samples, interpolation


def analyze(device):
    """Load the final checkpoint and write all standard-VAE diagnostics."""
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {CHECKPOINT}; run 1.0_vae.py first"
        )
    model, model_config = load_model_weights(
        CHECKPOINT,
        VAE,
        device=device,
        expected_metadata={"model_name": "vae"},
    )
    z_dim = model_config.get("z_dim")
    if isinstance(z_dim, bool) or not isinstance(z_dim, int) or z_dim < 1:
        raise ValueError("checkpoint model_config has an invalid z_dim")

    loader = image_folder_loader(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    metrics, comparison, prior_samples, interpolation = evaluate(
        model,
        loader,
        z_dim=z_dim,
        maximum_batches=MAXIMUM_BATCHES,
        active_variance_threshold=ACTIVE_VARIANCE_THRESHOLD,
        device=device,
    )
    reset_dir(str(OUTPUT_DIR))
    save_image(
        comparison,
        OUTPUT_DIR / "real_mean_and_sample_reconstruction.png",
        nrow=comparison.shape[0] // 3,
    )
    save_image(
        prior_samples,
        OUTPUT_DIR / "standard_normal_prior_samples.png",
        nrow=PRIOR_GRID_COLUMNS,
    )
    save_image(
        interpolation,
        OUTPUT_DIR / "posterior_mean_interpolation_not_generation.png",
        nrow=len(interpolation),
    )
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    console_metrics = {
        "posterior_mean_reconstruction": metrics["posterior_mean_reconstruction"],
        "posterior": {
            "examples": metrics["posterior"]["examples"],
            "kl_nats_per_image": metrics["posterior"]["kl_nats_per_image"],
            "active_variance_threshold": metrics["posterior"][
                "active_variance_threshold"
            ],
            "active_dimensions": metrics["posterior"]["active_dimensions"],
        },
    }
    print(json.dumps(console_metrics, indent=2))


def main():
    set_seed(SEED)
    device = try_gpu()
    analyze(device)


if __name__ == "__main__":
    main()
