"""Compare the core rate-distortion behavior of VAE and beta-VAE.

Both checkpoints use the same glasses-256 images, model architecture, and
fixed prior samples. The evaluation focuses on reconstruction distortion,
KL rate, beta-weighted rate, and per-dimension latent use. These capacity
diagnostics do not establish semantic disentanglement.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py. Folder
    labels are ignored by both compared models.

Checkpoints:
    output/vae/vae/vae.pth: required standard VAE
    output/vae/beta_vae/beta_vae.pth: required beta-VAE

Outputs:
    output/vae/beta_vae/evaluation/metrics.json: rate-distortion report
    output/vae/beta_vae/evaluation/beta_comparison.png: summary figure
    output/vae/beta_vae/evaluation/<model>/real_and_posterior_mean_reconstruction.png
    output/vae/beta_vae/evaluation/<model>/matched_standard_normal_prior_samples.png

Evaluation data -- glasses-256:
Available images:             4,500
Batch size:                      16
Maximum batches:                100
Evaluated images per model:   1,600
Matched prior samples:            18
Active-KL threshold:             0.05 nats per dimension

Default dimensions:
Evaluation input:             256x256 RGB
Generated image:              256x256 RGB
Latent vector:                    100 values

Model size:
Standard VAE:                  63.33 M parameters
Beta-VAE:                     63.33 M parameters
Loaded total:                126.66 M parameters
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.vision import image_folder_dataset
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.vae import VAE
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
STANDARD_CHECKPOINT = OUTPUT_ROOT / "vae" / "vae.pth"
BETA_CHECKPOINT = OUTPUT_ROOT / "beta_vae" / "beta_vae.pth"
OUTPUT_DIR = OUTPUT_ROOT / "beta_vae" / "evaluation"

BATCH_SIZE = 16
MAXIMUM_BATCHES = 100
ACTIVE_KL_THRESHOLD = 0.05
NUM_WORKERS = 0
SEED = 42
NUM_COMPARISON_IMAGES = 8
NUM_PRIOR_SAMPLES = 18
PRIOR_GRID_COLUMNS = 6


@dataclass(frozen=True)
class CheckpointInfo:
    model_name: str
    beta: float
    z_dim: int
    model_config: dict[str, object]
    checkpoint: str


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    beta: float
    distortion: float
    rate: float
    weighted_rate: float
    active_kl_dimensions: int
    inactive_kl_dimensions: int
    kl_per_dimension: list[float]
    checkpoint: str


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[VAE, CheckpointInfo]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    model_name = checkpoint.get("model_name")
    if model_name not in {"vae", "beta_vae"}:
        raise ValueError(f"unsupported model_name: {model_name!r}")
    if checkpoint.get("dataset") != "glasses-256":
        raise ValueError(f"checkpoint does not use glasses-256: {path}")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(
        state_dict, Mapping
    ):
        raise TypeError(f"checkpoint is missing model construction data: {path}")
    z_dim = model_config.get("z_dim")
    if isinstance(z_dim, bool) or not isinstance(z_dim, int) or z_dim < 1:
        raise ValueError(f"checkpoint has an invalid z_dim: {path}")
    beta = checkpoint.get("beta")
    if isinstance(beta, bool) or not isinstance(beta, int | float) or beta < 0:
        raise ValueError(f"checkpoint has an invalid beta: {path}")

    saved_config = dict(model_config)
    model = VAE(**saved_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model.eval(), CheckpointInfo(
        model_name=model_name,
        beta=float(beta),
        z_dim=z_dim,
        model_config=saved_config,
        checkpoint=str(path),
    )


def summarize_kl(
    kl_total: Tensor,
    *,
    examples: int,
    active_kl_threshold: float,
) -> tuple[float, list[float], int]:
    if examples < 1:
        raise ValueError("examples must be positive")
    kl_per_dimension = kl_total / examples
    return (
        kl_per_dimension.sum().item(),
        kl_per_dimension.tolist(),
        int((kl_per_dimension > active_kl_threshold).sum()),
    )


@torch.inference_mode()
def evaluate_model(
    model: VAE,
    loader: DataLoader,
    *,
    info: CheckpointInfo,
    maximum_batches: int,
    active_kl_threshold: float,
    fixed_z: Tensor,
    device: torch.device,
) -> tuple[EvaluationResult, Tensor, Tensor]:
    if maximum_batches < 1:
        raise ValueError("maximum_batches must be positive")
    squared_error_total = 0.0
    kl_total = torch.zeros(info.z_dim, dtype=torch.float64)
    examples = 0
    comparison = None
    for images, _ in islice(loader, maximum_batches):
        images = images.to(device, non_blocking=True)
        mu, logvar = model.encode(images)
        reconstructions = model.decoder(mu)
        squared_error_total += (reconstructions - images).square().sum().item()
        kl_total += (
            diagonal_gaussian_kl_from_logvar(mu, logvar)
            .detach()
            .double()
            .sum(dim=0)
            .cpu()
        )
        examples += images.shape[0]
        if comparison is None:
            count = min(NUM_COMPARISON_IMAGES, images.shape[0])
            comparison = torch.cat(
                (images[:count], reconstructions[:count])
            ).cpu()

    if comparison is None or examples == 0:
        raise ValueError("cannot evaluate an empty loader")
    rate, kl_per_dimension, active_dimensions = summarize_kl(
        kl_total,
        examples=examples,
        active_kl_threshold=active_kl_threshold,
    )
    result = EvaluationResult(
        model_name=info.model_name,
        beta=info.beta,
        distortion=squared_error_total / examples,
        rate=rate,
        weighted_rate=info.beta * rate,
        active_kl_dimensions=active_dimensions,
        inactive_kl_dimensions=info.z_dim - active_dimensions,
        kl_per_dimension=kl_per_dimension,
        checkpoint=info.checkpoint,
    )
    prior_samples = model.decoder(fixed_z).cpu()
    return result, comparison, prior_samples


def save_summary_plot(
    results: list[EvaluationResult], output_path: Path
) -> None:
    ordered = sorted(results, key=lambda result: result.beta)
    with plt.ioff():
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for index, result in enumerate(ordered):
            axes[0].scatter(result.rate, result.distortion)
            axes[0].annotate(
                f"beta={result.beta:g}",
                (result.rate, result.distortion),
                xytext=(5, 6 + 12 * index),
                textcoords="offset points",
            )
            axes[1].plot(
                range(1, len(result.kl_per_dimension) + 1),
                result.kl_per_dimension,
                label=f"beta={result.beta:g}",
            )
        axes[0].set(
            xlabel="KL rate (nats / image)",
            ylabel="Posterior-mean summed-pixel MSE / image",
            title="Rate-distortion comparison",
        )
        axes[1].set(
            xlabel="Latent dimension",
            ylabel="Mean KL (nats / image)",
            title="Per-dimension latent rate",
        )
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def analyze() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"dataset not found: {DATA_DIR}")
    set_seed(SEED)
    device = try_gpu()
    standard_model, standard_info = load_checkpoint(
        STANDARD_CHECKPOINT, device
    )
    beta_model, beta_info = load_checkpoint(BETA_CHECKPOINT, device)
    if standard_info.model_name != "vae":
        raise ValueError("STANDARD_CHECKPOINT must contain the standard VAE")
    if beta_info.model_name != "beta_vae":
        raise ValueError("BETA_CHECKPOINT must contain a beta-VAE")
    if standard_info.model_config != beta_info.model_config:
        raise ValueError("compared checkpoints must use the same model_config")

    loader = DataLoader(
        image_folder_dataset(DATA_DIR),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    fixed_z = torch.randn(
        NUM_PRIOR_SAMPLES, standard_info.z_dim, device=device
    )
    runs = (
        ("standard_vae", standard_model, standard_info),
        ("beta_vae", beta_model, beta_info),
    )
    results = []
    artifacts = []
    for run_name, model, info in runs:
        result, comparison, prior_samples = evaluate_model(
            model,
            loader,
            info=info,
            maximum_batches=MAXIMUM_BATCHES,
            active_kl_threshold=ACTIVE_KL_THRESHOLD,
            fixed_z=fixed_z,
            device=device,
        )
        results.append(result)
        artifacts.append((run_name, comparison, prior_samples))

    reset_dir(str(OUTPUT_DIR))
    for run_name, comparison, prior_samples in artifacts:
        run_dir = OUTPUT_DIR / run_name
        reset_dir(str(run_dir))
        save_image(
            comparison,
            run_dir / "real_and_posterior_mean_reconstruction.png",
            nrow=comparison.shape[0] // 2,
        )
        save_image(
            prior_samples,
            run_dir / "matched_standard_normal_prior_samples.png",
            nrow=PRIOR_GRID_COLUMNS,
        )
    report = {
        "protocol": {
            "dataset": "glasses-256",
            "distortion": "posterior-mean summed-pixel MSE per image",
            "active_kl_threshold_nats": ACTIVE_KL_THRESHOLD,
            "interpretation_warning": (
                "Capacity diagnostics do not establish semantic "
                "disentanglement without ground-truth factors."
            ),
        },
        "runs": [asdict(result) for result in results],
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    save_summary_plot(results, OUTPUT_DIR / "beta_comparison.png")


def main() -> None:
    analyze()


if __name__ == "__main__":
    main()
