"""Test whether each stochastic VAE layer adds information beyond its prior.

For both the ordinary HVAE and Ladder posterior, this script reports per-layer
KL and reconstruction distortion increases from two sampled interventions:

* replace only q(z1 | z2, x) with p(z1 | z2);
* replace only q(z2 | x) with p(z2), retaining lower data evidence.

It also saves a four-row mean-path reconstruction comparison and two controlled
resampling grids. The mean-path comparison uses z2=0 for top replacement;
the numerical interventions sample from the corresponding distributions.
Any apparent global/local division remains an empirical observation, not a
property implied by the hierarchical ELBO.

Data:
    data/glasses-256, read directly as 256x256 RGB in [0, 1].
    All 4,500 training images are used; labels are ignored. These are
    training-set diagnostics, not held-out generalization estimates.
    Distortion is summed RGB MSE per image, matching the training objective.
    Reported KL uses posterior means for deterministic layer diagnostics;
    it is not a Monte Carlo estimate of the hierarchical ELBO.

Checkpoints:
    output/vae/hierarchical_vae/hierarchical_vae.pth: default HVAE
    output/vae/ladder_vae/ladder_vae.pth: default Ladder VAE
    Run both 5.0 and 5.1 first to produce the compared checkpoints.

Outputs:
    output/vae/ladder_vae/evaluation/metrics.json: comparison report
    output/vae/ladder_vae/evaluation/<model>/posterior_and_prior_replacements.png
    output/vae/ladder_vae/evaluation/<model>/fixed_top_resampled_lower_prior.png
    output/vae/ladder_vae/evaluation/<model>/fixed_evidence_changed_top.png
    output/vae/ladder_vae/evaluation/metric_comparison.png

Evaluation defaults:
    Training-set diagnostic images: all 4,500; batch size: 16.
    Intervention samples per image: 8; visual variants per input: 6.
    Decode batch size: 16, including interventions and visual grids.
    Model dimensions are loaded from the checkpoint (z1=96 / z2=32 by default).
    Old FactorShapes32 checkpoints must be replaced by rerunning 5.0 and 5.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.glasses import glasses_data_config, glasses_dataset
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.inference import generate_in_batches
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.hierarchical_vae import (
    HierarchicalVAE,
    LadderVAE,
)
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
OUTPUT_DIR = OUTPUT_ROOT / "ladder_vae" / "evaluation"
DISPLAY_SAMPLES = 8


# Edit these defaults to explore the lesson.
BATCH_SIZE = 16
DECODE_BATCH_SIZE = 16
INTERVENTION_SAMPLES = 8
VISUAL_VARIANTS = 6
WORKERS = 4
SEED = 123
HVAE_CHECKPOINT = OUTPUT_ROOT / "hierarchical_vae" / "hierarchical_vae.pth"
LADDER_CHECKPOINT = OUTPUT_ROOT / "ladder_vae" / "ladder_vae.pth"


def load_model(
    path: Path, device: torch.device
) -> tuple[HierarchicalVAE, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("data_config") != glasses_data_config():
        raise ValueError(
            f"{path} is not a glasses-256 RGB hierarchy checkpoint; "
            "rerun 5.0_hierarchical_vae.py and 5.1_ladder_vae.py"
        )
    model_name = checkpoint.get("model_name")
    if model_name == "hierarchical_vae":
        model: HierarchicalVAE = HierarchicalVAE(**checkpoint["model_config"])
    elif model_name == "ladder_vae":
        model = LadderVAE(**checkpoint["model_config"])
    else:
        raise ValueError(f"{path} is not an HVAE/Ladder checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def make_evaluation_loader(device: torch.device) -> DataLoader:
    return make_device_aware_loader(
        glasses_dataset(DATA_DIR),
        BATCH_SIZE,
        device,
        shuffle=False,
        num_workers=WORKERS,
    )


def _sample_gaussian(mu: Tensor, logvar: Tensor, samples: int) -> Tensor:
    epsilon = torch.randn(
        mu.shape[0],
        samples,
        mu.shape[1],
        device=mu.device,
        dtype=mu.dtype,
    )
    return mu[:, None, :] + torch.exp(0.5 * logvar[:, None, :]) * epsilon


def _decode_distortion(model: HierarchicalVAE, z1: Tensor, target: Tensor) -> Tensor:
    batch_size, samples = z1.shape[:2]
    image_indices = torch.arange(batch_size, device=target.device).repeat_interleave(
        samples
    )
    # Reduce each decoded chunk immediately instead of retaining RGB particles.
    return generate_in_batches(
        (z1.flatten(0, 1), image_indices),
        DECODE_BATCH_SIZE,
        lambda latent, indices: (model.decode(latent) - target[indices])
        .square()
        .flatten(1)
        .sum(dim=1),
        output_device=target.device,
    ).reshape(batch_size, samples)


def _decode_grid(model: HierarchicalVAE, z1: Tensor) -> Tensor:
    return generate_in_batches(
        z1.reshape(-1, model.z1_dim), DECODE_BATCH_SIZE, model.decode
    ).reshape(*z1.shape[:-1], 3, 256, 256)


@torch.inference_mode()
def sampled_counterfactual_distortions(
    model: HierarchicalVAE,
    images: Tensor,
    *,
    samples: int,
) -> dict[str, Tensor]:
    """Use paired top samples to isolate lower and upper posterior removal."""
    h_1, mu_q_2, v_q_2 = model.q_2(images)
    batch_size = images.shape[0]
    repeated_evidence = (
        h_1[:, None, :]
        .expand(-1, samples, -1)
        .reshape(batch_size * samples, -1)
    )

    q_z2 = _sample_gaussian(mu_q_2, v_q_2, samples)
    q_z2_flat = q_z2.reshape(batch_size * samples, model.z2_dim)
    parameters_1 = model.lower_parameters(repeated_evidence, q_z2_flat)
    posterior_z1 = reparameterize_logvar(
        parameters_1["mu_q_1"], parameters_1["v_q_1"]
    ).reshape(batch_size, samples, model.z1_dim)
    lower_prior_z1 = reparameterize_logvar(
        parameters_1["mu_p_1"], parameters_1["v_p_1"]
    ).reshape(batch_size, samples, model.z1_dim)

    p_z2 = torch.randn_like(q_z2)
    p_z2_flat = p_z2.reshape(batch_size * samples, model.z2_dim)
    replaced_parameters_1 = model.lower_parameters(repeated_evidence, p_z2_flat)
    top_replaced_z1 = reparameterize_logvar(
        replaced_parameters_1["mu_q_1"], replaced_parameters_1["v_q_1"]
    ).reshape(batch_size, samples, model.z1_dim)
    return {
        "posterior": _decode_distortion(model, posterior_z1, images),
        "lower_prior": _decode_distortion(model, lower_prior_z1, images),
        "top_prior": _decode_distortion(model, top_replaced_z1, images),
    }


@torch.inference_mode()
def evaluate_model(
    model: HierarchicalVAE,
    loader: DataLoader,
    *,
    intervention_samples: int,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "posterior": 0.0,
        "lower_prior": 0.0,
        "top_prior": 0.0,
    }
    kl_z1 = 0.0
    kl_z2 = 0.0
    examples = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        latents = model.infer(images, sample=False)
        kl_z1 += (
            diagonal_gaussian_kl_from_logvar(
                latents["mu_q_1"],
                latents["v_q_1"],
                latents["mu_p_1"],
                latents["v_p_1"],
            )
            .sum()
            .item()
        )
        kl_z2 += (
            diagonal_gaussian_kl_from_logvar(
                latents["mu_q_2"], latents["v_q_2"]
            )
            .sum()
            .item()
        )
        counterfactuals = sampled_counterfactual_distortions(
            model, images, samples=intervention_samples
        )
        for name, values in counterfactuals.items():
            totals[name] += values.sum().item() / intervention_samples
        examples += images.shape[0]
    sampled_posterior = totals["posterior"] / examples
    return {
        "sampled_posterior_distortion": sampled_posterior,
        "lower_replacement_delta": (
            totals["lower_prior"] / examples - sampled_posterior
        ),
        "top_replacement_delta": (totals["top_prior"] / examples - sampled_posterior),
        "kl_z1": kl_z1 / examples,
        "kl_z2": kl_z2 / examples,
    }


@torch.inference_mode()
def save_counterfactual_grids(
    model: HierarchicalVAE,
    loader: DataLoader,
    out_dir: Path,
    *,
    variants: int,
    device: torch.device,
) -> None:
    images, _ = next(iter(loader))
    images = images[:DISPLAY_SAMPLES].to(device)
    h_1, mu_q_2, v_q_2 = model.q_2(images)
    posterior = model.lower_parameters(h_1, mu_q_2)
    zero_z2 = torch.zeros_like(mu_q_2)
    top_replaced = model.lower_parameters(h_1, zero_z2)
    summary = torch.cat(
        (
            images.cpu(),
            _decode_grid(model, posterior["mu_q_1"]),
            _decode_grid(model, posterior["mu_p_1"]),
            _decode_grid(model, top_replaced["mu_q_1"]),
        )
    )
    save_image(
        summary,
        out_dir / "posterior_and_prior_replacements.png",
        nrow=images.shape[0],
    )

    lower_samples = _sample_gaussian(
        posterior["mu_p_1"], posterior["v_p_1"], variants
    )
    lower_images = _decode_grid(model, lower_samples)
    save_image(
        lower_images.flatten(0, 1),
        out_dir / "fixed_top_resampled_lower_prior.png",
        nrow=variants,
    )

    top_samples = _sample_gaussian(mu_q_2, v_q_2, variants)
    repeated_evidence = (
        h_1[:, None, :]
        .expand(-1, variants, -1)
        .reshape(-1, h_1.shape[1])
    )
    changed_parameters_1 = model.lower_parameters(
        repeated_evidence,
        top_samples.reshape(-1, model.z2_dim),
    )
    upper_images = _decode_grid(
        model,
        changed_parameters_1["mu_q_1"].reshape(images.shape[0], variants, model.z1_dim),
    )
    save_image(
        upper_images.flatten(0, 1),
        out_dir / "fixed_evidence_changed_top.png",
        nrow=variants,
    )


def checkpoint_paths() -> dict[str, Path]:
    return {
        "hierarchical_vae": HVAE_CHECKPOINT,
        "ladder_vae": LADDER_CHECKPOINT,
    }


def save_metric_comparison(
    model_metrics: dict[str, dict[str, float]], output_path: Path
) -> None:
    """Compare per-layer KL and single-layer intervention impact."""
    names = list(model_metrics)
    positions = list(range(len(names)))
    panels = (
        ("Layer KL", ("kl_z1", "kl_z2")),
        (
            "Replacement distortion increase",
            (
                "lower_replacement_delta",
                "top_replacement_delta",
            ),
        ),
    )
    with plt.ioff():
        figure, axes = plt.subplots(1, len(panels), figsize=(10, 4))
        for axis, (title, metric_names) in zip(axes, panels):
            width = 0.8 / len(metric_names)
            for index, metric_name in enumerate(metric_names):
                offset = (index - (len(metric_names) - 1) / 2) * width
                axis.bar(
                    [position + offset for position in positions],
                    [model_metrics[name][metric_name] for name in names],
                    width=width,
                    label=metric_name.removesuffix("_delta").replace("_", " "),
                )
            axis.set(title=title, xticks=positions, xticklabels=names)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    paths = checkpoint_paths()
    loader = make_evaluation_loader(device)
    out_root = OUTPUT_DIR
    reset_dir(str(out_root))
    results: dict[str, object] = {
        "protocol": {
            **glasses_data_config(),
            "held_out": False,
            "evaluated_examples": len(loader.dataset),
            "seed": SEED,
            "distortion": "summed RGB MSE per image",
            "kl_evaluation": "at posterior-mean top latent (diagnostic, not ELBO)",
            "summary_rows": [
                "original",
                "posterior means",
                "lower prior mean",
                "top prior mean (z2=0) with lower evidence",
            ],
            "intervention_samples": INTERVENTION_SAMPLES,
            "interpretation": (
                "Layer interventions test incremental information; they do "
                "not prove a semantic global/local hierarchy."
            ),
        },
        "models": {},
    }
    for name, path in paths.items():
        model, checkpoint = load_model(path, device)
        set_seed(SEED)
        metrics = evaluate_model(
            model,
            loader,
            intervention_samples=INTERVENTION_SAMPLES,
            device=device,
        )
        model_out_dir = out_root / name
        reset_dir(str(model_out_dir))
        save_counterfactual_grids(
            model,
            loader,
            model_out_dir,
            variants=VISUAL_VARIANTS,
            device=device,
        )
        metrics["checkpoint"] = str(path.relative_to(PROJECT_ROOT))
        metrics["posterior_family"] = checkpoint["posterior_family"]
        metrics["warmup_epochs"] = checkpoint["warmup_epochs"]
        metrics["free_bits_per_group"] = checkpoint["free_bits_per_group"]
        metrics["model_config"] = checkpoint["model_config"]
        metrics["training_config"] = checkpoint["training_config"]
        results["models"][name] = metrics
    save_metric_comparison(results["models"], out_root / "metric_comparison.png")
    (out_root / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
