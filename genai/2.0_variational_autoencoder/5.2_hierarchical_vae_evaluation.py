"""Test whether each stochastic VAE layer adds information beyond its prior.

For both the ordinary HVAE and Ladder posterior, this script reports per-layer
rate and active units, then performs three sampled reconstruction
counterfactuals:

* replace only q(z1 | z2, x) with p(z1 | z2);
* replace only q(z2 | x) with p(z2), retaining lower data evidence;
* replace both posterior layers with their matching priors.

It also saves two controlled resampling grids.  Any apparent global/local
division remains an empirical observation, not a property implied by the
hierarchical ELBO.

Data:
    FactorShapes32 test split, generated with the same split seed as training.
    Factor annotations do not enter the models.

Checkpoints:
    output/vae/hierarchical_vae/baseline/hierarchical_vae.pth: default HVAE
    output/vae/ladder_vae/baseline/ladder_vae.pth: default Ladder VAE
    Run both 5.0 and 5.1 first to produce the compared checkpoints.

Outputs:
    output/vae/hierarchical_vae/evaluation/metrics.json: comparison report
    output/vae/hierarchical_vae/evaluation/<model>/posterior_and_prior_replacements.png
    output/vae/hierarchical_vae/evaluation/<model>/fixed_top_resampled_lower_prior.png
    output/vae/hierarchical_vae/evaluation/<model>/fixed_evidence_changed_top.png

Evaluation data -- FactorShapes32 test:
Available and evaluated images:       706
Batch size:                            64
Intervention samples per image:         8
Visual variants per fixed input:        6
Active-variance threshold:           0.01

Default dimensions:
Evaluation input:                32x32 grayscale
Generated image:                 32x32 grayscale
Latent vectors:                  z1=24 / z2=12 values

Model size:
Hierarchical VAE:                 0.876 M parameters
Ladder VAE:                       0.874 M parameters
Loaded total:                     1.750 M parameters
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dl_utils.data.factor_shapes import FactorShapes32
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.vae_common import (
    diagonal_gaussian_kl_from_logvar,
    reparameterize_logvar,
)
from dl_utils.vae.vae_hierarchy import (
    ActiveUnitAccumulator,
    HierarchicalVAE32,
    LadderVAE32,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
OUTPUT_DIR = OUTPUT_ROOT / "hierarchical_vae" / "evaluation"
DISPLAY_SAMPLES = 8


# Edit these defaults to explore the lesson.
BATCH_SIZE = 64
INTERVENTION_SAMPLES = 8
VISUAL_VARIANTS = 6
ACTIVE_VARIANCE_THRESHOLD = 1e-2
WORKERS = 4
SEED = 123
SPLIT_SEED = 2026
HVAE_CHECKPOINT = OUTPUT_ROOT / "hierarchical_vae" / "baseline" / "hierarchical_vae.pth"
LADDER_CHECKPOINT = OUTPUT_ROOT / "ladder_vae" / "baseline" / "ladder_vae.pth"


def load_model(
    path: Path, device: torch.device
) -> tuple[HierarchicalVAE32, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_name = checkpoint.get("model_name")
    if model_name == "hierarchical_vae":
        model: HierarchicalVAE32 = HierarchicalVAE32(**checkpoint["model_config"])
    elif model_name == "ladder_vae":
        model = LadderVAE32(**checkpoint["model_config"])
    else:
        raise ValueError(f"{path} is not an HVAE/Ladder checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def make_test_loader(
    *,
    split_seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = FactorShapes32(split="test", split_seed=split_seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=WORKERS > 0,
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


def _decode_distortion(model: HierarchicalVAE32, z1: Tensor, target: Tensor) -> Tensor:
    reconstruction = model.decode(z1)
    expanded_target = target[:, None, ...].expand_as(reconstruction)
    return (
        F.binary_cross_entropy(reconstruction, expanded_target, reduction="none")
        .flatten(2)
        .sum(dim=2)
    )


@torch.inference_mode()
def sampled_counterfactual_distortions(
    model: HierarchicalVAE32,
    images: Tensor,
    *,
    samples: int,
) -> dict[str, Tensor]:
    """Use paired top samples to isolate lower and upper posterior removal."""
    lower_evidence, q2_mu, q2_logvar = model.bottom_up(images)
    batch_size = images.shape[0]
    repeated_evidence = (
        lower_evidence[:, None, :]
        .expand(-1, samples, -1)
        .reshape(batch_size * samples, -1)
    )

    q_z2 = _sample_gaussian(q2_mu, q2_logvar, samples)
    q_z2_flat = q_z2.reshape(batch_size * samples, model.z2_dim)
    p1_mu, p1_logvar, q1_mu, q1_logvar = model.lower_distributions(
        repeated_evidence, q_z2_flat
    )
    posterior_z1 = reparameterize_logvar(q1_mu, q1_logvar).reshape(
        batch_size, samples, model.z1_dim
    )
    lower_prior_z1 = reparameterize_logvar(p1_mu, p1_logvar).reshape(
        batch_size, samples, model.z1_dim
    )

    p_z2 = torch.randn_like(q_z2)
    p_z2_flat = p_z2.reshape(batch_size * samples, model.z2_dim)
    p_top_p1_mu, p_top_p1_logvar, p_top_q1_mu, p_top_q1_logvar = (
        model.lower_distributions(repeated_evidence, p_z2_flat)
    )
    top_replaced_z1 = reparameterize_logvar(p_top_q1_mu, p_top_q1_logvar).reshape(
        batch_size, samples, model.z1_dim
    )
    both_replaced_z1 = reparameterize_logvar(p_top_p1_mu, p_top_p1_logvar).reshape(
        batch_size, samples, model.z1_dim
    )
    return {
        "posterior": _decode_distortion(model, posterior_z1, images),
        "lower_prior": _decode_distortion(model, lower_prior_z1, images),
        "top_prior": _decode_distortion(model, top_replaced_z1, images),
        "both_priors": _decode_distortion(model, both_replaced_z1, images),
    }


@torch.inference_mode()
def evaluate_model(
    model: HierarchicalVAE32,
    loader: DataLoader,
    *,
    intervention_samples: int,
    active_variance_threshold: float,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "posterior": 0.0,
        "lower_prior": 0.0,
        "top_prior": 0.0,
        "both_priors": 0.0,
    }
    deterministic_distortion = 0.0
    kl_z1 = 0.0
    kl_z2 = 0.0
    examples = 0
    active = ActiveUnitAccumulator()
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        latents = model.infer(images, sample=False)
        reconstruction = model.decode(latents["z1"])
        deterministic_distortion += float(
            F.binary_cross_entropy(reconstruction, images, reduction="sum")
        )
        kl_z1 += float(
            diagonal_gaussian_kl_from_logvar(
                latents["q1_mu"],
                latents["q1_logvar"],
                latents["p1_mu"],
                latents["p1_logvar"],
            ).sum()
        )
        kl_z2 += float(
            diagonal_gaussian_kl_from_logvar(
                latents["q2_mu"], latents["q2_logvar"]
            ).sum()
        )
        active.update(latents)
        counterfactuals = sampled_counterfactual_distortions(
            model, images, samples=intervention_samples
        )
        for name, values in counterfactuals.items():
            totals[name] += float(values.sum()) / intervention_samples
        examples += images.shape[0]
    active_z1, active_z2 = active.counts(variance_threshold=active_variance_threshold)
    sampled_posterior = totals["posterior"] / examples
    return {
        "posterior_mean_distortion": (deterministic_distortion / examples),
        "sampled_posterior_distortion": sampled_posterior,
        "lower_prior_replacement_distortion": (totals["lower_prior"] / examples),
        "top_prior_replacement_distortion": (totals["top_prior"] / examples),
        "both_priors_replacement_distortion": (totals["both_priors"] / examples),
        "lower_replacement_delta": (
            totals["lower_prior"] / examples - sampled_posterior
        ),
        "top_replacement_delta": (totals["top_prior"] / examples - sampled_posterior),
        "both_replacement_delta": (
            totals["both_priors"] / examples - sampled_posterior
        ),
        "kl_z1": kl_z1 / examples,
        "kl_z2": kl_z2 / examples,
        "total_rate": (kl_z1 + kl_z2) / examples,
        "active_z1_corrections": float(active_z1),
        "active_z2_units": float(active_z2),
    }


@torch.inference_mode()
def save_counterfactual_grids(
    model: HierarchicalVAE32,
    loader: DataLoader,
    out_dir: Path,
    *,
    variants: int,
    device: torch.device,
) -> dict[str, float]:
    images, _ = next(iter(loader))
    images = images[:DISPLAY_SAMPLES].to(device)
    lower_evidence, q2_mu, q2_logvar = model.bottom_up(images)
    posterior = model.infer_from_top(
        lower_evidence,
        q2_mu,
        q2_logvar,
        q2_mu,
        sample_lower=False,
    )
    zero_z2 = torch.zeros_like(q2_mu)
    top_replaced = model.infer_from_top(
        lower_evidence,
        q2_mu,
        q2_logvar,
        zero_z2,
        sample_lower=False,
    )
    zero_p1_mu, _, _, _ = model.lower_distributions(lower_evidence, zero_z2)
    summary = torch.cat(
        (
            images,
            model.decode(posterior["q1_mu"]),
            model.decode(posterior["p1_mu"]),
            model.decode(top_replaced["q1_mu"]),
            model.decode(zero_p1_mu),
        )
    )
    save_image(
        summary,
        out_dir / "posterior_and_prior_replacements.png",
        nrow=images.shape[0],
    )

    fixed_top = q2_mu
    p1_mu, p1_logvar, _, _ = model.lower_distributions(lower_evidence, fixed_top)
    lower_samples = _sample_gaussian(p1_mu, p1_logvar, variants)
    lower_images = model.decode(lower_samples)
    save_image(
        lower_images.flatten(0, 1),
        out_dir / "fixed_top_resampled_lower_prior.png",
        nrow=variants,
    )

    top_samples = _sample_gaussian(q2_mu, q2_logvar, variants)
    repeated_evidence = (
        lower_evidence[:, None, :]
        .expand(-1, variants, -1)
        .reshape(-1, lower_evidence.shape[1])
    )
    _, _, changed_q1_mu, _ = model.lower_distributions(
        repeated_evidence,
        top_samples.reshape(-1, model.z2_dim),
    )
    upper_images = model.decode(
        changed_q1_mu.reshape(images.shape[0], variants, model.z1_dim)
    )
    save_image(
        upper_images.flatten(0, 1),
        out_dir / "fixed_evidence_changed_top.png",
        nrow=variants,
    )
    return {
        "fixed_top_lower_pixel_standard_deviation": float(
            lower_images.std(dim=1).mean()
        ),
        "fixed_evidence_top_pixel_standard_deviation": float(
            upper_images.std(dim=1).mean()
        ),
    }


def checkpoint_paths() -> dict[str, Path]:
    return {
        "hierarchical_vae": HVAE_CHECKPOINT,
        "ladder_vae": LADDER_CHECKPOINT,
    }


def evaluate() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = checkpoint_paths()
    loader = make_test_loader(split_seed=SPLIT_SEED, device=device)
    out_root = OUTPUT_DIR
    reset_dir(str(out_root))
    results: dict[str, object] = {
        "protocol": {
            "dataset": "FactorShapes32 test",
            "intervention_samples": INTERVENTION_SAMPLES,
            "active_variance_threshold": ACTIVE_VARIANCE_THRESHOLD,
            "interpretation": (
                "Layer interventions test incremental information; they do "
                "not prove a semantic global/local hierarchy."
            ),
        },
        "models": {},
    }
    for name, path in paths.items():
        model, checkpoint = load_model(path, device)
        metrics = evaluate_model(
            model,
            loader,
            intervention_samples=INTERVENTION_SAMPLES,
            active_variance_threshold=ACTIVE_VARIANCE_THRESHOLD,
            device=device,
        )
        model_out_dir = out_root / name
        reset_dir(str(model_out_dir))
        metrics.update(
            save_counterfactual_grids(
                model,
                loader,
                model_out_dir,
                variants=VISUAL_VARIANTS,
                device=device,
            )
        )
        metrics["checkpoint"] = str(path.relative_to(PROJECT_ROOT))
        metrics["posterior_family"] = checkpoint["posterior_family"]
        metrics["warmup_epochs"] = checkpoint["warmup_epochs"]
        metrics["free_bits_per_group"] = checkpoint["free_bits_per_group"]
        results["models"][name] = metrics
        print(
            f"{name}: KL1={metrics['kl_z1']:.3f}, "
            f"KL2={metrics['kl_z2']:.3f}, "
            f"lower delta={metrics['lower_replacement_delta']:.3f}, "
            f"top delta={metrics['top_replacement_delta']:.3f}"
        )
    (out_root / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved evaluation to {out_root}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
