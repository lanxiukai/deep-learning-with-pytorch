"""Evaluate a frozen cVAE checkpoint without training any model.

Generate faces with/without glasses from p(z | c), sharing base noise between
class rows. Report sampled MSE + conditional KL and mean reconstructions on
the training dataset. These diagnostics do not measure held-out generalization
or establish quantitative superiority over the cGAN.

Data:
    data/glasses-256, read directly without resizing or normalization.
    Same class indices as the cGAN: G=0, NoG=1; model inputs are in [0, 1].

Checkpoint:
    output/vae/conditional_vae/conditional_vae.pth: saved by 3.0_conditional_vae.py

Outputs:
    output/vae/conditional_vae/evaluation/metrics.json
    output/vae/conditional_vae/evaluation/conditional_samples.png: G then NoG rows
    output/vae/conditional_vae/evaluation/real_reconstruction.png: alternating
        original / posterior-mean reconstruction rows for G, then NoG
    output/vae/conditional_vae/evaluation/metric_summary.png

Evaluation defaults:
    Training-set diagnostic images: all 4,500; batch size: 16.
    Generated images: 8 per class.
    Input and generated images: 256x256 RGB.
    Model dimensions are loaded from the checkpoint.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from dl_utils.data.glasses import (
    GLASSES_CLASS_NAMES,
    glasses_data_config,
    glasses_dataset,
)
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights
from dl_utils.vae.conditional_vae import (
    CVAE_OBJECTIVE,
    ConditionalVAE,
    evaluate_cvae,
    save_conditional_metric_summary,
    save_conditional_reconstructions,
    save_conditional_samples,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "conditional_vae" / "conditional_vae.pth"
OUTPUT_DIR = CHECKPOINT.parent / "evaluation"

# Edit these defaults to explore the lesson.
BATCH_SIZE = 16
SAMPLES_PER_CLASS = 8
WORKERS = 4
SEED = 123


def make_evaluation_loader(device: torch.device) -> DataLoader:
    return make_device_aware_loader(
        glasses_dataset(DATA_DIR),
        BATCH_SIZE,
        device,
        shuffle=False,
        num_workers=WORKERS,
    )


@torch.inference_mode()
def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {CHECKPOINT}; run 3.0_conditional_vae.py first"
        )
    data_config = glasses_data_config()
    model, _ = load_model_weights(
        CHECKPOINT,
        ConditionalVAE,
        device=device,
        expected_metadata={
            "model_name": "conditional_vae",
            "data_config": data_config,
            "objective": CVAE_OBJECTIVE,
        },
    )

    loader = make_evaluation_loader(device)
    metrics = evaluate_cvae(
        model,
        loader,
        device=device,
    )
    reset_dir(str(OUTPUT_DIR))
    save_conditional_samples(
        model,
        OUTPUT_DIR / "conditional_samples.png",
        device=device,
        samples_per_class=SAMPLES_PER_CLASS,
    )
    save_conditional_metric_summary(metrics, OUTPUT_DIR / "metric_summary.png")
    save_conditional_reconstructions(
        model,
        loader.dataset,
        OUTPUT_DIR / "real_reconstruction.png",
        device=device,
        samples_per_class=SAMPLES_PER_CLASS,
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "protocol": {
                    **data_config,
                    "evaluated_examples": len(loader.dataset),
                    "seed": SEED,
                    "objective": CVAE_OBJECTIVE,
                    "sample_rows": list(GLASSES_CLASS_NAMES),
                    "shared_base_noise_across_classes": True,
                    "held_out": False,
                },
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
