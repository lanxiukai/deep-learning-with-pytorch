"""Evaluate a frozen cVAE checkpoint without training any model.

Generate digits 0-9 from the conditional prior p(z | c) and evaluate the
same sampled conditional-ELBO metrics used during training over held-out
images.

Data:
    data/mnist, downloaded automatically by torchvision when absent.
    Only the MNIST test split is loaded.

Checkpoint:
    output/vae/conditional_vae/conditional_vae.pth: saved by 3.0_conditional_vae.py

Outputs:
    output/vae/conditional_vae/evaluation/metrics.json
    output/vae/conditional_vae/evaluation/conditional_samples.png: one row per digit

Evaluation defaults:
    Test images: 5,000 of 10,000; batch size: 256.
    Generated images: 8 per digit.
    Input and generated images: 32x32 grayscale.
    Model dimensions are loaded from the checkpoint.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.conditional_vae import (
    ConditionalVAE,
    evaluate_cvae,
    save_conditional_samples,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "conditional_vae" / "conditional_vae.pth"
OUTPUT_DIR = CHECKPOINT.parent / "evaluation"

# Edit these defaults to explore the lesson.
IMAGE_SIZE = 32
BATCH_SIZE = 256
MAX_EVALUATION_EXAMPLES = 5_000
SAMPLES_PER_CLASS = 8
ACTIVE_RATE_THRESHOLD = 0.05
WORKERS = 4
SEED = 123


def make_test_loader(device: torch.device) -> DataLoader:
    test_set = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        ),
    )
    subset = Subset(test_set, range(min(MAX_EVALUATION_EXAMPLES, len(test_set))))
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=WORKERS > 0,
    )


@torch.inference_mode()
def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {CHECKPOINT}; run 3.0_conditional_vae.py first"
        )
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = ConditionalVAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics = evaluate_cvae(
        model,
        make_test_loader(device),
        device=device,
        active_rate_threshold=ACTIVE_RATE_THRESHOLD,
    )
    reset_dir(str(OUTPUT_DIR))
    save_conditional_samples(
        model,
        OUTPUT_DIR / "conditional_samples.png",
        device=device,
        samples_per_class=SAMPLES_PER_CLASS,
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved evaluation to {OUTPUT_DIR}")


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
