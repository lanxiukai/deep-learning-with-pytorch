"""Evaluate a frozen cVAE checkpoint without training any model.

Generate digits 0-9 from the conditional prior p(z | c), compare real test
images with posterior-mean reconstructions, and report reconstruction BCE
and KL(q(z | x, c) || p(z | c)). Both metrics sum over dimensions and average
per image; posterior-mean BCE is a deterministic reconstruction diagnostic.

Data:
    data/mnist, downloaded automatically by torchvision when absent.
    Only the MNIST test split is loaded.

Checkpoint:
    output/vae/conditional_vae/conditional_vae.pth: saved by 3.0_conditional_vae.py

Outputs:
    output/vae/conditional_vae/evaluation/metrics.json
    output/vae/conditional_vae/evaluation/conditional_samples.png: one row per digit
    output/vae/conditional_vae/evaluation/reconstructions.png: originals above reconstructions

Evaluation defaults:
    Test images: 5,000 of 10,000; batch size: 256.
    Generated images: 8 per digit; reconstruction comparisons: 8.
    Input and generated images: 32x32 grayscale.
    Model dimensions are loaded from the checkpoint.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.conditional_vae import ConditionalVAE
from dl_utils.vae.vae_common import diagonal_gaussian_kl_from_logvar

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "conditional_vae" / "conditional_vae.pth"
OUTPUT_DIR = CHECKPOINT.parent / "evaluation"

# Edit these defaults to explore the lesson.
IMAGE_SIZE = 32
BATCH_SIZE = 256
MAX_RECONSTRUCTION_EXAMPLES = 5_000
SAMPLES_PER_CLASS = 8
NUM_COMPARISON_IMAGES = 8
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
    subset = Subset(test_set, range(min(MAX_RECONSTRUCTION_EXAMPLES, len(test_set))))
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=WORKERS > 0,
    )


@torch.inference_mode()
def evaluate_reconstruction(
    model: ConditionalVAE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float | int], Tensor]:
    model.eval()
    reconstruction_total = 0.0
    kl_total = 0.0
    examples = 0
    comparison = None
    with tqdm(loader, desc="Evaluate CVAE", unit="batch", mininterval=0.5) as progress:
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            q_mu, q_logvar = model.encode(images, labels)
            p_mu, p_logvar = model.prior(labels)
            reconstruction = model.decode(q_mu, labels)
            reconstruction_total += float(
                F.binary_cross_entropy(reconstruction, images, reduction="sum")
            )
            kl_total += float(
                diagonal_gaussian_kl_from_logvar(q_mu, q_logvar, p_mu, p_logvar).sum()
            )
            examples += images.shape[0]
            if comparison is None:
                count = min(NUM_COMPARISON_IMAGES, images.shape[0])
                comparison = torch.cat((images[:count], reconstruction[:count])).cpu()

    if comparison is None:
        raise ValueError("cannot evaluate an empty loader")
    return {
        "examples": examples,
        "posterior_mean_reconstruction_bce_nats_per_image": reconstruction_total
        / examples,
        "kl_nats_per_image": kl_total / examples,
    }, comparison


@torch.inference_mode()
def evaluate() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {CHECKPOINT}; run 3.0_conditional_vae.py first"
        )
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = ConditionalVAE(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    metrics, comparison = evaluate_reconstruction(
        model, make_test_loader(device), device
    )
    labels = torch.arange(model.num_classes, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    samples = model.generate(labels)

    reset_dir(str(OUTPUT_DIR))
    save_image(samples, OUTPUT_DIR / "conditional_samples.png", nrow=SAMPLES_PER_CLASS)
    save_image(
        comparison, OUTPUT_DIR / "reconstructions.png", nrow=comparison.shape[0] // 2
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
