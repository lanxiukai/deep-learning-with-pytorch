"""Ladder VAE: replace one lower posterior network with Gaussian fusion.

The generator, two conditional KL terms, decoder, latent sizes, warm-up, and
free-bits protocol match `5.0_hierarchical_vae.py`.  Only q(z1 | z2, x)
changes: the matching top-down prior and bottom-up evidence are multiplied,
so their precisions add and their means combine by precision weighting.

The top q(z2 | x) remains bottom-up evidence compared against N(0, I); it is
not fused with that fixed prior as though a third evidence source existed.

The script saves final model weights and the small set of constructor and
evaluation controls needed for comparison. It has no resume machinery.

Data:
    FactorShapes32, generated in memory with the same deterministic split as
    ``5.0_hierarchical_vae.py``.

Outputs:
    output/vae/ladder_vae/baseline/ladder_vae.pth: final Ladder VAE checkpoint
    output/vae/ladder_vae/baseline/prior_samples.png: prior sample grid

Training data -- FactorShapes32:
Available combinations:       3,528
Training split:                2,822 images
Test split:                      706 images
Batch size:                       128
Samples per epoch:             2,816 (22 full batches; drop_last=True)
Training epochs:                  40
Optimizer updates:               880
Note: Factor annotations define the dataset and later evaluation but do not
enter the training objective. Six shuffled images are omitted per epoch.

Default dimensions:
Training input:               32x32 grayscale
Generated image:              32x32 grayscale
Latent vectors:               z1=24 / z2=12 values

Model size:
Ladder VAE:                    0.874 M parameters
"""

from __future__ import annotations

import torch

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.hierarchy_training import (
    make_factor_shape_loaders,
    train_hierarchy,
)
from dl_utils.vae.vae_hierarchy import (
    LadderVAE32,
)

PROJECT_ROOT = infer_project_root()
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "ladder_vae" / "baseline"
MODEL_NAME = "ladder_vae"
SAMPLE_COUNT = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY = 5
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4


# Edit these defaults to explore the lesson.
EPOCHS = 40
BATCH_SIZE = 128
Z1_DIM = 24
Z2_DIM = 12
HIDDEN_CHANNELS = 64
CONTEXT_DIM = 192
LR = 2e-4
WARMUP_EPOCHS = 10.0
FREE_BITS = 1.0
ACTIVE_VARIANCE_THRESHOLD = 1e-2
WORKERS = 4
SPLIT_SEED = 2026
SEED = 42


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_factor_shape_loaders(
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        split_seed=SPLIT_SEED,
        device=device,
    )
    model = LadderVAE32(
        z1_dim=Z1_DIM,
        z2_dim=Z2_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        context_dim=CONTEXT_DIM,
    ).to(device)
    train_hierarchy(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LR,
        warmup_epochs=WARMUP_EPOCHS,
        free_bits=FREE_BITS,
        active_variance_threshold=ACTIVE_VARIANCE_THRESHOLD,
        out_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        sample_count=SAMPLE_COUNT,
        sample_grid_columns=SAMPLE_GRID_COLUMNS,
        sample_every=SAMPLE_EVERY,
        progress_interval=PROGRESS_INTERVAL,
        max_metric_panels=MAX_METRIC_PANELS,
        split_seed=SPLIT_SEED,
    )


if __name__ == "__main__":
    main()
