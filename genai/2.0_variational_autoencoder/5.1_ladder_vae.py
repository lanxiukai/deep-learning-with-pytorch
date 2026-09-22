"""Ladder VAE: replace one lower posterior network with Gaussian fusion.

The generator, two conditional KL terms, decoder, latent sizes, warm-up, and
free-bits protocol match `5.0_hierarchical_vae.py`.  Only q(z1 | z2, x)
changes: the matching top-down prior and bottom-up evidence are multiplied,
so their precisions add and their means combine by precision weighting.

The top q(z2 | x) remains bottom-up evidence compared against N(0, I); it is
not fused with that fixed prior as though a third evidence source existed.

Reading guide 3.3c, section 4 (layer 0 = x, layer 1 = z1, layer 2 = z2):
    q_2(x)         -> mu_q_2, v_q_2 -> sample z2
    p_1(z2)        -> mu_p_1, v_p_1
    q_hat_1(h_1(x))-> mu_hat_q_1, v_hat_q_1              (hatted evidence)
    precision fusion -> mu_q_1, v_q_1 -> sample z1       (superscript L)
    p_0(z1)        -> mu_p_0
Here v = log(sigma**2). q_2 includes the full image encoder and also returns
h_1(x) for q_hat_1. The hatted evidence is not sampled; the fused q,1
parameters are computed without a learned q_1 block. model(x) returns
(mu_p_0, latents), retaining both evidence and fused parameters in latents.
See dl_utils/vae/hierarchical_vae.py, especially LadderVAE.lower_parameters.

The script saves final model weights and the small set of constructor and
evaluation controls needed for comparison. It has no resume machinery.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the RGB cache directly, without resizing or normalization.
    Use all 4,500 images, matching VAE/CVAE; class labels are ignored.

Outputs:
    output/vae/ladder_vae/baseline/ladder_vae.pth: final checkpoint
    output/vae/ladder_vae/baseline/prior_samples.png: fixed-noise sample grid
    output/vae/ladder_vae/baseline/training/epoch_*.png: selected epochs
    output/vae/ladder_vae/baseline/ladder_vae_metrics.csv: epoch metrics
    output/vae/ladder_vae/baseline/ladder_vae_metrics_*.png: metric curves

Training data -- glasses-256:
Training images:          4,500
Batch size:                  16
Samples per epoch:        4,496 (281 full batches; drop_last=True)
Training epochs:              80
Optimizer updates:        22,480
Four shuffled images are omitted per epoch.

Default dimensions:
Training/generated image: 256x256 RGB in [0, 1]
Latent vectors:           z1=96 / z2=32 values
Encoder channels:        32, 64, 128, 256, 256, 256 (shared CVAE backbone)
Context hidden layers:   512 units
Model size:              8.846 M parameters
Objective:               summed RGB MSE + two conditional KL terms
Optimizer:               Adam, betas (0.9, 0.999)
Learning rate:           2e-4, cosine decay toward 2e-5 over 80 epochs

Run without arguments; edit the constants below to experiment. Comparison
with VAE/CVAE shares data and pixel loss, but capacity and training differ.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dl_utils.data.glasses import glasses_dataset
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.hierarchical_training import train_hierarchy
from dl_utils.vae.hierarchical_vae import LadderVAE

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "ladder_vae" / "baseline"
MODEL_NAME = "ladder_vae"
SAMPLE_COUNT = 16
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY = 5
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4


# Edit these defaults to explore the lesson.
EPOCHS = 80
BATCH_SIZE = 16
Z1_DIM = 96
Z2_DIM = 32
HIDDEN_CHANNELS = 256
CONTEXT_DIM = 512
LR = 2e-4
MIN_LR = 2e-5
WARMUP_EPOCHS = 10.0
FREE_BITS = 1.0
WORKERS = 4
SEED = 42


def make_train_loader(device: torch.device) -> DataLoader:
    return make_device_aware_loader(
        glasses_dataset(DATA_DIR),
        BATCH_SIZE,
        device,
        shuffle=True,
        drop_last=True,
        num_workers=WORKERS,
    )


def main() -> None:
    set_seed(SEED)
    device = try_gpu()
    train_loader = make_train_loader(device)
    model = LadderVAE(
        z1_dim=Z1_DIM,
        z2_dim=Z2_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        context_dim=CONTEXT_DIM,
    ).to(device)
    train_hierarchy(
        model,
        train_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LR,
        minimum_learning_rate=MIN_LR,
        warmup_epochs=WARMUP_EPOCHS,
        free_bits=FREE_BITS,
        out_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        sample_count=SAMPLE_COUNT,
        sample_grid_columns=SAMPLE_GRID_COLUMNS,
        sample_every=SAMPLE_EVERY,
        progress_interval=PROGRESS_INTERVAL,
        max_metric_panels=MAX_METRIC_PANELS,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
