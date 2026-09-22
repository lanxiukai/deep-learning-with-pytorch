"""Two-level HVAE baseline with matched top-down prior/posterior conditions.

The generator is

    p(z2) p(z1 | z2) p(x | z1),

and the recognition model is

    q(z2 | x) q(z1 | z2, x).

Reading guide 3.3c, section 3 (layer 0 = x, layer 1 = z1, layer 2 = z2):
    q_2(x)           -> mu_q_2, v_q_2 -> sample z2       (step 1)
    p_1(z2)          -> mu_p_1, v_p_1                    (step 2)
    q_1([h_1(x),z2]) -> mu_q_1, v_q_1 -> sample z1       (step 2)
    p_0(z1)          -> mu_p_0                          (step 3)
Here v = log(sigma**2). q_2 contains the full image encoder and also returns
the deterministic feature h_1(x), shared with q_1.
model(x) returns (mu_p_0, latents), with the named Gaussian parameters and
z1/z2 in latents. p(z2) is fixed N(0, I); observation variance is fixed 1/2.
See dl_utils/vae/hierarchical_vae.py for the numbered blocks and forward path.

The negative ELBO therefore contains a top KL against N(0, I) and a lower KL
between distributions conditioned on the same sampled z2.  KL warm-up and
group-wise free bits are visible optimization controls, not guarantees that a
layer is informative.  The next lesson changes only the lower posterior.

The script saves final model weights and the small set of constructor and
evaluation controls needed by the next lesson. It has no resume machinery.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the RGB cache directly, without resizing or normalization.
    Use all 4,500 images, matching VAE/CVAE; class labels are ignored.

Outputs:
    output/vae/hierarchical_vae/baseline/hierarchical_vae.pth: final checkpoint
    output/vae/hierarchical_vae/baseline/prior_samples.png: fixed-noise sample grid
    output/vae/hierarchical_vae/baseline/training/epoch_*.png: selected epochs
    output/vae/hierarchical_vae/baseline/hierarchical_vae_metrics.csv: epoch metrics
    output/vae/hierarchical_vae/baseline/hierarchical_vae_metrics_*.png: metric curves

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
Model size:              8.862 M parameters
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
from dl_utils.vae.hierarchical_vae import HierarchicalVAE

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vae" / "hierarchical_vae" / "baseline"
MODEL_NAME = "hierarchical_vae"
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
    model = HierarchicalVAE(
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
