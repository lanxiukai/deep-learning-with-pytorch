"""Train the introductory 256x256 VAE on ``glasses-256`` images.

The standard Gaussian-prior objective combines summed-pixel MSE with the
complete per-image KL. A fixed prior batch is decoded throughout training so
sample changes reflect model updates rather than newly drawn latent vectors.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py.

Outputs:
    output/vae/vae/training/epoch_*.png: fixed-z prior samples
    output/vae/vae/vae.pth: final model checkpoint
    output/vae/vae/vae_metrics.csv: epoch, total, reconstruction, and unweighted KL
    output/vae/vae/loss_curves.png: total, reconstruction, and KL panels

Training data -- glasses-256:
With glasses (G):          2,543 images
Without glasses (NoG):     1,957 images
Available total:           4,500 images
Batch size:                   16
Samples per epoch:         4,496 (281 full batches; drop_last=True)
Training epochs:             100
Optimizer updates:         28,100
Note: Class labels are ignored. Counts include 517 repository-tracked label
corrections; four shuffled images are omitted per epoch.

Default dimensions:
Training input:           256x256 RGB
Generated image:          256x256 RGB
Latent vector:                100 values

Model size:
Encoder:                  31.70 M parameters
Decoder:                  31.63 M parameters
Total:                    63.33 M parameters
"""

from functools import partial

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae import (
    train_vae,
    weighted_vae_loss,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "vae"
MODEL_CONFIG = {"z_dim": 100}

# Training configuration
EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
NUM_FIXED_SAMPLES = 18
SAMPLE_GRID_COLUMNS = 6
SAMPLE_EVERY_EPOCHS = 10
SEED = 42


def main():
    train_vae(
        loss_function=partial(weighted_vae_loss, beta=1.0),
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        checkpoint_name="vae.pth",
        model_name="vae",
        model_config=MODEL_CONFIG,
        metadata={"beta": 1.0},
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        num_fixed_samples=NUM_FIXED_SAMPLES,
        sample_grid_columns=SAMPLE_GRID_COLUMNS,
        sample_every_epochs=SAMPLE_EVERY_EPOCHS,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
