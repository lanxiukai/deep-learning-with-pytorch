"""Train a 256x256 beta-VAE on the same setup as ``1.0_vae.py``.

The model, data, reconstruction term, optimizer, and training loop are shared
with the standard VAE lesson. Only beta changes the weight of the complete
per-image KL term.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py.

Outputs:
    output/vae/beta_vae/training/epoch_*.png: fixed-z prior samples
    output/vae/beta_vae/beta_vae.pth: final beta-VAE checkpoint
    output/vae/beta_vae/beta_vae_metrics.csv: epoch, total, reconstruction, and unweighted KL
    output/vae/beta_vae/loss_curves.png: total, reconstruction, and KL panels

Training data -- glasses-256:
With glasses (G):          2,543 images
Without glasses (NoG):     1,957 images
Available total:           4,500 images
Batch size:                   16
Samples per epoch:         4,496 (281 full batches; drop_last=True)
Training epochs:             100
Optimizer updates:         28,100
Default beta:                   4
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
    train_glasses_vae,
    weighted_vae_loss,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "beta_vae"
MODEL_CONFIG = {"z_dim": 100}
BETA = 4.0


def main():
    train_glasses_vae(
        loss_function=partial(weighted_vae_loss, beta=BETA),
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        checkpoint_name="beta_vae.pth",
        model_name="beta_vae",
        model_config=MODEL_CONFIG,
        metadata={"beta": BETA},
    )


if __name__ == "__main__":
    main()
