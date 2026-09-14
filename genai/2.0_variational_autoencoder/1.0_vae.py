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

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.vae.vae import (
    reconstruction_and_kl,
    train_glasses_vae,
)

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "vae"
MODEL_CONFIG = {"z_dim": 100}


def vae_loss(images, reconstructions, mu, logvar):
    """Return the standard VAE objective: reconstruction + KL."""
    reconstruction_loss, kl_loss = reconstruction_and_kl(
        images, reconstructions, mu, logvar
    )
    return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss


def main():
    train_glasses_vae(
        loss_function=vae_loss,
        data_dir=DATA_DIR,
        out_dir=OUT_DIR,
        checkpoint_name="vae.pth",
        model_name="vae",
        model_config=MODEL_CONFIG,
    )


if __name__ == "__main__":
    main()
