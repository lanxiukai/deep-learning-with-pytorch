"""Importance-Weighted Autoencoder: tighten the bound without changing networks.

For each image, K samples from the same diagonal-Gaussian posterior form

    log_mean_exp(log p(x | z_k) + log p(z_k) - log q(z_k | x)).

The particle reduction happens per image and entirely in log space.  K=1 is
exactly the Monte Carlo ELBO.  Larger K changes the objective and compute
budget, not the VAE model family.

All default particle counts use the same architecture and number of training
epochs.
Final weights feed the companion evaluation script. Evaluation decodes
particles in small chunks to keep memory use bounded.

Data:
    data/mnist, downloaded automatically by torchvision when absent.

Outputs:
    output/vae/iwae/k1/: default K=1 checkpoint and prior samples
    output/vae/iwae/k4/: default K=4 checkpoint and prior samples
    output/vae/iwae/k8/: default K=8 checkpoint and prior samples
    output/vae/iwae/k16/: default K=16 checkpoint and prior samples
    output/vae/iwae/k32/: default K=32 checkpoint and prior samples
    output/vae/iwae/k*/iwae_metrics.csv: epoch loss, reconstruction loss,
        KL loss, and ESS/K
    output/vae/iwae/k*/iwae_metrics_01.png: curves for the same metrics

Training data -- MNIST:
Training images:          60,000
Batch size:                  128
Samples per full pass:    59,904 (468 full batches; drop_last=True)
Training epochs:              10 per K
Default runs:             K=1, K=4, K=8, K=16, and K=32
Optimizer updates:         4,680 per run / 23,400 total
Note: Digit labels are ignored.

Default dimensions:
Training input:           32x32 grayscale
Generated image:          32x32 grayscale
Latent vector:                 16 values
Context hidden layer:        128 units before the mean/log-variance heads

Model size (shared by every K):
Encoder/posterior:         0.431 M parameters
Decoder:                   0.199 M parameters
Total:                     0.631 M parameters
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.iwae import (
    GaussianVAE,
    importance_statistics,
)
from dl_utils.vae.training_artifacts import save_training_metrics

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae" / "iwae"
CHECKPOINT_NAME = "iwae.pth"
IMAGE_SIZE = 32
SAMPLE_COUNT = 64
SAMPLE_GRID_COLUMNS = 8
SAMPLE_EVERY = 5
PROGRESS_INTERVAL = 0.5
MAX_METRIC_PANELS = 4


# Edit these defaults to explore the lesson.
PARTICLES = (1, 4, 8, 16, 32)
EPOCHS = 10
BATCH_SIZE = 128
LATENT_DIM = 16
HIDDEN_CHANNELS = 128
CONTEXT_DIM = 128  # Posterior MLP width; independent of image channels.
LR = 2e-4
WORKERS = 4
SEED = 42


def make_train_loader(device: torch.device) -> DataLoader:
    transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    train_set = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": WORKERS > 0,
    }
    return DataLoader(train_set, shuffle=True, drop_last=True, **common)


def train_iwae_for_particles(
    *,
    particles: int,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    set_seed(SEED)
    model_config = {
        "latent_dim": LATENT_DIM,
        "hidden_channels": HIDDEN_CHANNELS,
        "context_dim": CONTEXT_DIM,
    }
    model = GaussianVAE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    out_dir = OUTPUT_ROOT / f"k{particles}"
    if not out_dir.exists():
        reset_dir(str(out_dir))
    training_dir = out_dir / "training"
    reset_dir(str(training_dir))
    history = []
    with tqdm(
        total=EPOCHS * len(train_loader),
        desc=f"IWAE K={particles} 1/{EPOCHS}",
        unit="batch",
        mininterval=PROGRESS_INTERVAL,
    ) as progress:
        for epoch in range(1, EPOCHS + 1):
            progress.set_description(f"IWAE K={particles} {epoch}/{EPOCHS}", refresh=False)
            model.train()
            metrics_accumulator = MetricAccumulator(
                ("loss", "reconstruction_loss", "kl_loss", "ess_fraction"),
                device=device,
            )
            for images, _ in train_loader:
                images = images.to(device, non_blocking=True)
                metrics = importance_statistics(
                    model, images, particles=particles
                )
                loss = metrics["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metrics_accumulator.update(
                    (
                        loss,
                        metrics["reconstruction_loss"],
                        metrics["kl_loss"],
                        metrics["ess_fraction"],
                    ),
                    num_examples=images.shape[0],
                )
                running_metrics = metrics_accumulator.compute()
                progress.set_postfix(
                    loss=f"{running_metrics['loss']:.3f}",
                    ess=f"{running_metrics['ess_fraction']:.3f}",
                    refresh=False,
                )
                progress.update(1)
            history.append(metrics_accumulator.compute())
            if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
                model.eval()
                with torch.inference_mode():
                    samples = model.sample(SAMPLE_COUNT, device=device)
                save_image(
                    samples,
                    training_dir / f"epoch_{epoch:03d}.png",
                    nrow=SAMPLE_GRID_COLUMNS,
                )

    save_training_metrics(
        history, out_dir, prefix="iwae", max_panels=MAX_METRIC_PANELS
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "iwae",
            "model_config": model_config,
        },
        out_dir / CHECKPOINT_NAME,
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(SAMPLE_COUNT, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=SAMPLE_GRID_COLUMNS)


def main() -> None:
    device = try_gpu()
    train_loader = make_train_loader(device)
    for particles in PARTICLES:
        train_iwae_for_particles(
            particles=particles,
            train_loader=train_loader,
            device=device,
        )


if __name__ == "__main__":
    main()
