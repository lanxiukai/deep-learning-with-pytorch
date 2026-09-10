"""Importance-Weighted Autoencoder: tighten the bound without changing networks.

For each image, K samples from the same diagonal-Gaussian posterior form

    log_mean_exp(log p(x | z_k) + log p(z_k) - log q(z_k | x)).

The particle reduction happens per image and entirely in log space.  K=1 is
exactly the Monte Carlo ELBO.  Larger K changes the objective and compute
budget, not the VAE model family.

Both K=1 and K=5 use the same architecture and number of training epochs.
Final weights feed the companion evaluation script. Evaluation decodes
particles in small chunks to keep memory use bounded.

Data:
    data/mnist, downloaded automatically by torchvision when absent.

Outputs:
    output/vae/iwae/k1/: default K=1 checkpoint and prior samples
    output/vae/iwae/k5/: default K=5 checkpoint and prior samples

Training data -- MNIST:
Training images:          60,000
Validation images:        10,000
Batch size:                  128
Samples per full pass:    59,904 (468 full batches; drop_last=True)
Training epochs:              10 per K
Default runs:             K=1 and K=5
Optimizer updates:         4,680 per run / 9,360 total
Note: Digit labels are ignored.

Default dimensions:
Training input:           32x32 grayscale
Generated image:          32x32 grayscale
Latent vector:                 16 values

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
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.inference import (
    GaussianVAE32,
    importance_diagnostics,
    importance_log_weights,
    log_mean_exp,
    model_config,
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
PARTICLES = (1, 5)
EPOCHS = 10
BATCH_SIZE = 128
LATENT_DIM = 16
HIDDEN_CHANNELS = 128
CONTEXT_DIM = 128
LR = 2e-4
VALIDATION_PARTICLES = 64
PARTICLE_CHUNK_SIZE = 8
VALIDATION_EXAMPLES = 2_048
WORKERS = 4
SEED = 42


def make_loaders(device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    )
    train_set = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    validation_set = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": WORKERS > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **common),
        DataLoader(validation_set, shuffle=False, drop_last=False, **common),
    )


@torch.inference_mode()
def evaluate_bound(
    model: GaussianVAE32,
    loader: DataLoader,
    *,
    particles: int,
    particle_chunk_size: int,
    max_examples: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for images, _ in loader:
        remaining = max_examples - examples
        if remaining <= 0:
            break
        images = images[:remaining].to(device, non_blocking=True)
        log_weights, terms = importance_log_weights(
            model,
            images,
            particles=particles,
            particle_chunk_size=particle_chunk_size,
        )
        diagnostics = importance_diagnostics(log_weights, terms)
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * images.shape[0]
        examples += images.shape[0]
    return {name: value / examples for name, value in totals.items()}


def train_one(
    *,
    particles: int,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> None:
    set_seed(SEED)
    model = GaussianVAE32(
        latent_dim=LATENT_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        context_dim=CONTEXT_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    out_dir = OUTPUT_ROOT / f"k{particles}"
    reset_dir(str(out_dir))
    training_dir = out_dir / "training"
    training_dir.mkdir()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        ess_sum = 0.0
        examples = 0
        progress = tqdm(
            train_loader,
            desc=f"IWAE K={particles} {epoch}/{EPOCHS}",
            mininterval=PROGRESS_INTERVAL,
        )
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            log_weights, terms = importance_log_weights(
                model, images, particles=particles
            )
            loss = -log_mean_exp(log_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            diagnostics = importance_diagnostics(log_weights, terms)
            loss_sum += loss.item() * images.shape[0]
            ess_sum += float(diagnostics["ess_fraction"]) * images.shape[0]
            examples += images.shape[0]
            progress.set_postfix(
                loss=f"{loss_sum / examples:.3f}",
                ess=f"{ess_sum / examples:.3f}",
                refresh=False,
            )
        history.append(
            {"loss": loss_sum / examples, "ess_fraction": ess_sum / examples}
        )
        if epoch == 1 or epoch % SAMPLE_EVERY == 0 or epoch == EPOCHS:
            model.eval()
            with torch.inference_mode():
                samples = model.sample(SAMPLE_COUNT, device=device)
            save_image(
                samples,
                training_dir / f"epoch_{epoch:03d}.png",
                nrow=SAMPLE_GRID_COLUMNS,
            )

    validation = evaluate_bound(
        model,
        validation_loader,
        particles=VALIDATION_PARTICLES,
        particle_chunk_size=PARTICLE_CHUNK_SIZE,
        max_examples=VALIDATION_EXAMPLES,
        device=device,
    )
    save_training_metrics(history, out_dir, prefix="iwae", max_panels=MAX_METRIC_PANELS)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "iwae",
            "model_config": model_config(model),
            "validation": validation,
        },
        out_dir / CHECKPOINT_NAME,
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(SAMPLE_COUNT, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=SAMPLE_GRID_COLUMNS)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = make_loaders(device)
    for particles in PARTICLES:
        train_one(
            particles=particles,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
        )


if __name__ == "__main__":
    main()
