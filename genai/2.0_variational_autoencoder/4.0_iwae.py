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

PROJECT_ROOT = infer_project_root()


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
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
        train=True,
        download=True,
        transform=transform,
    )
    validation_set = datasets.MNIST(
        PROJECT_ROOT / "data" / "mnist",
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
    out_dir = PROJECT_ROOT / "output" / "vae" / "iwae" / f"k{particles}"
    reset_dir(str(out_dir))
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        ess_sum = 0.0
        examples = 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            log_weights, terms = importance_log_weights(model, images, particles=particles)
            loss = -log_mean_exp(log_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            diagnostics = importance_diagnostics(log_weights, terms)
            loss_sum += loss.item() * images.shape[0]
            ess_sum += float(diagnostics["ess_fraction"]) * images.shape[0]
            examples += images.shape[0]
        print(
            f"K={particles} epoch {epoch:03d}: "
            f"loss={loss_sum / examples:.3f}, ESS/K={ess_sum / examples:.3f}"
        )

    validation = evaluate_bound(
        model,
        validation_loader,
        particles=VALIDATION_PARTICLES,
        particle_chunk_size=PARTICLE_CHUNK_SIZE,
        max_examples=VALIDATION_EXAMPLES,
        device=device,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "iwae",
            "model_config": model_config(model),
        },
        out_dir / "model.pth",
    )
    model.eval()
    with torch.inference_mode():
        samples = model.sample(64, device=device)
    save_image(samples, out_dir / "prior_samples.png", nrow=8)
    print(
        f"K={particles}: validation bound={validation['bound']:.3f}, "
        f"ESS/K={validation['ess_fraction']:.3f}; saved {out_dir}"
    )


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
