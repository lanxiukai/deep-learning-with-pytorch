"""Train a compact deterministic autoencoder on MNIST.

The lesson keeps one fixed, label-ordered reconstruction grid across epochs so
training progress is directly comparable. The final checkpoint contains model
weights and constructor metadata only; this short run intentionally has no
resume machinery.

Data:
    data/mnist, downloaded automatically by torchvision when absent.

Outputs:
    output/vae/autoencoder_mnist/training/epoch_*.png: fixed reconstructions
    output/vae/autoencoder_mnist/autoencoder.pth: final model checkpoint
    output/vae/autoencoder_mnist/loss_curves.png: per-image reconstruction loss

Training data -- MNIST:
Training images:          60,000
Test images:              10,000
Batch size:                   32
Samples per epoch:        60,000 (1,875 full batches)
Training epochs:              10
Optimizer updates:        18,750
Note: Digit labels select the fixed test grid but do not enter the training
objective.

Default dimensions:
Training input:           28x28 grayscale
Reconstructed image:      28x28 grayscale
Latent vector:                 20 values

Model size:
Encoder:                   0.161 M parameters
Decoder:                   0.162 M parameters
Total:                     0.323 M parameters
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import save_image
from tqdm import tqdm

from dl_utils.data.datasets.vision import vision_loaders
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.curves import save_loss_panels
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.runtime.timing import Timer
from dl_utils.training.checkpoints import save_model_weights
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.training.timing import format_epoch_timing

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "mnist"
OUT_DIR = PROJECT_ROOT / "output" / "vae" / "autoencoder_mnist"
TRAINING_DIR = OUT_DIR / "training"
CHECKPOINT_PATH = OUT_DIR / "autoencoder.pth"

NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 4
INPUT_DIMS = 28 * 28
Z_DIM = 20
HIDDEN_DIMS = 200
LEARNING_RATE = 2.5e-4
NUM_FIXED_DIGITS = 10
SAMPLE_EVERY_EPOCHS = 1
SEED = 42

MODEL_CONFIG = {
    "input_dims": INPUT_DIMS,
    "z_dim": Z_DIM,
    "hidden_dims": HIDDEN_DIMS,
}


class Autoencoder(nn.Module):
    """Fully connected MNIST autoencoder with one latent bottleneck."""

    def __init__(
        self,
        input_dims: int,
        z_dim: int,
        hidden_dims: int,
    ) -> None:
        super().__init__()
        self.encoder_hidden = nn.Linear(input_dims, hidden_dims)
        self.encoder_output = nn.Linear(hidden_dims, z_dim)
        self.decoder_hidden = nn.Linear(z_dim, hidden_dims)
        self.decoder_output = nn.Linear(hidden_dims, input_dims)

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder_output(F.relu(self.encoder_hidden(inputs)))

    def decode(self, z: Tensor) -> Tensor:
        hidden = F.relu(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_output(hidden))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(inputs)
        return self.decode(z), z


def collect_fixed_digits(loader) -> Tensor:
    """Return one test image for each digit, ordered from zero to nine."""
    examples: dict[int, Tensor] = {}
    for images, labels in loader:
        for image, label in zip(images, labels, strict=True):
            examples.setdefault(int(label), image)
        if len(examples) == NUM_FIXED_DIGITS:
            break
    if len(examples) != NUM_FIXED_DIGITS:
        raise ValueError("MNIST test data does not contain every digit class.")
    return torch.stack([examples[label] for label in range(NUM_FIXED_DIGITS)])


@torch.inference_mode()
def save_reconstruction_grid(
    model: Autoencoder,
    originals: Tensor,
    device: torch.device,
    output_path: Path,
) -> None:
    """Save fixed originals above their deterministic reconstructions."""
    was_training = model.training
    model.eval()
    flattened = originals.to(device).flatten(1)
    reconstructions, _ = model(flattened)
    reconstructions = reconstructions.reshape_as(originals).cpu()
    save_image(
        torch.cat((originals, reconstructions)),
        output_path,
        nrow=NUM_FIXED_DIGITS,
    )
    model.train(was_training)


def train_epoch(
    model: Autoencoder,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress_bar: tqdm,
) -> float:
    """Train one epoch and return summed-pixel MSE per image."""
    model.train()
    metrics = MetricAccumulator(("reconstruction",), device=device)
    for images, _ in loader:
        images = images.to(device, non_blocking=True).flatten(1)
        reconstructions, _ = model(images)
        reconstruction_loss = (
            (reconstructions - images).square().flatten(1).sum(dim=1).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        reconstruction_loss.backward()
        optimizer.step()
        metrics.add_batch_means((reconstruction_loss,), num_examples=images.shape[0])
        progress_bar.update(1)
    return metrics.compute_weighted_means(require_finite=True)["reconstruction"]


def main() -> None:
    if not OUT_DIR.exists():
        reset_dir(str(OUT_DIR))
    reset_dir(str(TRAINING_DIR))
    set_seed(SEED)
    device = try_gpu()
    train_loader, test_loader = vision_loaders(
        "mnist",
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    fixed_digits = collect_fixed_digits(test_loader)

    model = Autoencoder(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epochs: list[int] = []
    reconstruction_history: list[float] = []

    save_reconstruction_grid(
        model,
        fixed_digits,
        device,
        TRAINING_DIR / "epoch_000.png",
    )

    timer = Timer()
    with tqdm(
        total=NUM_EPOCHS * len(train_loader),
        desc=f"Epoch 1/{NUM_EPOCHS}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, NUM_EPOCHS + 1):
            progress_bar.set_description(f"Epoch {epoch}/{NUM_EPOCHS}", refresh=False)
            reconstruction_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                progress_bar,
            )
            epochs.append(epoch)
            reconstruction_history.append(reconstruction_loss)
            progress_bar.set_postfix(
                reconstruction=f"{reconstruction_loss:.4f}", refresh=False
            )

            if epoch % SAMPLE_EVERY_EPOCHS == 0:
                save_reconstruction_grid(
                    model,
                    fixed_digits,
                    device,
                    TRAINING_DIR / f"epoch_{epoch:03d}.png",
                )

    save_model_weights(
        model,
        CHECKPOINT_PATH,
        metadata={
            "model_name": "mnist_autoencoder",
            "model_config": MODEL_CONFIG,
        },
    )
    save_loss_panels(
        epochs,
        {
            "Reconstruction objective": {
                "Summed-pixel MSE per image": reconstruction_history,
            },
        },
        OUT_DIR / "loss_curves.png",
    )
    print(format_epoch_timing(timer.stop(), NUM_EPOCHS))


if __name__ == "__main__":
    main()
