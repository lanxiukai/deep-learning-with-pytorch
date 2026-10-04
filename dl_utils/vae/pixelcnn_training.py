"""Shared PixelCNN prior training for the VQ-VAE and FSQ lessons."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_workflow import (
    encode_dataset,
    epoch_checkpoint,
    save_loss_curves,
    seed_epoch_loader,
)
from dl_utils.vae.quantization import VQVAE, FSQAutoencoder
from dl_utils.vae.token_priors import PixelCNNPrior


def train_pixelcnn_prior_epoch(
    prior: PixelCNNPrior,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    desc: str = "PixelCNN",
) -> float:
    """Fit the next-token distribution on frozen token grids."""
    prior.train()
    metrics = MetricAccumulator(("nll",), device=device)
    with tqdm(total=len(loader), desc=desc) as progress:
        for indices, _ in loader:
            indices = indices.to(device)
            loss = F.cross_entropy(prior(indices), indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            metrics.add_batch_means((loss,), num_examples=len(indices))
            progress.update(1)
        nll = metrics.compute_weighted_means(require_finite=True)["nll"]
        progress.set_postfix(nll=f"{nll:.4f}", refresh=False)
    return nll


def train_pixelcnn_prior(
    tokenizer: VQVAE | FSQAutoencoder,
    images: DataLoader,
    device: torch.device,
    recipe: Mapping[str, Any],
    output_dir: Path,
    *,
    resume: bool = True,
    sample_every: int = 10,
    num_samples: int = 8,
    temperature: float = 1.0,
) -> PixelCNNPrior:
    """Train the same unconditional prior for either frozen tokenizer."""
    epochs = recipe["prior_epochs"]
    seed = recipe["seed"]
    prior = PixelCNNPrior(**recipe["model"]["prior"]).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=recipe["prior_lr"])
    # The prior's checkpoint also restores the frozen tokenizer it was trained on.
    checkpoint = epoch_checkpoint(
        output_dir / "prior_latest.pth",
        {"tokenizer": tokenizer, "prior": prior},
        {"prior": optimizer},
        recipe,
    )
    completed, state = checkpoint.resume(
        checkpoint.path if resume and checkpoint.path.is_file() else None,
        initial_state={"history": []},
    )
    tokenizer.eval().requires_grad_(False)
    if completed == epochs:
        save_loss_curves(state["history"], output_dir / "prior_loss.png")
        return prior.eval()
    tokens = encode_dataset(tokenizer, images, device)
    side = recipe["model"]["image_size"] // (2**tokenizer.downsample_steps)
    for epoch in range(completed + 1, epochs + 1):
        seed_epoch_loader(tokens, seed, epoch)
        nll = train_pixelcnn_prior_epoch(
            prior, tokens, optimizer, device, desc=f"PixelCNN {epoch}/{epochs}"
        )
        state["history"].append({"nll": nll})
        checkpoint.save(epoch, state)
        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            with torch.random.fork_rng(), torch.inference_mode():
                torch.manual_seed(seed)
                prior.eval()
                indices = prior.sample(
                    num_samples, side, side, device=device, temperature=temperature
                )
                samples = tokenizer.decode_indices(indices)
            save_image(
                samples.mul(0.5).add(0.5),
                output_dir / "training" / f"prior_epoch_{epoch:03d}.png",
                nrow=4,
            )
    save_loss_curves(state["history"], output_dir / "prior_loss.png")
    return prior.eval()
