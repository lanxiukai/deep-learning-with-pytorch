"""PixelCNN model, raster sampler, and shared VQ-VAE/FSQ prior training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.runtime.randomness import set_seed
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_tokenizers import VQVAE, FSQAutoencoder


class MaskedConv2d(nn.Conv2d):
    """PixelCNN mask: type A excludes the current token; B includes it."""

    mask: Tensor

    def __init__(self, mask_type: str, *args: Any, **kwargs: Any) -> None:
        if mask_type not in {"A", "B"}:
            raise ValueError("PixelCNN mask type must be A or B.")
        super().__init__(*args, **kwargs)
        # kernel shape: (C_out, C_in/groups, K_h, K_w)
        mask = torch.ones_like(self.weight)
        center_h = self.kernel_size[0] // 2
        center_w = self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        first_blocked = center_w if mask_type == "A" else center_w + 1
        mask[:, :, center_h, first_blocked:] = 0
        self.register_buffer("mask", mask)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNResidualBlock(nn.Module):
    """A plain 1x1 / masked 3x3 / 1x1 bottleneck residual block."""

    def __init__(self, channels: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(channels, bottleneck, 1)
        self.masked = MaskedConv2d("B", bottleneck, bottleneck, 3, padding=1)
        self.expand = nn.Conv2d(bottleneck, channels, 1)
        self.dropout = dropout

    def forward(self, hidden: Tensor) -> Tensor:
        update = self.reduce(F.relu(hidden))
        update = self.masked(F.relu(update))
        update = F.dropout(F.relu(update), self.dropout, self.training)
        return hidden + self.expand(update)


class PixelCNNPrior(nn.Module):
    """Plain bottleneck PixelCNN, following the residual layout of PixelRNN.

    The initial mask-A convolution can cover the whole token grid's past.
    Grouping this large kernel reduces its cost; the following 1x1 convolutions
    mix channels. Blocks contain only convolution, ReLU, dropout, and addition.
    Sampling recomputes the full grid in raster order.
    """

    def __init__(
        self,
        vocabulary_size: int,
        *,
        hidden_channels: int = 256,
        embedding_dim: int = 32,
        blocks: int = 15,
        bottleneck: int = 128,
        head_channels: int = 1024,
        first_kernel_size: int = 63,
        first_groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if first_kernel_size < 1 or first_kernel_size % 2 == 0:
            raise ValueError("PixelCNN kernel size must be positive and odd.")
        if blocks < 1:
            raise ValueError("Residual PixelCNN needs at least one block.")
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)  # (K, C)
        self.first = MaskedConv2d(
            "A",
            embedding_dim,
            hidden_channels,
            first_kernel_size,
            padding=first_kernel_size // 2,
            groups=first_groups,
        )
        self.blocks = nn.ModuleList(
            PixelCNNResidualBlock(hidden_channels, bottleneck, dropout)
            for _ in range(blocks)
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, head_channels, 1),
            nn.ReLU(),
            nn.Conv2d(head_channels, vocabulary_size, 1),
        )

    def forward(self, indices: Tensor) -> Tensor:
        # indices (B, h, w) -> hidden (B, hidden_channels, h, w)
        hidden = self.first(self.embedding(indices).permute(0, 3, 1, 2).contiguous())
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample in raster order with dropout disabled; restore the caller's mode."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive.")
        if min(count, height, width) < 1:
            raise ValueError("Sample count and grid dimensions must be positive.")
        was_training = self.training
        self.eval()
        try:
            indices = torch.zeros(count, height, width, dtype=torch.long, device=device)
            # Recompute the full grid, using only the current position's logits.
            for i in range(height):
                for j in range(width):
                    logits = self(indices)[:, :, i, j] / temperature
                    tokens = torch.multinomial(logits.softmax(dim=1), 1).squeeze(1)
                    indices[:, i, j] = tokens
            return indices
        finally:
            self.train(was_training)


def train_pixelcnn_prior_epoch(
    prior: PixelCNNPrior,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    progress: tqdm,
) -> float:
    """Fit the next-token distribution on frozen token grids."""
    prior.train()
    metrics = MetricAccumulator(("nll",), device=device)
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
    # Defer workflow imports so paired-model loading can import the prior.
    from dl_utils.vae import discrete_workflow

    epochs = recipe["prior_epochs"]
    seed = recipe["seed"]
    # Prior initialization must not depend on the number of tokenizer updates.
    set_seed(seed)
    prior = PixelCNNPrior(**recipe["model"]["prior"]).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=recipe["prior_lr"])
    # The prior's checkpoint also restores the frozen tokenizer it was trained on.
    checkpoint = discrete_workflow.epoch_checkpoint(
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
        discrete_workflow.save_loss_curves(
            state["history"], output_dir / "prior_loss.png"
        )
        return prior.eval()
    tokens = discrete_workflow.encode_dataset(
        tokenizer,
        images,
        device,
        horizontal_flip=recipe["prior_horizontal_flip"],
        token_batch_size=recipe["prior_batch_size"],
    )  # (B, h, w)
    side = recipe["model"]["image_size"] // (2**tokenizer.downsample_steps)
    with tqdm(
        total=epochs * len(tokens),
        initial=completed * len(tokens),
        desc="Stage 2/2: PixelCNN prior",
        unit="batch",
    ) as progress:
        for epoch in range(completed + 1, epochs + 1):
            progress.set_description(
                f"Stage 2/2: PixelCNN prior {epoch}/{epochs}", refresh=False
            )
            discrete_workflow.seed_epoch_loader(tokens, seed, epoch)
            nll = train_pixelcnn_prior_epoch(
                prior, tokens, optimizer, device, progress=progress
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
                training_dir = output_dir / "training"
                save_image(
                    samples.mul(0.5).add(0.5),
                    training_dir / f"prior_epoch_{epoch:03d}.png",
                    nrow=4,
                )
    discrete_workflow.save_loss_curves(state["history"], output_dir / "prior_loss.png")
    return prior.eval()


__all__ = [
    "MaskedConv2d",
    "PixelCNNPrior",
    "PixelCNNResidualBlock",
    "train_pixelcnn_prior",
    "train_pixelcnn_prior_epoch",
]
