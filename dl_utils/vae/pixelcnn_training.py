"""Frozen-token PixelCNN training, evaluation, and sampling for VQ-VAE and FSQ.

Shared data, token caches, monitoring artifacts, and epoch recovery live in
discrete_workflow. Read the epoch, evaluation, and sampling helpers before
the complete train_pixelcnn_prior workflow.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.training.artifacts import save_training_metrics
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.discrete_workflow import (
    TokenizerStage,
    image_contract,
    prepare_token_loaders,
    seed_epoch_loader,
)
from dl_utils.vae.quantization import VQVAE, FSQAutoencoder
from dl_utils.vae.token_priors import PixelCNNPrior


def train_pixelcnn_prior_epoch(
    prior: PixelCNNPrior,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    *,
    progress: tqdm,
    log_every: int = 100,
) -> float:
    """Train one causal-prior epoch over cached frozen-token grids."""
    prior.train()
    metrics = MetricAccumulator(("nll",), device=device)
    for batch_index, (indices, _) in enumerate(loader, 1):
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        loss = F.cross_entropy(prior(indices), indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.add_batch_means((loss,), num_examples=indices.shape[0])
        if batch_index % log_every == 0:
            nll = metrics.compute_weighted_means(require_finite=True)["nll"]
            progress.set_postfix(
                nll=f"{nll:.4f}", bpt=f"{nll / math.log(2):.3f}", refresh=False
            )
        progress.update(1)
    return metrics.compute_weighted_means(require_finite=True)["nll"]


@torch.inference_mode()
def evaluate_pixelcnn_prior(
    prior: PixelCNNPrior,
    loader: Iterable[tuple[Tensor, Tensor]],
    *,
    tokens_per_image: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure PixelCNN NLL for one frozen tokenizer."""
    prior.eval()
    metrics = MetricAccumulator(("nll",), device=device)
    for indices, _ in loader:
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        loss = F.cross_entropy(prior(indices), indices)
        metrics.add_batch_means((loss,), num_examples=indices.shape[0])
    nll = metrics.compute_weighted_means(require_finite=True)["nll"]
    return {
        "nll_nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "bits_per_image": tokens_per_image * nll / math.log(2),
    }


@torch.inference_mode()
def sample_pixelcnn_prior_images(
    tokenizer: VQVAE | FSQAutoencoder,
    prior: PixelCNNPrior,
    count: int,
    *,
    grid_size: int,
    device: torch.device,
    temperature: float,
) -> Tensor:
    """Sample a square token grid and decode it to an image batch."""
    indices = prior.sample(
        count,
        grid_size,
        grid_size,
        device=device,
        temperature=temperature,
    )
    return tokenizer.decode_indices(indices)


def train_pixelcnn_prior(
    tokenizer,
    tokenizer_payload,
    train_loader,
    monitor_loader,
    device,
    out_dir,
    monitor_protocol,
    *,
    model_name,
    data_dir,
    image_size,
    hidden_channels,
    layers,
    lr,
    epochs,
    resume,
    seed,
    sample_every,
    log_every,
    sample_count,
    sample_columns,
    temperature,
    checkpoint_name,
    progress_interval,
    max_metric_panels,
):
    grid_size = image_size // (2**tokenizer.downsample_steps)
    tokenizer.eval().requires_grad_(False)
    tokenizer_id = tokenizer_payload["snapshot_id"]
    tokens, monitor_tokens = prepare_token_loaders(
        tokenizer,
        train_loader,
        monitor_loader,
        out_dir,
        tokenizer_id=tokenizer_id,
        data_dir=data_dir,
        image_size=image_size,
        monitor_protocol=monitor_protocol,
        device=device,
    )
    config = {
        "vocabulary_size": tokenizer.quantizer.codebook_size,
        "hidden_channels": hidden_channels,
        "layers": layers,
    }
    prior = PixelCNNPrior(**config).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
    stage = TokenizerStage(
        out_dir / "prior",
        models={"model": prior},
        optimizers={"model": optimizer},
        metadata={
            **image_contract(image_size),
            "model_name": model_name,
            "model_config": config,
            "tokenizer_id": tokenizer_id,
            "monitor_protocol": monitor_protocol,
            "selection_metric": "training_subset_nll",
        },
        recipe={"lr": lr, "batch_size": train_loader.batch_size, "seed": seed},
        resume=resume,
    )
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    for epoch in stage.epochs(epochs):
        if stage.needs_training(epoch):
            seed_epoch_loader(tokens, seed, epoch)
            with tqdm(
                total=len(tokens),
                desc=f"PixelCNN {epoch}/{epochs}",
                unit="batch",
                mininterval=progress_interval,
            ) as progress:
                nll = train_pixelcnn_prior_epoch(
                    prior,
                    tokens,
                    optimizer,
                    device,
                    progress=progress,
                    log_every=log_every,
                )
            stage.record_training(
                epoch, {"nll": nll, "bits_per_token": nll / math.log(2)}
            )
        validation = evaluate_pixelcnn_prior(
            prior,
            monitor_tokens,
            tokens_per_image=grid_size**2,
            device=device,
        )
        stage.record_validation(
            epoch, validation, score=validation["nll_nats_per_token"]
        )
        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            # Preview randomness must not change the resumed optimization stream.
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                prior.eval()
                samples = sample_pixelcnn_prior_images(
                    tokenizer,
                    prior,
                    sample_count,
                    grid_size=grid_size,
                    device=device,
                    temperature=temperature,
                )
            save_image(
                samples.mul(0.5).add(0.5),
                training_dir / f"prior_epoch_{epoch:03d}.png",
                nrow=sample_columns,
            )
    stage.export_best(out_dir / checkpoint_name)
    save_training_metrics(
        stage.history, out_dir, prefix="prior", max_panels=max_metric_panels
    )
