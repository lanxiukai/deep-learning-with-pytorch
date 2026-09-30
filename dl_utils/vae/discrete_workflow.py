"""Data, monitoring, artifacts, and shared training for discrete-tokenizer lessons.

Tokenizer optimization and VQGAN objectives remain in the lesson scripts.
VQ-VAE and FSQ share the frozen-token PixelCNN training workflow here.
Validation / val_* artifact fields store these training-set diagnostics.
Recovery replays monitoring after an already saved training epoch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.data.vision import image_folder_dataset
from dl_utils.training.artifacts import save_training_metrics
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    atomic_torch_save,
    load_model_weights,
)
from dl_utils.training.history import save_metrics_csv
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.pixelcnn_prior import PixelCNNPrior
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS, VQVAE, FSQAutoencoder


def _token_usage_from_counts(counts: Tensor) -> dict[str, Tensor]:
    probabilities = counts / counts.sum().clamp_min(1.0)
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return {
        "perplexity": entropy.exp().detach(),
        "active_codes": nonzero.sum().detach(),
        "usage_fraction": nonzero.float().mean().detach(),
        "token_entropy_nats": entropy.detach(),
    }


class TokenUsageAccumulator:
    """Accumulate exact token counts over an epoch or evaluation window."""

    def __init__(self, vocabulary_size: int) -> None:
        self.vocabulary_size = vocabulary_size
        self.counts: Tensor | None = None

    def update(self, indices: Tensor) -> None:
        counts = torch.bincount(
            indices.detach().reshape(-1), minlength=self.vocabulary_size
        )
        if self.counts is None:
            self.counts = torch.zeros_like(counts)
        self.counts += counts

    def statistics(self) -> dict[str, Tensor]:
        if self.counts is None:
            raise ValueError("Token usage requires at least one observed batch.")
        return _token_usage_from_counts(self.counts.float())

    def training_metrics(self) -> dict[str, float]:
        statistics = self.statistics()
        return {
            "perplexity": statistics["perplexity"].item(),
            "active_codes": statistics["active_codes"].item(),
            "entropy_bits": statistics["token_entropy_nats"].item() / math.log(2),
        }

    def rate_metrics(self, tokens_per_image: int) -> dict[str, float]:
        statistics = self.statistics()
        entropy_bits = statistics["token_entropy_nats"].item() / math.log(2)
        return {
            "perplexity": statistics["perplexity"].item(),
            "active_codes": statistics["active_codes"].item(),
            "usage_fraction": statistics["usage_fraction"].item(),
            "marginal_entropy_bits_per_token": entropy_bits,
            "marginal_entropy_bits_per_image": tokens_per_image * entropy_bits,
            "fixed_length_bits_per_image": tokens_per_image
            * math.ceil(math.log2(self.vocabulary_size)),
        }


def image_contract(image_size: int, *, conditional: bool = False) -> dict[str, Any]:
    """Record glasses-256 preprocessing and the selected prior conditioning."""
    return {
        "dataset": "glasses-256",
        "image_size": image_size,
        "crop_size": None,
        "normalization": "[-1,1]",
        "horizontal_flip": False,
        "attribute": "glasses" if conditional else None,
        "class_names": list(GLASSES_CLASS_NAMES) if conditional else [],
        "conditioning": "class-conditional" if conditional else "unconditional",
    }


def glasses_loader(
    root,
    image_size,
    batch_size,
    device,
    *,
    shuffle=False,
    max_examples=None,
    seed=123,
    num_workers=0,
    conditional=False,
) -> tuple[DataLoader, dict[str, Any]]:
    """Reuse the VAE's training images; a seeded subset is only a diagnostic."""
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or None (all training images).")
    root = Path(root)
    dataset = image_folder_dataset(
        root, resize=(image_size, image_size), normalize=(0.5, 0.5)
    )
    if conditional and dataset.classes != list(GLASSES_CLASS_NAMES):
        raise ValueError(
            f"Expected conditional classes {GLASSES_CLASS_NAMES}, got {dataset.classes}"
        )
    indices = list(range(len(dataset)))
    if max_examples is not None and max_examples < len(indices):
        indices = torch.randperm(
            len(indices), generator=torch.Generator().manual_seed(seed)
        )[:max_examples].tolist()
    loader = make_device_aware_loader(
        Subset(dataset, indices),
        batch_size,
        device,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
    )
    loader.generator = torch.Generator().manual_seed(seed)
    protocol = {
        **image_contract(image_size, conditional=conditional),
        "split": "training",
        "evaluation_scope": "training-set diagnostics, not held-out performance",
        "subset_seed": seed,
        "examples": len(indices),
        "image_ids": [
            Path(dataset.samples[i][0]).relative_to(root).as_posix() for i in indices
        ],
        "psnr_aggregation": "both mean per-image PSNR and PSNR from pooled MSE; data_range=2",
        "rate_interpretation": (
            "conditional token cross-entropy; excludes labels, model weights and coder overhead"
            if conditional
            else "unconditional token cross-entropy; excludes model weights and coder overhead"
        ),
        "latent_mse_interpretation": "within-tokenizer diagnostic, not comparable across quantizer coordinate systems",
    }
    return loader, protocol


def seed_epoch_loader(loader: DataLoader, seed: int, epoch: int) -> None:
    """Keep epoch permutations reproducible across worker restarts."""
    loader.generator = torch.Generator().manual_seed(seed + epoch)
    if isinstance(loader.sampler, RandomSampler):
        loader.sampler.generator = torch.Generator().manual_seed(seed + epoch)


@torch.no_grad()
def cached_token_loader(
    tokenizer, image_loader, path, *, tokenizer_id, metadata, device, shuffle
) -> DataLoader:
    """Encode deterministic images once and bind the cache to frozen weights.

    Folder labels condition the VQGAN Transformer prior. VQ-VAE and FSQ
    keep the same batch format and ignore those labels.
    """
    path = Path(path)
    cache_metadata = {
        **metadata,
        "tokenizer_id": tokenizer_id,
        "examples": len(image_loader.dataset),
    }
    payload = (
        torch.load(path, map_location="cpu", weights_only=True)
        if path.is_file()
        else None
    )
    if payload is None or payload.get("metadata") != cache_metadata:
        tokenizer.eval().requires_grad_(False)
        # Include the last partial batch and visit the dataset in stable order.
        source = make_device_aware_loader(
            image_loader.dataset,
            image_loader.batch_size,
            device,
            shuffle=False,
            num_workers=image_loader.num_workers,
            drop_last=False,
        )
        source.generator = torch.Generator().manual_seed(0)
        dtype = (
            torch.int16 if tokenizer.quantizer.codebook_size <= 32768 else torch.int32
        )
        tokens, labels = [], []
        for images, batch_labels in source:
            indices = tokenizer.encode_indices(images.to(device, non_blocking=True))
            tokens.append(indices.cpu().to(dtype))
            labels.append(batch_labels.cpu())
        if not tokens:
            raise ValueError("Cannot cache an empty image dataset.")
        payload = {
            "metadata": cache_metadata,
            "tokens": torch.cat(tokens),
            "labels": torch.cat(labels),
        }
        atomic_torch_save(payload, path)
    return make_device_aware_loader(
        TensorDataset(payload["tokens"], payload["labels"]),
        image_loader.batch_size,
        device,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
    )


@torch.inference_mode()
def evaluate_mse_tokenizer(
    model, loader, *, tokens_per_image, device
) -> dict[str, float]:
    model.eval()
    vocabulary_size = model.quantizer.codebook_size
    metrics = MetricAccumulator(
        ("mse", "mean_psnr_db", "quantization_mse"), device=device
    )
    usage = TokenUsageAccumulator(vocabulary_size)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction, indices, *_, diagnostics = model(images)
        per_image_mse = (reconstruction - images).square().flatten(1).mean(1)
        metrics.add_batch_means(
            (
                per_image_mse.mean(),
                (10 * torch.log10(4 / per_image_mse.clamp_min(1e-12))).mean(),
                diagnostics["quantization_mse"],
            ),
            num_examples=images.shape[0],
        )
        usage.update(indices)
    means = metrics.compute_weighted_means(require_finite=True)
    return {
        **means,
        "psnr_from_pooled_mse_db": 10 * math.log10(4 / max(means["mse"], 1e-12)),
        **usage.rate_metrics(tokens_per_image),
    }


@torch.inference_mode()
def save_tokenizer_preview(model, loader, path, *, count, device) -> None:
    """Save the first monitoring images above their reconstructions."""
    images = next(iter(loader))[0][:count].to(device)
    reconstruction = model(images)[0]
    save_image(
        torch.cat((images, reconstruction)).mul(0.5).add(0.5), path, nrow=len(images)
    )


class TokenizerStage:
    """Persist training and subset monitoring through TrainingCheckpoint."""

    def __init__(self, directory, *, models, optimizers, metadata, recipe, resume):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = dict(models)
        self.metadata = dict(metadata)
        self.checkpoint = TrainingCheckpoint(
            self.directory / "latest.pth",
            unit="epoch",
            models=models,
            optimizers=optimizers,
        )
        self.checkpoint.metadata.update(model=metadata, recipe=recipe)
        resume_path = (
            self.checkpoint.path if resume and self.checkpoint.path.is_file() else None
        )
        self.completed_epoch, self.state = self.checkpoint.resume(
            resume_path,
            initial_state={
                "run_id": uuid4().hex,
                "history": [],
                "best_score": math.inf,
                "best_epoch": None,
                "pending_validation": False,
            },
        )

    @property
    def history(self):
        return self.state["history"]

    def epochs(self, total_epochs):
        if total_epochs < 1 or self.completed_epoch > total_epochs:
            raise ValueError(
                "Epoch budget must be positive and cannot precede the checkpoint."
            )
        start = self.completed_epoch + 1 - int(self.state["pending_validation"])
        return range(start, total_epochs + 1)

    def needs_training(self, epoch):
        return not (epoch == self.completed_epoch and self.state["pending_validation"])

    def record_training(self, epoch, metrics):
        """Save recoverable state before subset monitoring or plotting."""
        self.history.append({"epoch": epoch, **metrics})
        self.state["pending_validation"] = True
        self.completed_epoch = epoch
        self.checkpoint.save(epoch, self.state)

    def record_validation(self, epoch, validation, *, score):
        """Store training-subset diagnostics in validation / val_* fields."""
        if not math.isfinite(score):
            raise FloatingPointError("The model-selection metric must be finite.")
        payload = {
            **self.metadata,
            "epoch": epoch,
            "validation": validation,
            "snapshot_id": f"{self.state['run_id']}:{epoch}",
            "state_dict": self.models["model"].state_dict(),
            **{
                f"{name}_state_dict": model.state_dict()
                for name, model in self.models.items()
                if name != "model"
            },
        }
        atomic_torch_save(payload, self.directory / "last.pth")
        if score < self.state["best_score"]:
            atomic_torch_save(payload, self.directory / "best.pth")
            self.state.update(best_score=score, best_epoch=epoch)
        self.history[-1].update(
            {f"val_{key}": value for key, value in validation.items()}
        )
        self.state["pending_validation"] = False
        self.checkpoint.save(epoch, self.state)
        save_metrics_csv(
            {key: [row[key] for row in self.history] for key in self.history[0]},
            self.directory / "metrics.csv",
        )

    def export_best(self, destination):
        """Select frozen weights after training-subset monitoring completes."""
        if self.state["pending_validation"] or self.state["best_epoch"] is None:
            raise RuntimeError(
                "Finish training-subset monitoring before exporting this stage."
            )
        payload = torch.load(
            self.directory / "best.pth", map_location="cpu", weights_only=True
        )
        expected_id = f"{self.state['run_id']}:{self.state['best_epoch']}"
        if payload["snapshot_id"] != expected_id:
            raise ValueError(
                "Best weights do not belong to the resumed training stage."
            )
        for name, model in self.models.items():
            key = "state_dict" if name == "model" else f"{name}_state_dict"
            model.load_state_dict(payload[key])
        atomic_torch_save(payload, destination)
        return payload


def _current_tokenizer_config(
    config: dict[str, Any], downsample_steps: int
) -> dict[str, Any]:
    """Accept only the spatial compression used by the current lessons."""
    if config.get("downsample_steps") != downsample_steps:
        raise ValueError(
            "Tokenizer downsample_steps must match the current configuration."
        )
    return config


def load_tokenizer_weights[T: nn.Module](
    path,
    model_class: type[T],
    *,
    name,
    image_size,
    device,
    downsample_steps=TOKENIZER_DOWNSAMPLE_STEPS,
    conditional=False,
) -> tuple[T, dict[str, Any]]:
    """Reuse the common weight loader and return the identity needed by priors."""
    model, payload = load_model_weights(
        path,
        model_class,
        device=device,
        expected_metadata={
            **image_contract(image_size, conditional=conditional),
            "model_name": name,
        },
        return_checkpoint=True,
        config_transform=lambda config: _current_tokenizer_config(
            config, downsample_steps
        ),
    )
    if not isinstance(payload.get("snapshot_id"), str):
        raise TypeError("Tokenizer weights require a snapshot identity.")
    return cast(T, model.requires_grad_(False)), payload


def load_prior_weights[T: nn.Module](
    path,
    model_class: type[T],
    *,
    name,
    image_size,
    tokenizer,
    tokenizer_payload,
    device,
    conditional=False,
) -> T:
    """Require matching snapshots, conditioning, vocabulary and token grid."""
    prior, _ = load_model_weights(
        path,
        model_class,
        device=device,
        expected_metadata={
            **image_contract(image_size, conditional=conditional),
            "model_name": name,
            "tokenizer_id": tokenizer_payload["snapshot_id"],
        },
    )
    if prior.vocabulary_size != tokenizer.quantizer.codebook_size:
        raise ValueError("Prior vocabulary differs from the frozen tokenizer.")
    expected_classes = len(
        image_contract(image_size, conditional=conditional)["class_names"]
    )
    if prior.num_classes != expected_classes:
        raise ValueError("Prior conditioning differs from the image contract.")
    positions = (image_size // (2**tokenizer.downsample_steps)) ** 2
    # Transformer sequence length is fixed; PixelCNN accepts a grid at sampling.
    if hasattr(prior, "sequence_length") and prior.sequence_length != positions:
        raise ValueError("Prior sequence length differs from the tokenizer grid.")
    return cast(T, prior.requires_grad_(False))


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
    for batch_index, (indices, labels) in enumerate(loader, 1):
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if prior.num_classes else None
        loss = F.cross_entropy(prior(indices, labels=labels), indices)
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
    for indices, labels in loader:
        indices = indices.to(device=device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if prior.num_classes else None
        loss = F.cross_entropy(prior(indices, labels=labels), indices)
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
    labels: Tensor | None = None,
) -> Tensor:
    """Sample a square token grid and decode it to an image batch."""
    indices = prior.sample(
        count,
        grid_size,
        grid_size,
        device=device,
        labels=labels.to(device) if labels is not None else None,
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
    cache_metadata = {
        **image_contract(image_size),
        "source_root": str(data_dir.resolve()),
    }
    tokens = cached_token_loader(
        tokenizer,
        train_loader,
        out_dir / "token_cache_train.pth",
        tokenizer_id=tokenizer_id,
        metadata={**cache_metadata, "split": "train"},
        device=device,
        shuffle=True,
    )
    monitor_tokens = cached_token_loader(
        tokenizer,
        monitor_loader,
        out_dir / "token_cache_monitor.pth",
        tokenizer_id=tokenizer_id,
        metadata={**cache_metadata, "protocol": monitor_protocol},
        device=device,
        shuffle=False,
    )
    config = {
        "vocabulary_size": tokenizer.quantizer.codebook_size,
        "hidden_channels": hidden_channels,
        "layers": layers,
        "num_classes": 0,
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
