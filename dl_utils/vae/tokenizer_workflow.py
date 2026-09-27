"""Data and artifact helpers for the three discrete-tokenizer lessons.

Optimization and validation remain in the lesson scripts. Recovery replays
validation if an interruption follows an already saved training epoch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset

from dl_utils.data.datasets.celeba import (
    CELEBA_ALIGNED_CROP_SIZE,
    CELEBA_SMILING_ATTRIBUTE,
    CELEBA_SMILING_CLASSES,
    CelebAAlignedDataset,
    make_aligned_celeba_loader,
)
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.data.vision import image_folder_dataset
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    atomic_torch_save,
    load_model_weights,
)
from dl_utils.training.history import save_metrics_csv
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS


def image_contract(image_size: int, *, dataset: str = "celeba") -> dict[str, Any]:
    """Describe the input contract for the selected tokenizer lesson."""
    if dataset == "glasses-256":
        return {
            "dataset": dataset,
            "image_size": image_size,
            "crop_size": None,
            "normalization": "[-1,1]",
            "horizontal_flip": False,
            "attribute": None,
            "class_names": [],
            "conditioning": "unconditional",
        }
    if dataset != "celeba":
        raise ValueError(f"Unknown tokenizer dataset: {dataset}")
    return {
        "dataset": "celeba",
        "image_size": image_size,
        "crop_size": CELEBA_ALIGNED_CROP_SIZE,
        "normalization": "[-1,1]",
        "horizontal_flip": False,
        "attribute": CELEBA_SMILING_ATTRIBUTE,
        "class_names": list(CELEBA_SMILING_CLASSES),
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
) -> tuple[DataLoader, dict[str, Any]]:
    """Reuse the VAE's training images; a seeded subset is only a diagnostic."""
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or None (all training images).")
    root = Path(root)
    dataset = image_folder_dataset(
        root, resize=(image_size, image_size), normalize=(0.5, 0.5)
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
        **image_contract(image_size, dataset="glasses-256"),
        "split": "training",
        "evaluation_scope": "training-set diagnostics, not held-out performance",
        "subset_seed": seed,
        "examples": len(indices),
        "image_ids": [
            Path(dataset.samples[i][0]).relative_to(root).as_posix() for i in indices
        ],
        "psnr_aggregation": "both mean per-image PSNR and PSNR from pooled MSE; data_range=2",
        "rate_interpretation": "unconditional token cross-entropy; excludes model weights and coder overhead",
        "latent_mse_interpretation": "within-tokenizer diagnostic, not comparable across quantizer coordinate systems",
    }
    return loader, protocol


def heldout_loader(
    root, image_size, batch_size, device, *, split, max_examples, seed, num_workers
) -> tuple[DataLoader, dict[str, Any]]:
    """Use a recorded random subset for monitoring, or the complete test split."""
    if split not in {"validation", "test"}:
        raise ValueError("Held-out evaluation requires validation or test images.")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or None (the full split).")
    loader = make_aligned_celeba_loader(
        root,
        image_size,
        batch_size,
        device,
        split=split,
        attribute=CELEBA_SMILING_ATTRIBUTE,
        horizontal_flip=False,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    dataset = cast(CelebAAlignedDataset, loader.dataset)
    indices = list(range(len(dataset)))
    if max_examples is not None and max_examples < len(indices):
        generator = torch.Generator().manual_seed(seed)
        indices = (
            torch.randperm(len(indices), generator=generator)[:max_examples]
            .sort()
            .values.tolist()
        )
        loader = make_device_aware_loader(
            Subset(dataset, indices),
            batch_size,
            device,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    # DataLoader worker seeds must not consume the model's random stream.
    loader.generator = torch.Generator().manual_seed(seed)
    protocol = {
        **image_contract(image_size),
        "split": split,
        "subset_seed": seed,
        "examples": len(indices),
        "image_ids": [dataset.image_paths[i].name for i in indices],
        "psnr_aggregation": "both mean per-image PSNR and PSNR from pooled MSE; data_range=2",
        "rate_interpretation": "conditional token cross-entropy; excludes labels, model weights and coder overhead",
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
    """Encode each deterministic image once and bind the cache to frozen weights."""
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


class TokenizerStage:
    """Thin artifact wrapper around TrainingCheckpoint; no training callbacks."""

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
        """Save recoverable state before validation or optional plotting."""
        self.history.append({"epoch": epoch, **metrics})
        self.state["pending_validation"] = True
        self.completed_epoch = epoch
        self.checkpoint.save(epoch, self.state)

    def record_validation(self, epoch, validation, *, score):
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
        """Select frozen weights only after their validation has completed."""
        if self.state["pending_validation"] or self.state["best_epoch"] is None:
            raise RuntimeError("Finish validation before exporting this stage.")
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
    dataset="celeba",
    downsample_steps=TOKENIZER_DOWNSAMPLE_STEPS,
) -> tuple[T, dict[str, Any]]:
    """Reuse the common weight loader and return the identity needed by priors."""
    model, _ = load_model_weights(
        path,
        model_class,
        device=device,
        expected_metadata={
            **image_contract(image_size, dataset=dataset),
            "model_name": name,
        },
        config_transform=lambda config: _current_tokenizer_config(
            config, downsample_steps
        ),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
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
    dataset="celeba",
) -> T:
    """Reject mixed runs, labels, image preprocessing, vocabularies and grids."""
    prior, _ = load_model_weights(
        path,
        model_class,
        device=device,
        expected_metadata={
            **image_contract(image_size, dataset=dataset),
            "model_name": name,
            "tokenizer_id": tokenizer_payload["snapshot_id"],
        },
    )
    if prior.vocabulary_size != tokenizer.quantizer.codebook_size:
        raise ValueError("Prior vocabulary differs from the frozen tokenizer.")
    expected_classes = len(image_contract(image_size, dataset=dataset)["class_names"])
    if prior.num_classes != expected_classes:
        raise ValueError("Prior conditioning differs from the image contract.")
    positions = (image_size // (2**tokenizer.downsample_steps)) ** 2
    if hasattr(prior, "sequence_length") and prior.sequence_length != positions:
        raise ValueError("Prior sequence length differs from the tokenizer grid.")
    return cast(T, prior.requires_grad_(False))
