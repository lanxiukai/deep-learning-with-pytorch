"""Shared data, monitoring, caches, and artifacts for discrete-tokenizer lessons.

VQ-VAE, FSQ, and VQGAN reuse this infrastructure. Tokenizer optimization
and VQGAN objectives remain in the lesson scripts. The frozen-token
PixelCNN workflow for VQ-VAE and FSQ lives in pixelcnn_training.
Validation / val_* artifact fields store these training-set diagnostics.
Recovery replays monitoring after an already saved training epoch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.data.vision import image_folder_dataset
from dl_utils.plot.curves import save_loss_panels
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    atomic_torch_save,
    load_model_weights,
)
from dl_utils.training.history import save_metrics_csv
from dl_utils.training.metrics import MetricAccumulator
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS


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
        drop_last=False,
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
    }
    return loader, protocol


def seed_epoch_loader(loader: DataLoader, seed: int, epoch: int) -> None:
    """Keep epoch permutations reproducible across worker restarts."""
    loader.generator = torch.Generator().manual_seed(seed + epoch)
    if isinstance(loader.sampler, RandomSampler):
        loader.sampler.generator = torch.Generator().manual_seed(seed + epoch)


def _dataset_indices(dataset):
    """Resolve nested subsets to their base dataset and ordered indices."""
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [dataset.indices[index] for index in indices]
        dataset = dataset.dataset
    return dataset, indices


def _image_manifest(dataset) -> dict[str, object] | None:
    """Describe the actual ordered files; other dataset types are not cached across calls."""
    dataset, indices = _dataset_indices(dataset)
    if not isinstance(dataset, ImageFolder):
        return None
    files = []
    for index in indices:
        filename, label = dataset.samples[index]
        path = Path(filename)
        stat = path.stat()
        files.append(
            (
                path.relative_to(dataset.root).as_posix(),
                label,
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    return {"root": str(Path(dataset.root).resolve()), "files": files}


@torch.no_grad()
def cached_token_loader(
    tokenizer, image_loader, path, *, tokenizer_id, metadata, device, shuffle
) -> DataLoader:
    """Encode deterministic images once and bind the cache to frozen weights.

    Folder labels condition the VQGAN Transformer prior. VQ-VAE and FSQ
    keep the same batch format and ignore those labels. File manifests detect
    ordinary image replacements via paths, labels, sizes, and modification times.
    Datasets without inspectable files are re-encoded on each call.
    """
    path = Path(path)
    manifest = _image_manifest(image_loader.dataset)
    cache_metadata = {
        **metadata,
        "tokenizer_id": tokenizer_id,
        "examples": len(image_loader.dataset),
        "image_manifest": manifest,
    }
    payload = (
        torch.load(path, map_location="cpu", weights_only=True)
        if manifest is not None and path.is_file()
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
        drop_last=False,
    )


def prepare_token_loaders(
    tokenizer,
    train_loader,
    monitor_loader,
    out_dir,
    *,
    tokenizer_id,
    data_dir,
    image_size,
    device,
    conditional=False,
) -> tuple[DataLoader, DataLoader]:
    """Encode training images once; monitoring shares those cached tokens."""
    cache_metadata = {
        **image_contract(image_size, conditional=conditional),
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
    train_base, train_indices = _dataset_indices(train_loader.dataset)
    monitor_base, monitor_indices = _dataset_indices(monitor_loader.dataset)
    same_source = train_base is monitor_base
    if isinstance(train_base, ImageFolder) and isinstance(monitor_base, ImageFolder):
        same_source = (
            Path(train_base.root).resolve() == Path(monitor_base.root).resolve()
            and train_base.samples == monitor_base.samples
            and repr(train_base.transform) == repr(monitor_base.transform)
        )
    if not same_source:
        raise ValueError("Monitoring must use a subset of the training images.")
    positions = {index: position for position, index in enumerate(train_indices)}
    if any(index not in positions for index in monitor_indices):
        raise ValueError("Monitoring images are missing from the training token cache.")
    monitor_tokens = make_device_aware_loader(
        Subset(tokens.dataset, [positions[index] for index in monitor_indices]),
        monitor_loader.batch_size,
        device,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    monitor_tokens.generator = torch.Generator().manual_seed(0)
    (out_dir / "token_cache_monitor.pth").unlink(missing_ok=True)
    return tokens, monitor_tokens


@torch.inference_mode()
def evaluate_mse_tokenizer(model, loader, *, device) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator(("mse",), device=device)
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        reconstruction = model(images)[0]
        per_image_mse = (reconstruction - images).square().flatten(1).mean(1)
        metrics.add_batch_means(
            (per_image_mse.mean(),),
            num_examples=images.shape[0],
        )
    return metrics.compute_weighted_means(require_finite=True)


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

    def __init__(
        self,
        directory,
        *,
        models,
        optimizers,
        metadata,
        recipe,
        resume,
        metric_names=None,
    ):
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
        previous_protocol = metadata.get("monitor_protocol")
        if resume_path is not None:
            saved = torch.load(
                resume_path, map_location="cpu", weights_only=False, mmap=True
            )
            saved_model = saved["metadata"]["model"]
            stable = {
                key: value
                for key, value in metadata.items()
                if key != "monitor_protocol"
            }
            saved_stable = {
                key: value
                for key, value in saved_model.items()
                if key != "monitor_protocol"
            }
            if stable != saved_stable:
                raise ValueError("Checkpoint model metadata mismatch.")
            # Monitoring policy may change; architecture and recipe still match strictly.
            self.checkpoint.metadata["model"] = saved_model
            previous_protocol = saved_model.get("monitor_protocol")
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
        self.checkpoint.metadata.update(model=metadata)
        self.metric_names = metric_names
        if metric_names is not None:
            history = []
            for row in self.history:
                row = dict(row)
                if "val_nll_nats_per_token" in row:
                    row["val_nll"] = row.pop("val_nll_nats_per_token")
                history.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key == "epoch" or key in metric_names
                    }
                )
            self.state["history"] = history
        protocol_keys = ("image_ids", "subset_seed", "examples")
        if previous_protocol is not None and any(
            previous_protocol.get(key) != metadata["monitor_protocol"].get(key)
            for key in protocol_keys
        ):
            self.state.update(
                best_score=math.inf, best_epoch=None, pending_validation=True
            )
        (self.directory / "last.pth").unlink(missing_ok=True)

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

    def should_monitor(self, epoch, total_epochs, every):
        return (
            self.state["best_epoch"] is None
            or epoch % every == 0
            or epoch == total_epochs
        )

    def record_training(self, epoch, metrics):
        """Save recoverable state before subset monitoring or plotting."""
        self.history.append({"epoch": epoch, **metrics})
        self.state["pending_validation"] = True
        self.completed_epoch = epoch
        self.checkpoint.save(epoch, self.state)

    def finish_epoch(self, epoch, validation=None, *, score=None):
        """Finish an epoch, optionally selecting weights on the monitoring subset."""
        if validation is not None:
            if score is None or not math.isfinite(score):
                raise FloatingPointError("The model-selection metric must be finite.")
            if score < self.state["best_score"]:
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
                atomic_torch_save(payload, self.directory / "best.pth")
                self.state.update(best_score=score, best_epoch=epoch)
            self.history[-1].update(
                {f"val_{key}": value for key, value in validation.items()}
            )
        self.state["pending_validation"] = False
        self.checkpoint.save(epoch, self.state)
        keys = dict.fromkeys(key for row in self.history for key in row)
        save_metrics_csv(
            {key: [row.get(key, math.nan) for row in self.history] for key in keys},
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
        if self.metric_names is not None:
            validation = dict(payload["validation"])
            if "nll_nats_per_token" in validation:
                validation["nll"] = validation.pop("nll_nats_per_token")
            payload["validation"] = {
                key: value
                for key, value in validation.items()
                if f"val_{key}" in self.metric_names
            }
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


def save_stage_loss_curves(stage: TokenizerStage) -> None:
    """Render only the basic training losses beside the stage's single CSV."""
    names = [
        name
        for name in (stage.metric_names or stage.history[0])
        if name != "epoch" and not name.startswith("val_")
    ]
    panels = {}
    for name in names:
        series = {"training": [row.get(name, math.nan) for row in stage.history]}
        panels[name.replace("_", " ").capitalize()] = series
    save_loss_panels(
        [row["epoch"] for row in stage.history],
        panels,
        stage.directory / "loss_curves.png",
        xlabel="Epoch",
        ylabel="Loss",
    )
    prefix = stage.directory.name
    (stage.directory.parent / f"{prefix}_metrics.csv").unlink(missing_ok=True)
    for path in stage.directory.parent.glob(f"{prefix}_metrics_*.png"):
        path.unlink()


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
    expected_classes = len(GLASSES_CLASS_NAMES) if conditional else 0
    actual_classes = getattr(prior, "num_classes", 0)
    if actual_classes != expected_classes:
        raise ValueError("Prior conditioning differs from the image contract.")
    positions = (image_size // (2**tokenizer.downsample_steps)) ** 2
    # Transformer sequence length is fixed; PixelCNN accepts a grid at sampling.
    if hasattr(prior, "sequence_length") and prior.sequence_length != positions:
        raise ValueError("Prior sequence length differs from the tokenizer grid.")
    return cast(T, prior.requires_grad_(False))
