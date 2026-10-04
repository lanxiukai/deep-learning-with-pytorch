"""Small data, checkpoint, and image helpers for the discrete-tokenizer lessons.

The lesson scripts own the stage order; PixelCNN training is shared separately.
Tokens are encoded once into memory. Final checkpoints store both models together.
"""

from collections.abc import Mapping, Sequence, Sized
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.utils import save_image

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.data.vision import image_folder_dataset
from dl_utils.filesystem.directories import reset_dir
from dl_utils.plot.curves import save_loss_panels
from dl_utils.training.checkpoints import TrainingCheckpoint, atomic_torch_save
from dl_utils.vae.perceptual_autoencoder import VQPerceptualAutoencoder
from dl_utils.vae.quantization import VQVAE, FSQAutoencoder, validate_image_size
from dl_utils.vae.token_priors import CausalTransformerPrior, PixelCNNPrior

MODEL_TYPES = {
    "vq_vae": (VQVAE, PixelCNNPrior),
    "fsq": (FSQAutoencoder, PixelCNNPrior),
    "vqgan": (VQPerceptualAutoencoder, CausalTransformerPrior),
}
PAIR_FORMAT = "discrete-pair-v1"


def prepare_training_output(
    output_dir: Path, *, resume: bool, recipe: Mapping[str, Any]
) -> None:
    """Keep the output root and reset planned previews once at startup."""
    write_previews = True
    if resume:
        for name in ("prior_latest.pth", "tokenizer_latest.pth"):
            path = output_dir / name
            if path.is_file():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint["metadata"] != {"unit": "epoch", "recipe": dict(recipe)}:
                    raise ValueError(
                        "Checkpoint metadata mismatch; use the same recipe."
                    )
                write_previews = (
                    name != "prior_latest.pth"
                    or checkpoint["epoch"] < recipe["prior_epochs"]
                )
                break
    if not output_dir.exists():
        reset_dir(output_dir)
    if not resume:
        (output_dir / "tokenizer_latest.pth").unlink(missing_ok=True)
        (output_dir / "prior_latest.pth").unlink(missing_ok=True)
    if write_previews:
        reset_dir(output_dir / "training")


def glasses_loader(
    root: str | Path,
    image_size: int,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
    conditional: bool = False,
) -> DataLoader:
    dataset = image_folder_dataset(
        root, resize=(image_size, image_size), normalize=(0.5, 0.5)
    )
    if conditional and dataset.classes != list(GLASSES_CLASS_NAMES):
        raise ValueError(
            f"Expected classes {GLASSES_CLASS_NAMES}, got {dataset.classes}"
        )
    loader = make_device_aware_loader(
        dataset,
        batch_size,
        device,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    loader.generator = torch.Generator().manual_seed(0)
    return loader


def seed_epoch_loader(loader: DataLoader, seed: int, epoch: int) -> None:
    """Give each epoch its own shuffle order, including after a restart."""
    loader.generator = torch.Generator().manual_seed(seed + epoch)
    if isinstance(loader.sampler, RandomSampler):
        loader.sampler.generator = torch.Generator().manual_seed(seed + epoch)


def fixed_images(loader: DataLoader, count: int) -> Tensor:
    """Read a few fixed originals without advancing the training iterator."""
    return torch.stack(
        [
            loader.dataset[index][0]
            for index in range(min(count, len(cast(Sized, loader.dataset))))
        ]
    )


@torch.no_grad()
def encode_dataset(
    tokenizer: VQVAE | FSQAutoencoder | VQPerceptualAutoencoder,
    image_loader: DataLoader,
    device: torch.device,
) -> DataLoader:
    """Freeze the tokenizer and encode each image once, in stable dataset order."""
    tokenizer.eval().requires_grad_(False)
    batch_size = cast(int, image_loader.batch_size)
    source = make_device_aware_loader(
        image_loader.dataset,
        batch_size,
        device,
        shuffle=False,
        num_workers=image_loader.num_workers,
        drop_last=False,
    )
    source.generator = torch.Generator().manual_seed(0)
    tokens, labels = [], []
    for images, batch_labels in source:
        tokens.append(tokenizer.encode_indices(images.to(device)).cpu())
        labels.append(batch_labels.cpu())
    return make_device_aware_loader(
        TensorDataset(torch.cat(tokens), torch.cat(labels)),
        batch_size,
        device,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )


def epoch_checkpoint(
    path: str | PathLike[str],
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, Optimizer],
    recipe: Mapping[str, Any],
) -> TrainingCheckpoint:
    """Configure ordinary epoch recovery; the lesson calls resume/save directly."""
    checkpoint = TrainingCheckpoint(
        path, unit="epoch", models=models, optimizers=optimizers
    )
    metadata: dict[str, Any] = checkpoint.metadata
    metadata.update(recipe=recipe)
    return checkpoint


@torch.inference_mode()
def save_reconstruction(
    model: VQVAE | FSQAutoencoder | VQPerceptualAutoencoder,
    originals: Tensor,
    path: str | Path,
    device: torch.device,
) -> None:
    """Save into the training directory initialized by the lesson."""
    model.eval()
    images = originals.to(device)
    reconstruction = model(images)[0]
    save_image(
        torch.cat((images, reconstruction)).mul(0.5).add(0.5), path, nrow=len(images)
    )


def save_loss_curves(
    history: Sequence[Mapping[str, float]], path: str | PathLike[str]
) -> None:
    """One figure for the basic training losses of one stage."""
    panels = {
        name.replace("_", " ").capitalize(): {name: [row[name] for row in history]}
        for name in history[0]
    }
    save_loss_panels(
        range(1, len(history) + 1), panels, path, xlabel="Epoch", ylabel="Loss"
    )


def save_pair(
    path: str | PathLike[str],
    name: str,
    tokenizer: VQVAE | FSQAutoencoder | VQPerceptualAutoencoder,
    prior: PixelCNNPrior | CausalTransformerPrior,
    config: Mapping[str, Any],
) -> None:
    """Store the exact frozen tokenizer used by this prior in the same file."""
    atomic_torch_save(
        {
            "format": PAIR_FORMAT,
            "model_name": name,
            "config": config,
            "class_names": list(GLASSES_CLASS_NAMES) if name == "vqgan" else [],
            "tokenizer": tokenizer.state_dict(),
            "prior": prior.state_dict(),
        },
        path,
    )


def load_pair(
    path: str | PathLike[str],
    name: str,
    device: torch.device,
    *,
    image_size: int = 256,
) -> tuple[
    VQVAE | FSQAutoencoder | VQPerceptualAutoencoder,
    PixelCNNPrior | CausalTransformerPrior,
]:
    """Load a final pair and check the image, vocabulary, grid, and label shapes."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != PAIR_FORMAT:
        raise ValueError("Expected a discrete-pair-v1 checkpoint.")
    config = payload["config"]
    classes = list(GLASSES_CLASS_NAMES) if name == "vqgan" else []
    if (
        payload["model_name"] != name
        or config["image_size"] != image_size
        or payload["class_names"] != classes
    ):
        raise ValueError(
            "Model name, image size, or class names do not match this lesson."
        )
    tokenizer_type, prior_type = MODEL_TYPES[name]
    tokenizer = tokenizer_type(**config["tokenizer"]).to(device)
    prior = prior_type(**config["prior"]).to(device)
    tokenizer.load_state_dict(payload["tokenizer"])
    prior.load_state_dict(payload["prior"])
    validate_image_size(image_size, image_size, tokenizer.downsample_steps)
    side = image_size // (2**tokenizer.downsample_steps)
    if (
        prior.vocabulary_size != tokenizer.quantizer.codebook_size
        or getattr(prior, "sequence_length", side**2) != side**2
    ):
        raise ValueError("Prior vocabulary or token grid does not match the tokenizer.")
    if getattr(prior, "num_classes", 0) != len(classes):
        raise ValueError("Prior conditioning does not match the class names.")
    return tokenizer.eval().requires_grad_(False), prior.eval().requires_grad_(False)
