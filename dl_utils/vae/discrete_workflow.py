"""Small data, checkpoint, and image helpers for the discrete-tokenizer lessons.

The lesson scripts own both epoch loops and their order. Tokens are encoded
once into memory. Final checkpoints store the tokenizer and prior together.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.utils import save_image

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.data.loading import make_device_aware_loader
from dl_utils.data.vision import image_folder_dataset
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


def glasses_loader(
    root,
    image_size,
    batch_size,
    device,
    *,
    shuffle=False,
    num_workers=0,
    conditional=False,
):
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


def fixed_images(loader, count):
    """Read a few fixed originals without advancing the training iterator."""
    return torch.stack(
        [loader.dataset[index][0] for index in range(min(count, len(loader.dataset)))]
    )


@torch.no_grad()
def encode_dataset(tokenizer, image_loader, device):
    """Freeze the tokenizer and encode each image once, in stable dataset order."""
    tokenizer.eval().requires_grad_(False)
    source = make_device_aware_loader(
        image_loader.dataset,
        image_loader.batch_size,
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
        image_loader.batch_size,
        device,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )


def epoch_checkpoint(path, models, optimizers, recipe):
    """Configure ordinary epoch recovery; the lesson calls resume/save directly."""
    checkpoint = TrainingCheckpoint(
        path, unit="epoch", models=models, optimizers=optimizers
    )
    checkpoint.metadata.update(recipe=recipe)
    return checkpoint


@torch.inference_mode()
def save_reconstruction(model, originals, path, device):
    model.eval()
    images = originals.to(device)
    reconstruction = model(images)[0]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat((images, reconstruction)).mul(0.5).add(0.5), path, nrow=len(images)
    )


def save_loss_curves(history, path):
    """One figure for the basic training losses of one stage."""
    panels = {
        name.replace("_", " ").capitalize(): {name: [row[name] for row in history]}
        for name in history[0]
    }
    save_loss_panels(
        range(1, len(history) + 1), panels, path, xlabel="Epoch", ylabel="Loss"
    )


def save_pair(path, name, tokenizer, prior, config):
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


def load_pair(path, name, device, *, image_size=256):
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
