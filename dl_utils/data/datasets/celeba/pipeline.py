"""CPU/PIL and CUDA/nvJPEG loading pipelines for aligned CelebA."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_jpeg

from dl_utils.data.loading import make_device_aware_loader

from .dataset import CelebAAlignedDataset, CelebAEncodedDataset, _aligned_image_paths

CELEBA_ALIGNED_CROP_SIZE = 178


CELEBA_PIPELINES = frozenset({"auto", "cpu", "cuda"})


def _collate_encoded(batch):
    return list(batch), None


def make_encoded_celeba_loader(
    root,
    batch_size: int,
    device: torch.device,
    *,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Load encoded JPEG batches for a CUDA nvJPEG preprocessing path."""
    if device.type != "cuda":
        raise ValueError("encoded CelebA loading requires a CUDA device.")
    dataset = CelebAEncodedDataset(root, split=split)
    return make_device_aware_loader(
        dataset,
        batch_size,
        device,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        collate_fn=_collate_encoded,
        # Raw JPEG tensors fail descriptor-based IPC in the local runtime.
        # Select the transport in workers for every caller, including fresh runs.
        worker_sharing_strategy="file_system",
    )


@torch.no_grad()
def prepare_encoded_celeba_batch(
    encoded_images: list[torch.Tensor],
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode and transform one encoded JPEG batch entirely on CUDA."""
    decoded = decode_jpeg(
        encoded_images,
        mode=ImageReadMode.RGB,
        device=device,
    )
    if not isinstance(decoded, list) or not decoded:
        raise RuntimeError("CUDA JPEG decoding returned an invalid batch.")
    images = torch.stack(decoded)
    height, width = images.shape[-2:]
    if height < CELEBA_ALIGNED_CROP_SIZE or width < CELEBA_ALIGNED_CROP_SIZE:
        raise ValueError(
            f"CelebA image is smaller than the aligned center crop: {height}x{width}"
        )
    top = (height - CELEBA_ALIGNED_CROP_SIZE) // 2
    left = (width - CELEBA_ALIGNED_CROP_SIZE) // 2
    images = images[
        :,
        :,
        top : top + CELEBA_ALIGNED_CROP_SIZE,
        left : left + CELEBA_ALIGNED_CROP_SIZE,
    ].float()
    images = F.interpolate(
        images,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    images.clamp_(0.0, 255.0).mul_(2.0 / 255.0).sub_(1.0)
    flip_mask = torch.rand(len(images), device=device) < 0.5
    images = torch.where(
        flip_mask.view(-1, 1, 1, 1),
        images.flip(-1),
        images,
    )
    return images.contiguous(memory_format=torch.channels_last)


def cuda_jpeg_works(root, device: torch.device) -> tuple[bool, str | None]:
    """Probe nvJPEG with one dataset image and return a fallback reason."""
    if device.type != "cuda":
        return False, "CUDA is unavailable"
    dataset = CelebAEncodedDataset(root)
    try:
        encoded = dataset[0]
        decoded = decode_jpeg(
            [encoded],
            mode=ImageReadMode.RGB,
            device=device,
        )
        if not isinstance(decoded, list) or len(decoded) != 1:
            return False, "nvJPEG returned an invalid probe result"
    except (RuntimeError, ValueError) as error:
        return False, str(error).splitlines()[0]
    return True, None


def resolve_celeba_pipeline(root, device: torch.device, requested="auto") -> str:
    """Resolve CUDA JPEG decoding with an explicit CPU fallback policy."""
    if requested not in CELEBA_PIPELINES:
        choices = ", ".join(sorted(CELEBA_PIPELINES))
        raise ValueError(f"CelebA pipeline must be one of: {choices}.")
    if requested == "cpu":
        return "cpu"
    works, reason = cuda_jpeg_works(root, device)
    if works:
        return "cuda"
    if requested == "cuda":
        raise RuntimeError(f"CUDA JPEG pipeline is unavailable: {reason}")
    print(f"CUDA JPEG pipeline unavailable ({reason}); using CPU/PIL.")
    return "cpu"


def aligned_celeba_transform(resolution: int, *, horizontal_flip: bool = False):
    """Center-crop aligned faces and normalize RGB pixels to [-1, 1]."""
    operations = [
        transforms.CenterCrop(CELEBA_ALIGNED_CROP_SIZE),
        transforms.Resize((resolution, resolution), antialias=True),
    ]
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(
        [*operations, transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )


def make_aligned_celeba_loader(
    root,
    resolution: int,
    batch_size: int,
    device: torch.device,
    *,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    attribute: str | None = None,
    horizontal_flip: bool = True,
) -> DataLoader:
    """Create a normalized, resolution-specific aligned CelebA loader."""
    if resolution < 1 or batch_size < 1:
        raise ValueError("resolution and batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    dataset = CelebAAlignedDataset(
        root,
        split=split,
        transform=aligned_celeba_transform(resolution, horizontal_flip=horizontal_flip),
        attribute=attribute,
    )
    return make_device_aware_loader(
        dataset,
        batch_size,
        device,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )


def make_aligned_celeba_train_validation_loaders(
    root,
    resolution: int,
    batch_size: int,
    device: torch.device,
    *,
    num_workers: int = 4,
    attribute: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create official train/validation loaders for the tokenizer lessons.

    Both splits use deterministic image transforms. Only the training loader
    shuffles images and drops its final incomplete batch.
    """
    train_loader = make_aligned_celeba_loader(
        root,
        resolution,
        batch_size,
        device,
        num_workers=num_workers,
        attribute=attribute,
        horizontal_flip=False,
    )
    validation_loader = make_aligned_celeba_loader(
        root,
        resolution,
        batch_size,
        device,
        split="validation",
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        attribute=attribute,
        horizontal_flip=False,
    )
    return train_loader, validation_loader


def make_celeba_training_loader(
    root,
    resolution: int,
    batch_size: int,
    device: torch.device,
    *,
    pipeline: str,
    split: str = "train",
    num_workers: int = 4,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create one CPU/PIL or CUDA/nvJPEG CelebA training loader."""
    if pipeline == "cuda":
        return make_encoded_celeba_loader(
            root,
            batch_size,
            device,
            split=split,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    if pipeline == "cpu":
        return make_aligned_celeba_loader(
            root,
            resolution,
            batch_size,
            device,
            split=split,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    raise ValueError("pipeline must already be resolved to 'cpu' or 'cuda'.")


class CelebATrainingStream:
    """Reuse CelebA loaders and return device-ready training batches."""

    def __init__(
        self,
        root,
        device: torch.device,
        *,
        pipeline: str = "auto",
        split: str = "train",
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive.")
        self.root = Path(root)
        self.device = device
        self.split = split
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pipeline = resolve_celeba_pipeline(
            self.root,
            device,
            requested=pipeline,
        )
        self.dataset_size = len(_aligned_image_paths(self.root, split))
        self._loader: DataLoader | None = None
        self._batches: Iterator[Any] | None = None
        self._loader_key = None

    def next_batch(
        self,
        resolution: int,
        batch_size: int,
        *,
        limit: int | None = None,
    ) -> torch.Tensor:
        """Return the next normalized batch, restarting or rebuilding as needed."""
        if resolution < 1 or batch_size < 1:
            raise ValueError("resolution and batch_size must be positive.")
        loader_key = (
            batch_size,
            None if self.pipeline == "cuda" else resolution,
        )
        if loader_key != self._loader_key:
            self._loader = make_celeba_training_loader(
                self.root,
                resolution,
                batch_size,
                self.device,
                pipeline=self.pipeline,
                split=self.split,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
            )
            self._batches = iter(self._loader)
            self._loader_key = loader_key

        loader = self._loader
        batches = self._batches
        if loader is None or batches is None:
            raise RuntimeError("CelebA loader initialization failed.")
        try:
            raw_images, _ = next(batches)
        except StopIteration:
            batches = iter(loader)
            self._batches = batches
            raw_images, _ = next(batches)

        if self.pipeline == "cuda":
            images = prepare_encoded_celeba_batch(
                raw_images,
                resolution,
                self.device,
            )
        else:
            images = raw_images.to(self.device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        if limit is not None:
            if not 1 <= limit <= len(images):
                raise ValueError("limit must fit within the loaded batch.")
            images = images[:limit]
        return images
