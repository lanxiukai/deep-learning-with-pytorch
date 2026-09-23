"""Shared data, EMA checkpoints, and records; optimization stays in each lesson."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dl_utils.data.datasets.celeba import CelebAAlignedDataset, aligned_celeba_transform
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "celeba"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "diffusion"


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--image-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)


def add_training_arguments(parser: argparse.ArgumentParser, name: str) -> None:
    add_data_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / name)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=(64, 128, 256, 384)
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate EMA every N epochs and at the end; 0 disables it.",
    )
    add_quality_arguments(parser)


def add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=20260909)


def make_image_loader(args, device, *, split="train", augment=None, shuffle=None):
    """Full official CelebA split, 178px center crop, RGB in [-1, 1]."""
    training = split == "train"
    dataset = CelebAAlignedDataset(
        args.data_dir,
        split=split,
        transform=aligned_celeba_transform(
            args.image_size, horizontal_flip=training if augment is None else augment
        ),
        attribute="Smiling" if getattr(args, "conditional", False) else None,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training if shuffle is None else shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


def prepare_output(args) -> None:
    # A resumed run keeps its checkpoints and metric history.
    if args.resume_from is None or not args.output_dir.exists():
        reset_dir(str(args.output_dir))


def training_metadata(args) -> dict:
    return {
        "dataset": "CelebA",
        "image_size": args.image_size,
        "data_range": [-1.0, 1.0],
        "split": "train",
        "preprocessing": "178px center crop, bilinear resize, random horizontal flip",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "ema_decay": args.ema_decay,
    }


def save_checkpoint(path, model, averaged, optimizer, epoch, **metadata) -> None:
    torch.save(
        {
            "format_version": 3,
            **metadata,
            "epoch": epoch,
            "model_config": model.config(),
            "model_state": model.state_dict(),
            "ema_state": averaged.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
        path,
    )


def restore_checkpoint(path, model, averaged, optimizer, **expected) -> int:
    if path is None:
        return 1
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in {"model_config": model.config(), **expected}.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Resume checkpoint has a different {key}.")
    model.load_state_dict(checkpoint["model_state"])
    averaged.load_state_dict(checkpoint["ema_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return int(checkpoint["epoch"]) + 1


def append_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


class BinnedLoss:
    """Mean per-image error overall and in three declared coordinate bins."""

    def __init__(self, coordinate_name="noise"):
        self.coordinate_name = coordinate_name
        self.sums = torch.zeros(3, dtype=torch.float64)
        self.counts = torch.zeros(3, dtype=torch.long)

    def update(self, per_image, coordinate):
        bins = (coordinate.detach().cpu() * 3).long().clamp(0, 2)
        self.sums.scatter_add_(0, bins, per_image.detach().cpu().double())
        self.counts.scatter_add_(0, bins, torch.ones_like(bins))

    def result(self):
        return {
            "loss": float(self.sums.sum() / self.counts.sum()),
            **{
                f"{self.coordinate_name}_bin_{i}_loss": float(
                    self.sums[i] / self.counts[i]
                )
                for i in range(3)
                if self.counts[i] > 0
            },
        }
