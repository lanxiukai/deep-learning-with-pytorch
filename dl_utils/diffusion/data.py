"""Shared prepared Food-101 images: 700/50/250 images per class."""

import json
import random
from pathlib import Path

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NUM_CLASSES = 101
SPLIT_SEED = 42
CACHE_SPEC = {
    "version": 1,
    "image_size": 256,
    "mode": "RGB",
    "format": "PNG",
    "geometry": "bicubic short-edge resize; center crop",
}


def prepared_manifest(root):
    path = Path(root) / "diffusion_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared Food-101 manifest not found: {path}. Run "
            "tool_scripts/download_dataset.py --dataset food101 first."
        )
    manifest = json.loads(path.read_text())
    if (
        manifest.get("dataset") != "Food-101"
        or len(manifest.get("classes", [])) != NUM_CLASSES
        or manifest.get("cache") != CACHE_SPEC
    ):
        raise ValueError(
            "Expected the prepared Food-101 256px PNG cache. Run "
            "tool_scripts/download_dataset.py --dataset food101 and point "
            "--data-dir to data/food101-256, not the original archive directory."
        )
    return manifest


def prepare_food101(root, *, download=False):
    root = Path(root)
    if download:
        from torchvision.datasets import Food101

        Food101(root, split="train", download=True)
    source = root / "food-101"
    train = json.loads((source / "meta/train.json").read_text())
    test = json.loads((source / "meta/test.json").read_text())
    classes = sorted(train)
    if len(classes) != NUM_CLASSES or classes != sorted(test):
        raise ValueError("Expected the official 101 Food-101 classes.")
    splits = {name: [] for name in ("train", "validation", "test")}
    rng = random.Random(SPLIT_SEED)
    for label, name in enumerate(classes):
        ids = sorted(train[name])
        if len(ids) != 750 or len(test[name]) != 250:
            raise ValueError(f"Incomplete official split for {name}.")
        all_ids = ids + test[name]
        if len(set(all_ids)) != 1000 or any(
            Path(item).parts != (name, Path(item).name) for item in all_ids
        ):
            raise ValueError(f"Invalid or duplicate official image IDs for {name}.")
        rng.shuffle(ids)
        for split, selected in (
            ("validation", ids[:50]),
            ("train", ids[50:]),
            ("test", sorted(test[name])),
        ):
            splits[split].extend([[f"{item}.jpg", label] for item in selected])
    manifest = {
        "dataset": "Food-101",
        "split_seed": SPLIT_SEED,
        "classes": classes,
        "splits": splits,
    }
    destination = root / "diffusion_manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError("Existing manifest differs; choose a separate data directory.")
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


class FoodImages(Dataset):
    def __init__(self, root, split="train", image_size=256, augment=False, limit=None):
        self.root = Path(root)
        if image_size not in (128, 256):
            raise ValueError("Expected 256px images or derived 128px observations.")
        self.image_size = image_size
        manifest = prepared_manifest(self.root)
        self.classes = manifest["classes"]
        self.records = manifest["splits"][split]
        if limit is not None:
            groups = [
                [r for r in self.records if r[1] == c] for c in range(NUM_CLASSES)
            ]
            self.records = [r for row in zip(*groups) for r in row][:limit]
        operations = []
        if augment:
            operations.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(
            operations + [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample, label = self.records[index]
        path = self.root / "images" / Path(sample).with_suffix(".png")
        with Image.open(path) as image:
            if image.size != (256, 256) or image.mode != "RGB":
                raise ValueError(f"Expected a prepared 256x256 RGB image: {path}")
            tensor = self.transform(image)
        if self.image_size == 128:
            tensor = low_resolution(tensor.unsqueeze(0)).squeeze(0)
        return tensor, label


def make_loader(args, split="train", *, augment=None, shuffle=None, limit=None):
    training = split == "train"
    dataset = FoodImages(
        args.data_dir,
        split,
        args.image_size,
        training if augment is None else augment,
        limit,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training if shuffle is None else shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )


def data_config(args):
    manifest = prepared_manifest(args.data_dir)
    return {
        "dataset": "Food-101",
        "classes": manifest["classes"],
        "split_seed": manifest["split_seed"],
        "image_size": args.image_size,
        "preprocessing": {
            "cache": dict(manifest["cache"]),
            "training_augmentation": "horizontal flip",
            "input_128px": "antialiased bicubic tensor downsample from the 256px cache",
        },
        "range": [-1, 1],
        "split_sizes": {k: len(v) for k, v in manifest["splits"].items()},
    }


def low_resolution(images):
    return F.interpolate(
        images, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True
    )


def upsample(low, size):
    return F.interpolate(
        low, size=size, mode="bicubic", align_corners=False, antialias=True
    )
