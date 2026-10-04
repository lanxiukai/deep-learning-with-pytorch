"""One Food-101 manifest for every lesson: 700/50/250 images per class."""

import json
import random
from pathlib import Path

import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 101
SPLIT_SEED = 42


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
        manifest = json.loads((self.root / "diffusion_manifest.json").read_text())
        self.classes = manifest["classes"]
        if manifest["dataset"] != "Food-101" or len(self.classes) != NUM_CLASSES:
            raise ValueError("Prepare the shared Food-101 manifest first.")
        self.records = manifest["splits"][split]
        if limit is not None:
            groups = [
                [r for r in self.records if r[1] == c] for c in range(NUM_CLASSES)
            ]
            self.records = [r for row in zip(*groups) for r in row][:limit]
        operations = [
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.CenterCrop(image_size),
        ]
        if augment:
            operations.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(
            operations + [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        sample, label = self.records[index]
        with Image.open(self.root / "food-101/images" / sample) as image:
            return self.transform(image.convert("RGB")), label


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
    manifest = json.loads((Path(args.data_dir) / "diffusion_manifest.json").read_text())
    return {
        "dataset": "Food-101",
        "classes": manifest["classes"],
        "split_seed": manifest["split_seed"],
        "image_size": args.image_size,
        "preprocessing": "bicubic short-edge resize; center crop; train horizontal flip",
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
