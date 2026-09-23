"""Official CelebA partitions, optional attributes, and image readers."""

import csv
from functools import lru_cache
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_file

from dl_utils.data.images import load_rgb_image

CELEBA_PARTITIONS = {"train": 0, "validation": 1, "test": 2}


CELEBA_SMILING_ATTRIBUTE = "Smiling"


CELEBA_SMILING_CLASSES = ("Not smiling", "Smiling")


@lru_cache(maxsize=16)
def _aligned_image_paths(root: Path, split: str) -> tuple[Path, ...]:
    """Resolve and validate one split once per process."""
    root = Path(root)
    if split not in CELEBA_PARTITIONS:
        choices = ", ".join(sorted(CELEBA_PARTITIONS))
        raise ValueError(f"split must be one of {{{choices}}}.")

    partition_path = root / "list_eval_partition.csv"
    image_candidates = (
        root / "img_align_celeba" / "img_align_celeba",
        root / "img_align_celeba",
    )
    image_dir = next(
        (candidate for candidate in image_candidates if candidate.is_dir()),
        None,
    )
    missing = [
        path
        for path in (partition_path, image_dir)
        if path is None or not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"aligned CelebA files are incomplete under {root}: {missing}"
        )
    assert image_dir is not None

    partition = CELEBA_PARTITIONS[split]
    with partition_path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        image_paths = tuple(
            image_dir / row["image_id"]
            for row in rows
            if int(row["partition"]) == partition
        )
    if not image_paths:
        raise ValueError(f"CelebA split {split!r} contains no images.")

    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        preview = ", ".join(str(path) for path in missing_images[:3])
        raise FileNotFoundError(
            f"CelebA split {split!r} is missing {len(missing_images)} "
            f"images; first missing paths: {preview}"
        )
    return image_paths


class CelebAAlignedDataset(Dataset):
    """Read one official split from the locally prepared aligned CelebA."""

    def __init__(self, root, split="train", transform=None, *, attribute=None):
        super().__init__()
        self.image_paths = _aligned_image_paths(Path(root), split)
        self.transform = transform
        if attribute is None:
            self.targets = [0] * len(self.image_paths)
        else:
            with (Path(root) / "list_attr_celeba.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                labels = {
                    row["image_id"]: int(int(row[attribute]) > 0)
                    for row in csv.DictReader(stream)
                }
            self.targets = [labels[path.name] for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = load_rgb_image(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


class CelebAEncodedDataset(Dataset):
    """Read encoded aligned JPEG bytes for batched CUDA decoding."""

    def __init__(self, root, split="train"):
        super().__init__()
        self.image_paths = _aligned_image_paths(Path(root), split)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return read_file(str(self.image_paths[index]))
