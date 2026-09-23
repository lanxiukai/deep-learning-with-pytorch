"""Glasses classification, reviewed corrections, and the glasses-256 data contract."""

import csv
import filecmp
import json
import shutil
from pathlib import Path

from PIL import Image
from torchvision.datasets import ImageFolder

from dl_utils.data.vision import image_folder_dataset

GLASSES_CLASS_NAMES = ("G", "NoG")


GLASSES_IMAGE_SIZE = 256


def glasses_data_config() -> dict[str, object]:
    """Describe the cache and image contract used for checkpoint validation."""
    return {
        "dataset": "glasses-256",
        "class_names": list(GLASSES_CLASS_NAMES),
        "image_size": GLASSES_IMAGE_SIZE,
        "image_channels": 3,
        "pixel_range": [0.0, 1.0],
        "split": "all images (training set)",
    }


def glasses_dataset(root: Path) -> ImageFolder:
    """Read the prepared RGB cache and validate its class-directory contract."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset cache not found: {root}. Prepare it with "
            "python tool_scripts/download_dataset.py --dataset glasses"
        )
    dataset = image_folder_dataset(root)
    if dataset.classes != list(GLASSES_CLASS_NAMES):
        raise ValueError(
            f"Expected classes {GLASSES_CLASS_NAMES}, got {dataset.classes}"
        )
    return dataset


CORRECTIONS_PATH = Path(__file__).with_name("glasses_label_corrections.json")


EXPECTED_TRAINING_IMAGES = 4500


CORRECTED_CLASS_COUNTS = {"G": 2543, "NoG": 1957}


RAW_IMAGE_DIRS = (
    Path("faces-spring-2020") / "faces-spring-2020",
    Path("faces-spring-2020"),
)


def load_corrections() -> tuple[list[int], list[int]]:
    """Load and validate the static Vision-LLM review results."""
    corrections = json.loads(CORRECTIONS_PATH.read_text(encoding="utf-8"))
    g_to_nog = corrections["G_to_NoG"]
    nog_to_g = corrections["NoG_to_G"]

    if corrections.get("total_reviewed") != 4500:
        raise ValueError("Expected corrections reviewed against 4500 images")
    if len(g_to_nog) != 415 or len(nog_to_g) != 102:
        raise ValueError("Expected 415 G→NoG and 102 NoG→G corrections")
    if len(set(g_to_nog)) != len(g_to_nog):
        raise ValueError("Duplicate image IDs in G_to_NoG")
    if len(set(nog_to_g)) != len(nog_to_g):
        raise ValueError("Duplicate image IDs in NoG_to_G")
    if set(g_to_nog) & set(nog_to_g):
        raise ValueError("The correction directions contain overlapping IDs")
    return g_to_nog, nog_to_g


def _same_image(first: Path, second: Path) -> bool:
    """Return whether two files have identical bytes or decoded pixels."""
    if filecmp.cmp(first, second, shallow=False):
        return True
    with Image.open(first) as first_image, Image.open(second) as second_image:
        return (
            first_image.size == second_image.size
            and first_image.mode == second_image.mode
            and first_image.tobytes() == second_image.tobytes()
        )


def _load_training_labels(data_dir: Path) -> dict[str, str]:
    """Return the CSV-defined destination class for every training image."""
    train_csv = data_dir / "train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training labels not found: {train_csv}")

    labels = {}
    with train_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            filename = f"face-{int(row['id'])}.png"
            label = "G" if int(row["glasses"]) == 1 else "NoG"
            if filename in labels:
                raise ValueError(f"Duplicate training image ID: {filename}")
            labels[filename] = label
    if len(labels) != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES} training labels, "
            f"found {len(labels)} in {train_csv}"
        )
    return labels


def ensure_glasses_classification(data_dir: Path) -> tuple[int, int]:
    """Create or complete G/NoG from raw images; return (copied, existing)."""
    data_dir = Path(data_dir)
    labels = _load_training_labels(data_dir)
    class_dirs = {name: data_dir / name for name in ("G", "NoG")}
    for class_dir in class_dirs.values():
        class_dir.mkdir(parents=True, exist_ok=True)

    present = {
        path.name
        for class_dir in class_dirs.values()
        for path in class_dir.glob("*.png")
    }
    unexpected = sorted(present - labels.keys())
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise ValueError(
            f"{len(unexpected)} unexpected classified images under "
            f"{data_dir}: {preview}"
        )

    missing = labels.keys() - present
    if not missing:
        print(f"  Glasses classification already complete: {len(present)} images.")
        return 0, len(present)

    raw_dir = next(
        (
            data_dir / relative
            for relative in RAW_IMAGE_DIRS
            if (data_dir / relative).is_dir()
        ),
        None,
    )
    if raw_dir is None:
        raise FileNotFoundError(
            f"{len(missing)} classified images are missing and no raw image "
            f"directory was found under {data_dir}"
        )

    missing_sources = sorted(
        filename for filename in missing if not (raw_dir / filename).is_file()
    )
    if missing_sources:
        preview = ", ".join(missing_sources[:5])
        raise FileNotFoundError(
            f"{len(missing_sources)} raw training images are missing from "
            f"{raw_dir}: {preview}"
        )

    for filename in sorted(missing, key=lambda name: int(name[5:-4])):
        shutil.copy2(raw_dir / filename, class_dirs[labels[filename]] / filename)
    print(
        f"  Glasses classification completed: {len(missing)} copied, "
        f"{len(present)} already classified."
    )
    return len(missing), len(present)


def validate_glasses_classification(
    data_dir: Path,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Validate unique G/NoG membership and return the two class counts."""
    data_dir = Path(data_dir)
    files = {
        name: {path.name for path in (data_dir / name).glob("*.png")}
        for name in ("G", "NoG")
    }
    duplicates = sorted(files["G"] & files["NoG"])
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"{len(duplicates)} images occur in both G/ and NoG/ under "
            f"{data_dir}: {preview}"
        )
    total = len(files["G"] | files["NoG"])
    if total != EXPECTED_TRAINING_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TRAINING_IMAGES} classified images under "
            f"{data_dir}, found {total}"
        )

    counts = {name: len(paths) for name, paths in files.items()}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(
            f"Expected corrected class counts {expected_counts}, found "
            f"{counts} under {data_dir}"
        )
    return counts


def apply_glasses_label_corrections(data_dir: Path) -> tuple[int, int]:
    """Move reviewed images to the correct class; return (moved, unchanged)."""
    data_dir = Path(data_dir)
    g_dir = data_dir / "G"
    nog_dir = data_dir / "NoG"
    if not g_dir.is_dir() or not nog_dir.is_dir():
        raise FileNotFoundError(
            f"Expected G/ and NoG/ class directories under {data_dir}"
        )

    g_to_nog, nog_to_g = load_corrections()
    planned = [
        (g_dir / f"face-{image_id}.png", nog_dir / f"face-{image_id}.png")
        for image_id in g_to_nog
    ]
    planned.extend(
        (nog_dir / f"face-{image_id}.png", g_dir / f"face-{image_id}.png")
        for image_id in nog_to_g
    )

    missing = [
        source.name
        for source, target in planned
        if not source.is_file() and not target.is_file()
    ]
    conflicts = [
        source.name
        for source, target in planned
        if source.is_file() and target.is_file() and not _same_image(source, target)
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} reviewed images are missing under {data_dir}: {preview}"
        )
    if conflicts:
        preview = ", ".join(conflicts[:5])
        raise FileExistsError(
            f"{len(conflicts)} corrections have different source and target "
            f"files under {data_dir}: {preview}"
        )

    moved = 0
    unchanged = 0
    for source, target in planned:
        if not source.is_file():
            unchanged += 1
            continue
        if target.is_file():
            source.unlink()
        else:
            source.replace(target)
        moved += 1

    print(f"  Glasses labels corrected: {moved} moved, {unchanged} already correct.")
    validate_glasses_classification(data_dir, CORRECTED_CLASS_COUNTS)
    return moved, unchanged


__all__ = [
    "GLASSES_CLASS_NAMES",
    "GLASSES_IMAGE_SIZE",
    "apply_glasses_label_corrections",
    "ensure_glasses_classification",
    "glasses_data_config",
    "glasses_dataset",
    "load_corrections",
    "validate_glasses_classification",
]
