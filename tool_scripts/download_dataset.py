"""Download and prepare all, one, or several lesson datasets in sequence."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence

from dl_utils.d2l.time_machine import read_time_machine
from dl_utils.data.datasets.celeba.preparation import (
    celeba_dataset_is_ready,
    prepare_celeba_cyclegan_splits,
)
from dl_utils.data.datasets.glasses import (
    apply_glasses_label_corrections,
    ensure_glasses_classification,
)
from dl_utils.data.datasets.vision import vision_loaders
from dl_utils.data.downloads import download, download_extract, download_kaggle_dataset
from dl_utils.data.preparation import build_image_folder_cache
from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data"


def _download_mnist() -> None:
    vision_loaders(
        dataset="mnist",
        data_dir=DATA_DIR / "mnist",
        batch_size=256,
    )
    print("The MNIST Dataset has been downloaded.")


def _download_fashion_mnist() -> None:
    vision_loaders(
        dataset="fashion_mnist",
        data_dir=DATA_DIR / "fashion_mnist",
        batch_size=256,
    )
    print("The Fashion-MNIST Dataset has been downloaded.")


def _download_house_prices() -> None:
    import pandas as pd

    pd.read_csv(download("kaggle_house_train", data_root=DATA_DIR))
    pd.read_csv(download("kaggle_house_test", data_root=DATA_DIR))
    print("The Kaggle House Price Dataset has been downloaded.")


def _download_time_machine() -> None:
    lines = read_time_machine(data_root=DATA_DIR)
    print(f"The Time Machine Dataset has been downloaded: {lines[0]}")


def _download_celeba() -> None:
    celeba_dir = DATA_DIR / "celeba"
    downloaded = download_kaggle_dataset(
        "jessicali9530/celeba-dataset",
        celeba_dir,
        is_ready=celeba_dataset_is_ready,
    )
    if not downloaded:
        raise RuntimeError("kagglehub is required to download CelebA")
    if not prepare_celeba_cyclegan_splits(celeba_dir):
        raise RuntimeError("CelebA CycleGAN split preparation failed")
    print(f"The CelebA Dataset is ready: {celeba_dir}")


def _download_anime_face() -> None:
    downloaded = download_kaggle_dataset(
        "splcher/animefacedataset",
        DATA_DIR / "anime_face",
    )
    if not downloaded:
        raise RuntimeError("kagglehub is required to download Anime Face")
    print("The Anime Face Dataset has been downloaded.")


def _download_glasses() -> None:
    out_root = DATA_DIR / "glasses"
    downloaded = download_kaggle_dataset(
        "jeffheaton/glasses-or-no-glasses",
        out_root,
    )
    if not downloaded:
        raise RuntimeError("kagglehub is required to download Glasses")

    ensure_glasses_classification(out_root)
    apply_glasses_label_corrections(out_root)
    resized_root = DATA_DIR / "glasses-256"
    resized_root.mkdir(parents=True, exist_ok=True)
    unexpected_classes = sorted(
        path.name
        for path in resized_root.iterdir()
        if path.is_dir() and path.name not in {"G", "NoG"}
    )
    if unexpected_classes:
        raise RuntimeError(
            f"Unexpected class directories in {resized_root}: {unexpected_classes}"
        )
    workers = min(16, os.cpu_count() or 1)
    for class_name in ("G", "NoG"):
        build_image_folder_cache(
            out_root / class_name,
            resized_root / class_name,
            size=256,
            workers=workers,
        )
    apply_glasses_label_corrections(resized_root)
    print(f"The Glasses training dataset is ready: {resized_root}")


def _download_airfoil() -> None:
    import numpy as np

    data = np.genfromtxt(
        download("airfoil", data_root=DATA_DIR),
        dtype=np.float32,
        delimiter="\t",
    )
    print(f"The Airfoil Self-Noise Dataset has been downloaded: {data.shape}")


def _download_fra_eng() -> None:
    data_dir = download_extract("fra-eng", data_root=DATA_DIR)
    print(f"The English-French Dataset has been downloaded: {data_dir}")


def _download_pokemon() -> None:
    data_dir = download_extract("pokemon", data_root=DATA_DIR)
    print(f"The Pokemon Dataset has been downloaded: {data_dir}")


DOWNLOADERS: dict[str, Callable[[], None]] = {
    "mnist": _download_mnist,
    "fashion-mnist": _download_fashion_mnist,
    "house-prices": _download_house_prices,
    "time-machine": _download_time_machine,
    "celeba": _download_celeba,
    "anime-face": _download_anime_face,
    "glasses": _download_glasses,
    "airfoil": _download_airfoil,
    "fra-eng": _download_fra_eng,
    "pokemon": _download_pokemon,
}
DATASET_ORDER = tuple(DOWNLOADERS)


def _resolve_dataset_names(selected: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve CLI selections while preserving their first-seen order."""
    if selected is None:
        return DATASET_ORDER
    names = tuple(selected)
    if names == ("all",):
        return DATASET_ORDER
    if "all" in names:
        raise ValueError("'all' cannot be combined with dataset names")
    return tuple(dict.fromkeys(names))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="extend",
        nargs="+",
        choices=("all", *DATASET_ORDER),
        metavar="NAME",
        help=(
            "one or more datasets to prepare in the given order; may be "
            "repeated (default: all datasets in the documented order). "
            f"Available names: {', '.join(DATASET_ORDER)}"
        ),
    )
    args = parser.parse_args(argv)
    try:
        args.datasets = _resolve_dataset_names(args.datasets)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(f"Preparation order: {', '.join(args.datasets)}")
    failures: list[str] = []
    for index, name in enumerate(args.datasets, start=1):
        print(f"[{index}/{len(args.datasets)}] Preparing {name}...")
        try:
            DOWNLOADERS[name]()
        # Providers raise heterogeneous exceptions; a sequence should report each.
        except Exception as error:  # noqa: BLE001
            failures.append(name)
            print(f"Failed to prepare {name}: {error}")

    if failures:
        raise SystemExit(f"Dataset preparation failed: {', '.join(failures)}")
    print(f"Prepared {len(args.datasets)} dataset(s) successfully.")


if __name__ == "__main__":
    main()
