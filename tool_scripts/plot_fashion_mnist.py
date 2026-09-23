"""
Draw one sample per label (0-9) from the Fashion-MNIST training set and plot
them as grayscale images; also plot the binarized versions (enabled by default).
Saves both as a 5x2 grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional
from dl_utils.filesystem.project_root import infer_project_root

_PROJECT_ROOT = infer_project_root()

from dl_utils.plot._backend import pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import torchvision.transforms as T  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one Fashion-MNIST sample per label")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "fashion_mnist"),
        help="Directory to download/cache the dataset (default: <project_root>/data/fashion_mnist)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=64,
        help="Resize images to a square of this size (default: 64).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(_PROJECT_ROOT / "output" / "fmnist_labels.png"),
        help="Output path for the grayscale grid",
    )
    parser.add_argument(
        "--plot-binary",
        action="store_true",
        default=True,
        help="Also save a binarized version (same 5x2 layout). Default: enabled.",
    )
    parser.add_argument(
        "--binary-out",
        type=str,
        default=str(_PROJECT_ROOT / "output" / "fmnist_labels_binary.png"),
        help="Output path for the binarized grid (requires --plot-binary)",
    )
    return parser.parse_args()


def build_transform(resize: Optional[int]) -> T.Compose:
    transforms: list = [T.ToTensor()]
    if resize is not None:
        # Resize applies on PIL image before converting to Tensor.
        transforms.insert(0, T.Resize((resize, resize)))
    return T.Compose(transforms)


def collect_one_per_label(
    dataset: torch.utils.data.Dataset,
    num_classes: int = 10,
) -> list[torch.Tensor]:
    """
    Iterate the dataset and collect the first image for each label.
    Returns a list ordered by label 0..num_classes-1.
    """
    samples: list[Optional[torch.Tensor]] = [None] * num_classes
    remaining = set(range(num_classes))
    for img, label in dataset:
        if label in remaining:
            samples[label] = img
            remaining.remove(label)
            if not remaining:
                break
    if remaining:
        raise RuntimeError(f"Could not find samples for labels: {sorted(remaining)}")
    return [s for s in samples if s is not None]


def plot_grid(
    images: Iterable[torch.Tensor],
    labels: Iterable[int],
    label_names: Optional[Iterable[str]] = None,
    *,
    title: str,
    out_path: str,
) -> None:
    """
    Arrange 10 single-channel images as a 2x5 grid and save.
    """
    imgs = list(images)
    labs = list(labels)
    names = list(label_names) if label_names is not None else None
    if len(imgs) != 10 or len(labs) != 10:
        raise ValueError("Exactly 10 images and 10 labels are required.")
    if names is not None and len(names) != 10:
        raise ValueError("label_names must have length 10 if provided.")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    for i, (img, lab) in enumerate(zip(imgs, labs)):
        ax = axes[i]
        ax.imshow(img.squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        label_text = names[lab] if names is not None else f"Label {lab}"
        ax.set_title(label_text)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_file}")


def main() -> None:
    args = parse_args()
    transform = build_transform(args.resize)

    train_set = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    samples = collect_one_per_label(train_set, num_classes=10)
    labels = list(range(10))
    label_names = list(getattr(train_set, "classes", [])) or None

    plot_grid(
        images=samples,
        labels=labels,
        label_names=label_names,
        title="Fashion-MNIST (gray)",
        out_path=args.out,
    )

    if args.plot_binary:
        # Stochastic binarization: Bernoulli sampling with pixel values as probabilities.
        binary_imgs = [torch.bernoulli(img) for img in samples]
        plot_grid(
            images=binary_imgs,
            labels=labels,
            label_names=label_names,
            title="Fashion-MNIST (binary, stochastic)",
            out_path=args.binary_out,
        )


if __name__ == "__main__":
    main()
