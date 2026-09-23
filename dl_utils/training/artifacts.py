"""Compose metric records, inference, and plots into training-run artifacts."""

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot.curves import save_curve, save_loss_panels
from dl_utils.plot.images import save_image_row_grid
from dl_utils.training.history import (
    MetricHistory,
    as_list,
    has_any_finite,
    save_metrics_csv,
)


def save_training_metrics(
    history: list[dict[str, float]],
    out_dir: Path,
    *,
    prefix: str,
    max_panels: int,
) -> None:
    """Keep metric curves and their numeric values together in the run root."""
    metrics = {name: [row[name] for row in history] for name in history[0]}
    epochs = list(range(1, len(history) + 1))
    save_metrics_csv(
        {"epoch": epochs, **metrics},
        out_dir / f"{prefix}_metrics.csv",
    )
    names = list(metrics)
    for start in range(0, len(names), max_panels):
        panels = {
            name.replace("_", " ").capitalize(): {name: metrics[name]}
            for name in names[start : start + max_panels]
        }
        page = start // max_panels + 1
        save_loss_panels(
            epochs,
            panels,
            out_dir / f"{prefix}_metrics_{page:02d}.png",
            xlabel="Epoch",
            ylabel="Value",
        )


def maybe_save_curve(
    steps: Sequence[float],
    metrics: MetricHistory,
    series: Mapping[str, str],
    path: str | PathLike[str],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    verbose: bool = False,
) -> None:
    """
    Plot curve(s) only when the needed metric keys exist and contain valid data.
    Automatically truncates to a common length to avoid length-mismatch errors.
    """
    curves: dict[str, list[float]] = {}
    lengths: list[int] = []
    for label, key in series.items():
        if key not in metrics:
            continue
        metric_values = as_list(metrics.get(key))
        if not metric_values:
            continue
        if not has_any_finite(metric_values):
            continue
        curves[label] = [float(value) for value in metric_values]
        lengths.append(len(metric_values))

    if not curves:
        return

    common_length = min([len(steps)] + lengths) if lengths else len(steps)
    if common_length <= 0:
        return

    x_use = steps[:common_length]
    curves_use = {key: values[:common_length] for key, values in curves.items()}
    try:
        save_curve(
            x_use, curves_use, path=path, xlabel=xlabel, ylabel=ylabel, title=title
        )
    except Exception as err:  # noqa: BLE001 - Optional EBM plots must not abort training.
        if verbose:
            print(f"[genai] skip plot {path!r}: {err}")


def save_training_samples(
    generator,
    noise,
    labels,
    output_path,
    *,
    class_names,
    title,
    dpi=200,
    shared_latents_across_classes=False,
    inference_batch_size: int | None = None,
    class_indices=None,
) -> None:
    """Generate and save fixed class-conditional samples grouped by class."""
    class_names = tuple(class_names)
    if class_indices is None:
        class_indices = tuple(range(len(class_names)))
    else:
        class_indices = tuple(class_indices)
    if len(class_indices) != len(class_names):
        raise ValueError("class_indices and class_names must have equal length.")
    sample_batch_size = (
        len(noise) if inference_batch_size is None else inference_batch_size
    )
    samples = generate_in_batches(
        (noise, labels),
        sample_batch_size,
        lambda noise_batch, label_batch: generator(
            noise_batch,
            label_batch,
        ).float(),
        module=generator,
    )

    cpu_labels = labels.cpu()
    image_rows = [samples[cpu_labels == class_index] for class_index in class_indices]
    save_image_row_grid(
        image_rows,
        [name.title() for name in class_names],
        output_path,
        title=title,
        column_labels=[
            (
                f"Shared z {index + 1}"
                if shared_latents_across_classes
                else f"Sample {index + 1}"
            )
            for index in range(len(image_rows[0]) if image_rows else 0)
        ],
        dpi=dpi,
    )
