"""Static curve and panel rendering from prepared numeric series."""

import csv
import os
from collections.abc import Mapping, Sequence
from os import PathLike

from dl_utils.plot._backend import pyplot as _plt


def save_curve(
    steps: Sequence[float],
    curves: Mapping[str, Sequence[float]],
    path: str | PathLike[str],
    *,
    xlabel: str = "Step",
    ylabel: str = "Value",
    title: str | None = None,
    csv_path: str | PathLike[str] | None = None,
) -> None:
    """
    Plot one or more curves and optionally save the raw values to CSV.

    Args:
        steps: x-axis values (e.g., steps or epochs).
        curves: mapping from curve name to y values.
        path: output image path.
        xlabel, ylabel, title: labeling for the figure.
        csv_path: optional path to save the underlying data.
    """
    if not curves:
        raise ValueError("save_curve: curves is empty.")

    x_list = list(steps)
    step_count = len(x_list)

    # Normalize inputs to lists of floats and validate lengths
    norm_curves: dict[str, list[float]] = {}
    for name, curve_values in curves.items():
        y_list = list(map(float, curve_values))
        if len(y_list) != step_count:
            raise ValueError(
                f"save_curve: length mismatch for '{name}', expected {step_count} got {len(y_list)}."
            )
        norm_curves[name] = y_list

    path_str = os.fspath(path)
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    _plt.figure(figsize=(8, 5))
    for name, curve_values in norm_curves.items():
        _plt.plot(x_list, curve_values, label=name)
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    if title:
        _plt.title(title)
    if len(norm_curves) > 1:
        _plt.legend()
    _plt.grid(True, alpha=0.3)
    _plt.tight_layout()
    _plt.savefig(path_str, dpi=200)
    _plt.close()

    if csv_path:
        csv_path_str = os.fspath(csv_path)
        os.makedirs(os.path.dirname(csv_path_str), exist_ok=True)
        header = [xlabel] + list(norm_curves.keys())
        rows = zip(x_list, *norm_curves.values())
        with open(csv_path_str, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)


def save_loss_panels(
    steps: Sequence[float],
    panels: Mapping[str, Mapping[str, Sequence[float]]],
    path: str | PathLike[str],
    *,
    xlabel: str = "epoch",
    ylabel: str = "loss",
) -> None:
    """Save related loss groups on vertically stacked independent-y panels.

    ``panels`` maps each subplot title to one or more labelled curves. Every
    curve must have the same number of values as ``steps``.
    """
    x_values = list(map(float, steps))
    if not x_values:
        raise ValueError("save_loss_panels: steps is empty.")
    if not panels:
        raise ValueError("save_loss_panels: panels are empty.")

    normalized_panels = {}
    for title, curves in panels.items():
        if not curves:
            raise ValueError(f"save_loss_panels: panel '{title}' has no curves.")
        normalized_curves = {}
        for label, values in curves.items():
            normalized_values = list(map(float, values))
            if len(normalized_values) != len(x_values):
                raise ValueError(
                    f"save_loss_panels: curve '{label}' in panel "
                    f"'{title}' does not match steps."
                )
            normalized_curves[label] = normalized_values
        normalized_panels[title] = normalized_curves

    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    num_panels = len(normalized_panels)
    fig, axes = _plt.subplots(
        num_panels,
        1,
        figsize=(7, 3 * num_panels),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    axes = axes[:, 0]
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    line_styles = ("-", "--", "-.", ":")

    for axis, (title, curves) in zip(axes, normalized_panels.items()):
        for index, (label, values) in enumerate(curves.items()):
            axis.plot(
                x_values,
                values,
                color=colors[index % len(colors)],
                linestyle=line_styles[index % len(line_styles)],
                label=label,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path_str, dpi=300)
    _plt.close(fig)
