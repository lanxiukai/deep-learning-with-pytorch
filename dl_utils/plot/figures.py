"""Generic figure and curve-rendering primitives."""

import csv
import os
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from dl_utils.plot._backend import pyplot as _plt

import matplotlib_inline.backend_inline as backend_inline
from IPython import get_ipython

from dl_utils.training.metrics import MetricHistory, as_list, has_any_finite


def use_svg_display():
    """Use SVG display in Jupyter (no-op outside IPython)."""
    if get_ipython() is not None:
        backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size in Matplotlib."""
    use_svg_display()
    _plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes in Matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(horizontal_values, vertical_values=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot the data in Matplotlib."""
    if legend is None:
        legend = []
    
    set_figsize(figsize)
    axes = axes if axes else _plt.gca()
    
    def has_one_axis(values):
        return (hasattr(values, "ndim") and values.ndim == 1 or isinstance(values, list)
                and not hasattr(values[0], "__len__"))
        
    x_values = [horizontal_values] if has_one_axis(horizontal_values) else horizontal_values
    y_values = x_values if vertical_values is None else ([vertical_values] if has_one_axis(vertical_values) else vertical_values)
    if vertical_values is None:
        x_values = [[]] * len(x_values)
    if len(x_values) != len(y_values):
        x_values = x_values * len(y_values)
    axes.cla()
    for horizontal_series, vertical_series, fmt in zip(x_values, y_values, fmts):
        if len(horizontal_series):
            axes.plot(horizontal_series, vertical_series, fmt)
        else:
            axes.plot(vertical_series, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def annotate(text, xy, xytext):
    _plt.gca().annotate(text, xy=xy, xytext=xytext,
                        arrowprops=dict(arrowstyle='->'))


class Animator:
    """Animate data curves using Matplotlib interactive mode."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = _plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.horizontal_data, self.vertical_data, self.fmts = None, None, fmts
        if _plt.get_backend().lower() != "agg":
            _plt.ion()
            self.fig.show()
        self._closed = False

    def add(self, horizontal_values, vertical_values):
        """Add the data to the animator."""
        if not hasattr(vertical_values, "__len__"):
            vertical_values = [vertical_values]
        series_count = len(vertical_values)
        if not hasattr(horizontal_values, "__len__"):
            horizontal_values = [horizontal_values] * series_count
        x_data = self.horizontal_data
        if not x_data:
            x_data = [[] for _ in range(series_count)]
            self.horizontal_data = x_data
        y_data = self.vertical_data
        if not y_data:
            y_data = [[] for _ in range(series_count)]
            self.vertical_data = y_data
        for i, (horizontal_value, vertical_value) in enumerate(zip(horizontal_values, vertical_values)):
            if horizontal_value is not None and vertical_value is not None:
                x_data[i].append(horizontal_value)
                y_data[i].append(vertical_value)
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(x_data, y_data, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        self.fig.canvas.draw_idle()
        if _plt.get_backend().lower() != "agg":
            self.fig.canvas.flush_events()
            _plt.pause(0.001)


def heatmap(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
            cmap='Reds'):
    """
    Show heatmaps of matrices.
    Args:
        matrices: (number of rows for display, number of columns for display, number of queries, number of keys)
        xlabel: x-axis label
        ylabel: y-axis label
        titles: titles for each subplot
        figsize: figure size
        cmap: color map
    """
    use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = _plt.subplots(num_rows, num_cols, figsize=figsize,
                              sharex=True, sharey=True, squeeze=False)
    pcm = None
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for column_index, (axis, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = axis.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                axis.set_xlabel(xlabel)
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if titles:
                axis.set_title(titles[column_index])
    if pcm is not None:
        fig.colorbar(pcm, ax=axes, shrink=0.6)


def trace2d(objective, results):
    """Show the trace of 2D variables during optimization"""
    import torch

    set_figsize()
    _plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1), indexing='ij')
    _plt.contour(x1, x2, objective(x1, x2), colors='#1f77b4')
    _plt.xlabel('x1')
    _plt.ylabel('x2')


def seq_len_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot a histogram of sequence length pairs."""
    set_figsize()
    _, _, patches = _plt.hist(
        [[len(sequence) for sequence in xlist], [len(sequence) for sequence in ylist]])
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    patch_groups: Any = patches
    for patch in patch_groups[1].patches:
        patch.set_hatch('/')
    _plt.legend(legend)


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


def save_loss_curves(
    steps: Sequence[float],
    discriminator_losses: Sequence[float],
    generator_losses: Sequence[float],
    generator_adversarial_losses: Mapping[str, Sequence[float]],
    generator_reconstruction_losses: Mapping[str, Sequence[float]],
    path: str | PathLike[str],
    *,
    xlabel: str = "epoch",
) -> None:
    """Save total and component losses on four independent-y subplots.

    The component mappings supply the plotted curves and their legend labels;
    every series must align with ``steps``.
    """
    x_values = list(steps)
    discriminator_values = list(map(float, discriminator_losses))
    generator_values = list(map(float, generator_losses))
    if len(discriminator_values) != len(x_values):
        raise ValueError(
            "save_loss_curves: discriminator loss length does not match steps."
        )
    if len(generator_values) != len(x_values):
        raise ValueError(
            "save_loss_curves: generator loss length does not match steps."
        )

    def normalize_components(
        name: str,
        curves: Mapping[str, Sequence[float]],
    ) -> dict[str, list[float]]:
        if not curves:
            raise ValueError(f"save_loss_curves: {name} losses are empty.")
        normalized = {}
        for label, values in curves.items():
            normalized_values = list(map(float, values))
            if len(normalized_values) != len(x_values):
                raise ValueError(
                    f"save_loss_curves: {name} loss '{label}' length "
                    "does not match steps."
                )
            normalized[label] = normalized_values
        return normalized

    adversarial_values = normalize_components(
        "generator adversarial",
        generator_adversarial_losses,
    )
    reconstruction_values = normalize_components(
        "generator reconstruction",
        generator_reconstruction_losses,
    )

    path_str = os.fspath(path)
    parent = os.path.dirname(path_str)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig, axes = _plt.subplots(
        4,
        1,
        figsize=(7, 12),
        sharex=True,
        sharey=False,
    )
    axes[0].plot(x_values, discriminator_values, color="tab:blue")
    axes[0].set_title("Total D loss")
    axes[1].plot(x_values, generator_values, color="tab:orange")
    axes[1].set_title("Total G loss")

    component_groups = (
        (axes[2], "G adversarial loss", adversarial_values),
        (axes[3], "G reconstruction loss", reconstruction_values),
    )
    colors = ("tab:blue", "tab:orange")
    line_styles = ("-", "--")
    for axis, title, curves in component_groups:
        for index, (label, values) in enumerate(curves.items()):
            axis.plot(
                x_values,
                values,
                color=colors[index % len(colors)],
                linestyle=line_styles[index % len(line_styles)],
                label=label,
            )
        axis.set_title(title)
        axis.legend()

    for axis in axes:
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path_str, dpi=300)
    _plt.close(fig)


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
            raise ValueError(
                f"save_loss_panels: panel '{title}' has no curves."
            )
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
        save_curve(x_use, curves_use, path=path, xlabel=xlabel, ylabel=ylabel, title=title)
    except Exception as err:
        if verbose:
            print(f"[genai] skip plot {path!r}: {err}")
