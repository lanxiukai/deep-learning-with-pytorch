"""Basic figures and live textbook plots; static exports live in curves."""

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

from IPython import get_ipython
from matplotlib_inline import backend_inline

from dl_utils.plot._backend import pyplot as _plt
from dl_utils.plot.curves import save_curve


def use_svg_display():
    """Use SVG display in Jupyter (no-op outside IPython)."""
    if get_ipython() is not None:
        backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size in Matplotlib."""
    use_svg_display()
    _plt.rcParams["figure.figsize"] = figsize


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


def plot(
    horizontal_values,
    vertical_values=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """Plot the data in Matplotlib."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else _plt.gca()

    def has_one_axis(values):
        return (
            hasattr(values, "ndim")
            and values.ndim == 1
            or isinstance(values, list)
            and not hasattr(values[0], "__len__")
        )

    x_values = (
        [horizontal_values] if has_one_axis(horizontal_values) else horizontal_values
    )
    y_values = (
        x_values
        if vertical_values is None
        else ([vertical_values] if has_one_axis(vertical_values) else vertical_values)
    )
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
    _plt.gca().annotate(text, xy=xy, xytext=xytext, arrowprops={"arrowstyle": "->"})


class Animator:
    """Animate data curves using Matplotlib interactive mode."""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        if legend is None:
            legend = []
        self.fig, self.axes = _plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
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
        for i, (horizontal_value, vertical_value) in enumerate(
            zip(horizontal_values, vertical_values)
        ):
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


def heatmap(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap="Reds"):
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
    fig, axes = _plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
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
    _plt.plot(*zip(*results), "-o", color="#ff7f0e")
    x1, x2 = torch.meshgrid(
        torch.arange(-5.5, 1.0, 0.1), torch.arange(-3.0, 1.0, 0.1), indexing="ij"
    )
    _plt.contour(x1, x2, objective(x1, x2), colors="#1f77b4")
    _plt.xlabel("x1")
    _plt.ylabel("x2")


def seq_len_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot a histogram of sequence length pairs."""
    set_figsize()
    _, _, patches = _plt.hist(
        [[len(sequence) for sequence in xlist], [len(sequence) for sequence in ylist]]
    )
    _plt.xlabel(xlabel)
    _plt.ylabel(ylabel)
    patch_groups: Any = patches
    for patch in patch_groups[1].patches:
        patch.set_hatch("/")
    _plt.legend(legend)


def maybe_save_curve(
    steps: Sequence[float],
    metrics: Mapping[str, Sequence[int | float]],
    series: Mapping[str, str],
    path: str | PathLike[str],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    verbose: bool = False,
) -> None:
    """Preserve the EBM plotting entry point; new callers use training.artifacts."""
    from dl_utils.training.artifacts import maybe_save_curve as save_metric_curve

    save_metric_curve(
        steps,
        metrics,
        series,
        path,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        verbose=verbose,
    )


__all__ = [
    "Animator",
    "annotate",
    "heatmap",
    "maybe_save_curve",
    "plot",
    "save_curve",
    "seq_len_hist",
    "set_axes",
    "set_figsize",
    "trace2d",
    "use_svg_display",
]
