"""Metric accumulators, with legacy D2L and EBM import entry points."""

from collections.abc import Sequence

import torch

from .history import (
    MetricHistory,
    NumericScalar,
    align_and_drop_all_nan_rows,
    align_metrics_for_csv,
    as_list,
    has_any_finite,
    save_metrics_csv,
)


class MetricAccumulator:
    """Accumulate named scalar tensors with example-count weighting."""

    def __init__(self, names: Sequence[str], *, device: torch.device):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("metric names must be non-empty and unique.")
        if any(not isinstance(name, str) or not name for name in self.names):
            raise ValueError("metric names must be non-empty strings.")
        self._accumulated_values = torch.zeros(len(self.names), device=device)
        self._accumulated_examples = 0

    def add_batch_means(
        self,
        values: Sequence[torch.Tensor],
        *,
        num_examples: int,
    ) -> None:
        """Add batch-mean scalar metrics weighted by the number of examples."""
        if num_examples < 1:
            raise ValueError("num_examples must be positive.")
        if len(values) != len(self.names):
            raise ValueError("metric values do not match configured names.")
        if any(value.ndim != 0 for value in values):
            raise ValueError("metric values must be scalar tensors.")
        stacked_values = (
            torch.stack(tuple(values)).detach().to(self._accumulated_values)
        )
        self._accumulated_values += stacked_values * num_examples
        self._accumulated_examples += num_examples

    def compute_weighted_means(
        self, *, require_finite: bool = False
    ) -> dict[str, float]:
        """Return example-weighted means, optionally rejecting non-finite values."""
        if self._accumulated_examples == 0:
            raise RuntimeError("cannot compute metrics before adding a batch.")
        means = self._accumulated_values / self._accumulated_examples
        if require_finite:
            nonfinite_indices = (~torch.isfinite(means)).nonzero().flatten().tolist()
            if nonfinite_indices:
                metric_index = nonfinite_indices[0]
                name = self.names[metric_index]
                value = means[metric_index].item()
                raise FloatingPointError(f"non-finite mean metric {name}={value}")
        return dict(zip(self.names, means.tolist(), strict=True))


class Accumulator:
    """
    Accumulate sums for multiple metrics.

    Args:
        metric_count: the number of metrics to initialize
    """

    def __init__(self, metric_count):
        self.data = [0.0] * metric_count  # initialize one zero per metric

    def add(self, *args):
        """Add the arguments to the data."""
        vals = []
        for value in args:
            if torch.is_tensor(value):
                value = value.detach()
                if value.dim() == 0:
                    vals.append(value.item())
                else:
                    vals.append(value.float().sum().item())
            else:
                vals.append(float(value))
        self.data = [current + v for current, v in zip(self.data, vals)]

    def __getitem__(self, idx):  # double underscores getitem: get the data at the index
        """Get the data at the index."""
        return self.data[idx]  # return the data at the index


# Keep the textbook entry points without an import cycle with supervised evaluation.
def accuracy(predictions, targets):
    from dl_utils.evaluation.supervised import accuracy as evaluate

    return evaluate(predictions, targets)


def evaluate_accuracy(net, data_iter):
    from dl_utils.evaluation.supervised import evaluate_accuracy as evaluate

    return evaluate(net, data_iter)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    from dl_utils.evaluation.supervised import evaluate_accuracy_gpu as evaluate

    return evaluate(net, data_iter, device=device)


def evaluate_loss(net, data_iter, loss):
    from dl_utils.evaluation.supervised import evaluate_loss as evaluate

    return evaluate(net, data_iter, loss)


__all__ = [
    "Accumulator",
    "MetricAccumulator",
    "MetricHistory",
    "NumericScalar",
    "accuracy",
    "align_and_drop_all_nan_rows",
    "align_metrics_for_csv",
    "as_list",
    "evaluate_accuracy",
    "evaluate_accuracy_gpu",
    "evaluate_loss",
    "has_any_finite",
    "save_metrics_csv",
]
