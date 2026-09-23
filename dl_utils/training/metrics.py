"""Training evaluation and metrics serialization primitives."""

import csv
import math
import os
from collections.abc import Mapping, Sequence
from os import PathLike

import torch
from torch import nn

type NumericScalar = int | float
type MetricHistory = Mapping[str, Sequence[NumericScalar]]


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


def accuracy(predictions, targets):
    """
    Compute the number of correct predictions.

    Args:
        predictions: the predicted value (batch_size, num_outputs) or (batch_size,)
        targets: the true value (batch_size,)
    Returns:
        the number of correct predictions
    """
    # A second dimension with multiple entries means predictions is a matrix.
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predicted_indices = predictions.argmax(
            axis=1
        )  # (batch_size,), index of the maximum probability
        matches = predicted_indices.type(targets.dtype) == targets  # True or False
        return float(matches.type(targets.dtype).sum())  # sum True values as a float
    return 0.0  # return 0.0 if predictions is not a matrix


def evaluate_accuracy(net, data_iter):
    """
    Compute the accuracy for a model on a dataset.

    Args:
        net: the network
        data_iter: the data iterator
    Returns:
        the accuracy of the model
    """
    if isinstance(
        net, torch.nn.Module
    ):  # Determine whether net is an instance of torch.nn.Module
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for features, labels in data_iter:
            metric.add(
                accuracy(net(features), labels), labels.numel()
            )  # metric.add(correct predictions, total predictions)
    return metric[0] / metric[1]  # return the accuracy


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    Evaluate the accuracy of the model on the given dataset using GPU.

    Args:
        net: the model
        data_iter: the data iterator
        device: the device to use (Default: None)
    Returns:
        The accuracy of the model on the given dataset using GPU
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            # Get the device of the first parameter of the net
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for features, labels in data_iter:
            if isinstance(features, list):
                # Required for BERT fine-tuning (to be introduced later)
                device_features = [feature.to(device) for feature in features]
            else:
                device_features = features.to(device)
            device_labels = labels.to(device)
            metric.add(
                accuracy(net(device_features), device_labels), device_labels.numel()
            )
    return metric[0] / metric[1]  # Return the accuracy


def evaluate_loss(net, data_iter, loss):
    """
    Evaluate the model's loss on the given dataset.

    Args:
        net: the model
        data_iter: the data iterator
        loss: the loss function
    Returns:
        the average loss
    """
    metric = Accumulator(2)  # loss_sum, num_samples
    for features, labels in data_iter:
        out = net(features)
        labels = labels.reshape(out.shape)
        batch_loss = loss(out, labels)
        metric.add(batch_loss.sum(), batch_loss.numel())
    return metric[0] / metric[1]


def as_list(values: Sequence[NumericScalar] | None) -> list[NumericScalar]:
    """Return an owned numeric metric list, preserving sequence order."""
    return [] if values is None else list(values)


def has_any_finite(values: Sequence[NumericScalar]) -> bool:
    """Return True if the list contains at least one finite numeric value."""
    return any(math.isfinite(value) for value in values)


def align_metrics_for_csv(metrics: MetricHistory) -> dict[str, list[NumericScalar]]:
    """
    Align a metrics dict to a common length by padding shorter lists with NaN.
    This makes CSV saving robust even when some optional metrics are missing.
    """
    if not metrics:
        return {}
    lengths = [len(as_list(values)) for values in metrics.values()]
    maximum_length = max(lengths) if lengths else 0
    aligned: dict[str, list[NumericScalar]] = {}
    for key, values in metrics.items():
        lst = as_list(values)
        if len(lst) < maximum_length:
            lst = lst + [math.nan] * (maximum_length - len(lst))
        else:
            lst = lst[:maximum_length]
        aligned[key] = lst
    return aligned


def align_and_drop_all_nan_rows(
    metrics: MetricHistory,
    *,
    exclude_keys: set[str] | None = None,
) -> dict[str, list[NumericScalar]]:
    """
    Align a metrics dict to a common length, then DROP rows where all "value" columns
    are non-finite (NaN/inf).

    This is mainly for step-level metrics where we intentionally store NaN as a
    placeholder for steps that skip metric computation (e.g. `log_every_steps`).
    Keeping those NaN rows makes CSVs huge and confusing.

    Args:
        metrics: key -> list values.
        exclude_keys: keys that are treated as coordinates/metadata (not used to
            decide whether a row is "all-NaN"). Defaults to {"step","epoch","step_in_epoch"}.

    Returns:
        A NEW metrics dict with aligned and filtered rows.
    """
    if not metrics:
        return {}

    exclude = exclude_keys or {"step", "epoch", "step_in_epoch"}
    aligned = align_metrics_for_csv(metrics)
    if not aligned:
        return {}

    # Determine which keys are "value columns" we use to decide keep/drop.
    value_keys = [key for key in aligned if key not in exclude]
    if not value_keys:
        return aligned

    # All lists should now be the same length.
    first_key = next(iter(aligned.keys()))
    row_count = len(as_list(aligned[first_key]))
    if row_count <= 0:
        return aligned

    keep_mask: list[bool] = []
    for i in range(row_count):
        keep_mask.append(any(math.isfinite(aligned[key][i]) for key in value_keys))

    # Fast path: nothing to drop
    if all(keep_mask):
        return aligned

    filtered: dict[str, list[NumericScalar]] = {}
    for key, values in aligned.items():
        lst = as_list(values)[:row_count]
        filtered[key] = [lst[i] for i, keep in enumerate(keep_mask) if keep]
    return filtered


def save_metrics_csv(metrics: MetricHistory, path: str | PathLike[str]) -> None:
    """
    Save a metrics dict (key -> list of values) to CSV.

    All lists must have the same length. Values are written as-is (via str()).
    """
    if not metrics:
        raise ValueError("save_metrics_csv: metrics is empty.")

    keys = list(metrics.keys())
    row_count = len(metrics[keys[0]])
    for key in keys[1:]:
        if len(metrics[key]) != row_count:
            raise ValueError(
                f"save_metrics_csv: length mismatch for '{key}', expected {row_count} got {len(metrics[key])}."
            )

    path_str = os.fspath(path)
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    with open(path_str, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        for i in range(row_count):
            writer.writerow([metrics[key][i] for key in keys])
