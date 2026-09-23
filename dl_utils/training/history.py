"""Metric-history alignment and CSV serialization, independent of model execution."""

import csv
import math
import os
from collections.abc import Mapping, Sequence
from os import PathLike

type NumericScalar = int | float


type MetricHistory = Mapping[str, Sequence[NumericScalar]]


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
