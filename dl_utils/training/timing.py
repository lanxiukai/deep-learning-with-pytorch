"""Training-time reports and the retained textbook Timer import."""

from dl_utils.runtime.timing import Timer

__all__ = ["Timer", "format_epoch_timing"]


def format_epoch_timing(
    total_seconds: float,
    num_epochs: int,
    precision: int = 1,
) -> str:
    """Format total training time and the average time per epoch."""
    if total_seconds < 0:
        raise ValueError("total_seconds must be non-negative.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if precision < 0:
        raise ValueError("precision must be non-negative.")

    units_per_second = 10**precision
    total_units = round(total_seconds * units_per_second)
    hours, remainder = divmod(total_units, 3600 * units_per_second)
    minutes, second_units = divmod(remainder, 60 * units_per_second)
    seconds = second_units / units_per_second

    average_units = round(total_seconds / num_epochs * units_per_second)
    average_minutes, average_second_units = divmod(
        average_units,
        60 * units_per_second,
    )
    average_seconds = average_second_units / units_per_second

    return (
        f"training time {hours} h {minutes} min "
        f"{seconds:.{precision}f} s, avg {average_minutes} min "
        f"{average_seconds:.{precision}f} s/epoch"
    )
