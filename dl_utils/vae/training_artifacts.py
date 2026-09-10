"""Save VAE epoch metric CSV files and paginated panels in each run root."""

from pathlib import Path

from dl_utils.plot.figures import save_loss_panels
from dl_utils.training.metrics import save_metrics_csv


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
