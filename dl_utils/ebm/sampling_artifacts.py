import os
from collections.abc import Sequence

import imageio.v2 as imageio
import torch

from dl_utils.plot._backend import pyplot as plt

from ..plot.figures import save_curve
from dl_utils.ebm._ebm_types import _DBNSamplingConfig, _RBMSamplingConfig


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_sample_data(step: int, prob: torch.Tensor, binary: torch.Tensor, out_dir: str) -> None:
    for idx in range(prob.size(0)):
        payload = {
            "step": step,
            "prob": prob[idx],
            "binary": binary[idx],
        }
        path = os.path.join(out_dir, f"sample{idx:04d}_step{step:04d}.pt")
        torch.save(payload, path)


def save_timelines_for_samples(
    steps: Sequence[int],
    step_to_prob: dict[int, torch.Tensor],
    cfg: _RBMSamplingConfig | _DBNSamplingConfig,
    out_dir: str,
) -> None:
    ordered_steps = [s for s in steps if s in step_to_prob]
    if not ordered_steps:
        return
    num_samples = next(iter(step_to_prob.values())).size(0)
    H, W = cfg.size_h, cfg.size_w

    for idx in range(num_samples):
        fig, axes = plt.subplots(1, len(ordered_steps), figsize=(3.0 * len(ordered_steps), 3.0))
        if len(ordered_steps) == 1:
            axes = [axes]
        for ax, step in zip(axes, ordered_steps):
            img = step_to_prob[step][idx].reshape(H, W)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"step {step}")
            ax.axis("off")
        fig.suptitle(f"Sample {idx}")
        fig.tight_layout()
        path = os.path.join(out_dir, f"sample{idx:04d}_steps.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)


def save_gif_from_grids(steps: Sequence[int], grid_dir: str, gif_dir: str, fps: int, gif_name: str) -> None:
    frames = []
    for step in steps:
        img_path = os.path.join(grid_dir, f"step{step:04d}_grid.png")
        if os.path.exists(img_path):
            frames.append(imageio.imread(img_path))
    if not frames:
        print(f"No grid images found in {grid_dir}; skip GIF.")
        return
    ensure_dir(gif_dir)
    gif_path = os.path.join(gif_dir, f"{gif_name}.gif")
    imageio.mimsave(gif_path, frames, fps=max(1, int(fps)))


def save_gibbs_energy_curves(
    steps: Sequence[float],
    energy: Sequence[float],
    free_energy: Sequence[float],
    out_dir: str,
    *,
    prefix: str,
    title_prefix: str | None = None,
) -> None:
    """
    Save Gibbs-step energy/free-energy curves (combined and separate) and a single CSV
    containing both curves.
    """
    if not steps or not energy or not free_energy:
        return

    n = min(len(steps), len(energy), len(free_energy))
    if n <= 0:
        return

    steps_use = list(steps)[:n]
    energy_use = list(energy)[:n]
    free_energy_use = list(free_energy)[:n]

    out_dir = ensure_dir(out_dir)
    combined_csv = os.path.join(out_dir, f"{prefix}_gibbs_energy_free_energy.csv")

    def _title(base: str) -> str:
        if title_prefix:
            return f"{title_prefix} - {base}"
        return base

    # Combined curve (energy + free energy) — only this saves CSV
    save_curve(
        steps_use,
        {"energy": energy_use, "free_energy": free_energy_use},
        path=os.path.join(out_dir, f"{prefix}_gibbs_energy_free_energy.png"),
        xlabel="Gibbs step",
        ylabel="Energy",
        title=_title("Energy and free energy vs Gibbs step"),
        csv_path=combined_csv,
    )

    # Separate curves — no extra CSVs to avoid duplication
    save_curve(
        steps_use,
        {"energy": energy_use},
        path=os.path.join(out_dir, f"{prefix}_gibbs_energy.png"),
        xlabel="Gibbs step",
        ylabel="Energy",
        title=_title("Energy vs Gibbs step"),
    )
    save_curve(
        steps_use,
        {"free_energy": free_energy_use},
        path=os.path.join(out_dir, f"{prefix}_gibbs_free_energy.png"),
        xlabel="Gibbs step",
        ylabel="Free energy",
        title=_title("Free energy vs Gibbs step"),
    )
