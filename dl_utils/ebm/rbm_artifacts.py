import math
import os

import torch

from ..plot.images import save_grid
from dl_utils.ebm._ebm_types import _RBMArtifactConfig
from dl_utils.ebm.rbm_model import BinaryRBM


def save_filters_grids(
    cfg: _RBMArtifactConfig,
    W: torch.Tensor,
    epoch: int,
    *,
    filters_subdir: str = "filters",
) -> None:
    """
    Visualize columns of W as square "filters".
    W: [V, H]
    """
    filters_dir = os.path.join(cfg.out_dir, filters_subdir)
    V, H = W.shape
    n = min(H, cfg.max_filters)
    filt = W[:, :n].t().contiguous().reshape(n, 1, cfg.size_h, cfg.size_w)

    # Normalize per-filter to [0,1] for visualization
    fmin = filt.amin(dim=(1, 2, 3), keepdim=True)
    fmax = filt.amax(dim=(1, 2, 3), keepdim=True)
    filt_norm = (filt - fmin) / (fmax - fmin + 1e-8)

    nrow = int(math.sqrt(n))
    nrow = max(1, nrow)
    save_grid(
        filt_norm,
        os.path.join(filters_dir, f"epoch{epoch:02d}_filters.png"),
        nrow=nrow,
        title="Learned filters (W columns)",
        hw=(cfg.size_h, cfg.size_w),
    )


@torch.no_grad()
def save_rbm_recon_grids(
    rbm: BinaryRBM,
    v0: torch.Tensor,
    epoch: int,
    step: int,
    cfg: _RBMArtifactConfig,
    *,
    x_orig: torch.Tensor | None = None,
    recon_prob_subdir: str = "recon_prob",
    recon_binary_subdir: str = "recon_binary",
) -> None:
    """
    Save grayscale/probability reconstructions and binarized reconstructions into separate folders.
      - recon_prob: original grayscale inputs and one-step reconstruction probability maps
      - recon_binary: binarized inputs and one-step binarized reconstructions
    """
    count = min(cfg.recon_vis_count, v0.size(0))
    pv, v1 = rbm.reconstruct(v0[:count])
    nrow = int(math.sqrt(count))
    nrow = max(1, nrow)

    # Directories
    prob_dir = os.path.join(cfg.out_dir, recon_prob_subdir)
    binary_dir = os.path.join(cfg.out_dir, recon_binary_subdir)

    # Use a loop to avoid repeated nearly-identical save_grid calls.
    gray = x_orig[:count] if x_orig is not None else v0[:count]
    grids = [
        (gray, prob_dir, "orig_gray", "Original (gray)"),
        (pv[:count], prob_dir, "recon_prob", "Reconstruction prob"),
        (v0[:count], binary_dir, "orig_binary", "Original (binary)"),
        (v1[:count], binary_dir, "recon_binary", "Reconstruction binary"),
    ]
    for images, directory, label, title_base in grids:
        save_grid(
            images,
            os.path.join(directory, f"epoch{epoch:02d}_batch{step:02d}_{label}.png"),
            nrow=nrow,
            title=f"{title_base} - epoch {epoch} batch {step}",
            hw=(cfg.size_h, cfg.size_w),
        )
