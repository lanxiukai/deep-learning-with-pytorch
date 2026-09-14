"""
Binary RBM (Bernoulli-Bernoulli) training script with PyTorch

Overview:
  1) Loads a torchvision vision dataset and flattens images to vectors.
     Default is MNIST; FashionMNIST is also supported.
  2) Preprocesses inputs to float in [0,1] and binarizes visible units
     (stochastic Bernoulli sampling or fixed threshold).
  3) Trains a Bernoulli-Bernoulli RBM with CD-k, with optional Persistent CD (PCD).
     Updates are done manually (no autograd) with momentum, optional weight decay,
     optional max-norm on weight columns, and optional visible-bias init from data.
  4) Logs step/epoch metrics (reconstruction MSE/BCE, energy, free energy, and
     optional pseudo-likelihood) and saves them to CSV and plotted curves.
  5) Saves visual artifacts: originals vs reconstructions, generated samples
     (binary and probability grids), and learned filters (weight columns).

Notes:
  - Visible units: Bernoulli (pixels in {0,1} after binarization).
  - Hidden units: Bernoulli
  - Outputs are written under cfg.out_dir (e.g. <project_root>/output/...).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os

from dl_utils.runtime.devices import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.ebm._ebm_types import Config as EBMConfig
from dl_utils.ebm.rbm_sampling import sampling_rbm
from dl_utils.ebm.rbm_training import train_rbm

_PROJECT_ROOT = infer_project_root()
_OUTPUT_ROOT = str(_PROJECT_ROOT / "output")

# -----------------------------
# Config
# -----------------------------
@dataclass(kw_only=True)
class Config(EBMConfig):
    seed: int | None = 42

    # System
    data_dir: str = str(_PROJECT_ROOT / "data" / "mnist")
    num_workers: int = 16    # number of CPU cores working on data loading
    pin_memory: bool = True  # whether to pin memory for faster data transfer

    # Dataset
    dataset: str = "mnist"  # "mnist", "fashion_mnist"
    resize: int | tuple[int, int] | None = 28    # NOTE: image size (after optional resize)
    size_h: int = 28 if resize is None else (resize if isinstance(resize, int) else resize[0])
    size_w: int = 28 if resize is None else (resize if isinstance(resize, int) else resize[1])

    # Binarization
    binarize_mode: str = "stochastic"  # "stochastic" or "threshold"
    threshold: float = 0.5

    # Model parameters
    batch_size: int = 256
    n_visible: int = size_h * size_w
    n_hidden: int = 512       # NOTE: the number of hidden units

    # Training parameters
    epochs: int = 100         # NOTE: the number of epochs
    lr: float = 0.05          # NOTE: the learning rate
    cd_ks: list[int] = field(default_factory=lambda: [1])  # NOTE: CD-k values to train with
    cd_k: int = 1
    use_pcd: bool = True      # NOTE: Persistent CD (PCD) often improves sample quality vs plain CD
    pcd_init: str = "random"  # "random" or "data"

    # Useful tricks
    lr_decay: float = 1.0           # per-epoch multiplicative decay (1.0 = no decay)
    momentum: float = 0.5           # momentum SGD
    momentum_final: float = 0.5
    momentum_anneal_epoch: int = 5  # switch to momentum_final at this epoch (1-based); <=1 disables warmup
    weight_decay: float = 1e-4      # L2 on W
    max_w_norm: float = 0.0         # >0 enables per-hidden-unit max-norm constraint on columns of W
    init_bv_from_data: bool = True  # initialize visible bias bv with logit(mean pixel) to speed up training
    init_bv_eps: float = 1e-4
    monitor_pseudo_likelihood: bool = True  # log pseudo-likelihood (cheap, correlates better than MSE)
    rbm_param_fixed_point: bool = False
    rbm_param_q_frac: int = 12

    # Logging / visualization
    out_root: str = (f"{_OUTPUT_ROOT}/{dataset}-size={size_h}x{size_w}/"
                    f"rbm-hiddens={n_hidden}-epochs={epochs}-lr={lr}")

    monitor_step_metrics: bool = True  # NOTE: log per-step metrics (can be slow)
    log_every_steps: int = 1           # NOTE: provide log accelerate
    recon_vis_batches: int = 1   # how many batches to visualize per epoch
    recon_vis_count: int = 25    # how many images to show (must be square-ish)
    gen_count: int = 25          # generated samples count (square number recommended)
    gen_gibbs_steps: int = 1000  # NOTE: Gibbs steps for generation
    gen_init: str = "random"     # "random" or "data"
    max_filters: int = 25        # maximum number of filters to save
    save_filters: bool = False   # whether to save learned filters
    save_every_epochs: int = 10  # NOTE: only save grid/filters when epoch==1 
                                 # or epoch%save_every_epochs==0 (clamped to >=1)
    data_save: bool = False      # whether to save numeric dumps; default off
    data_save_every: int = 100   # save numeric data every k steps (step 1 always saved when enabled)
    # Steps to save images/grids:
    # 1) If save_steps is a tuple, use it as-is (after validation).
    # 2) If save_steps is None, auto-generate: (1, all steps divisible 
    # by save_steps_divisor) within [1, gen_gibbs_steps].
    # NOTE: save_steps_divisor is only used when save_steps is None (auto-generated).
    save_steps: tuple[int, ...] | None = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
    save_steps_divisor: int = 20
    gen_desync_steps: int | None = None

    # Output subdirectories (must not clash with training subfolders)
    sample_root: str = "sample"
    data_subdir: str = "data"
    timeline_binary_subdir: str = "timelines_binary"
    timeline_prob_subdir: str = "timelines_prob"
    grid_binary_subdir: str = "grids_binary"
    grid_prob_subdir: str = "grids_prob"
    metrics_subdir: str = "metrics"

    # Optional GIF settings
    save_gif: bool = True
    gif_fps: int = 2
    gif_subdir: str = "gifs"

    # Optional override for checkpoint filenames
    ckpt_name: str | None = None

# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    device = try_gpu()
    cfg = Config()
    for cd_k in cfg.cd_ks:
        cfg.out_dir = os.path.join(cfg.out_root, f"cd_k={cd_k}")
        cfg.cd_k = cd_k
        train_rbm(cfg, device)
        sampling_rbm(cfg, device)


if __name__ == "__main__":
    main()
