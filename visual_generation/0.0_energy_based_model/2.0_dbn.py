"""
Two-layer Deep Belief Network (stacked RBMs) training script with PyTorch.

Workflow:
  1) Train the first-layer RBM on binarized pixels.
  2) Train the second-layer RBM on samples from the first layer.
  3) Save metrics/filters/checkpoints for each layer and DBN-generated samples.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os

from dl_utils.runtime.devices import try_gpu
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.ebm._ebm_types import Config as EBMConfig
from dl_utils.ebm.dbn import sampling_dbn, train_dbn

_PROJECT_ROOT = infer_project_root()
_OUTPUT_ROOT = str(_PROJECT_ROOT / "output")


# -----------------------------
# Config
# -----------------------------
@dataclass(kw_only=True)
class Config(EBMConfig):
    seed: int | None = 42

    # System
    data_dir: str = str(_PROJECT_ROOT / "data" / "fashion_mnist")
    num_workers: int = 16
    pin_memory: bool = True

    # Dataset
    dataset: str = "fashion_mnist"  # "mnist", "fashion_mnist"
    resize: int | tuple[int, int] | None = None    # NOTE: image size (after optional resize)
    size_h: int = 28 if resize is None else (resize if isinstance(resize, int) else resize[0])
    size_w: int = 28 if resize is None else (resize if isinstance(resize, int) else resize[1])

    # Binarization
    binarize_mode: str = "stochastic"  # "stochastic" or "threshold"
    threshold: float = 0.5

    # Model sizes
    batch_size: int = 256
    n_visible: int = size_h * size_w
    n_hidden1: int = 2048     # NOTE: the number of hidden units for the first layer
    n_hidden2: int = 1024     # NOTE: the number of hidden units for the second layer

    # NOTE: the training params (layer-specific)
    epochs1: int = 20
    epochs2: int = 20
    lr1: float = 0.01
    lr2: float = 0.03
    # NOTE: the values of cd_k for both layers (layer-specific)
    cd_k1s: list[int] = field(default_factory=lambda: [1])
    cd_k2s: list[int] = field(default_factory=lambda: [10]) 
    cd_k1: int = 1
    cd_k2: int = 10
    use_pcd1: bool = True
    use_pcd2: bool = True
    pcd_init: str = "random"  # "random" or "data"

    # Optim tricks
    lr_decay: float = 1.0
    momentum: float = 0.5
    momentum_final: float = 0.5
    momentum_anneal_epoch: int = 5
    weight_decay: float = 1e-4
    max_w_norm: float = 0.0
    init_bv_from_data: bool = True
    init_bv_hidden: bool = True  # initialize layer-2 visible bias from layer-1 samples
    init_bv_eps: float = 1e-4
    monitor_pseudo_likelihood: bool = True
    rbm_param_fixed_point: bool = False
    rbm_param_q_frac: int = 12

    # Logging / visualization
    out_root: str = (
        f"{_OUTPUT_ROOT}/{dataset}-size={size_h}x{size_w}/"
        f"dbn-h1={n_hidden1}-h2={n_hidden2}/"
        f"epochs1={epochs1}-epochs2={epochs2}-lr1={lr1}-lr2={lr2}"
    )
    
    monitor_step_metrics: bool = False  # NOTE: log per-step metrics (can be slow)
    log_every_steps: int = 1            # NOTE: log every N steps (when monitor_step_metrics is enabled)
    recon_vis_batches: int = 1
    recon_vis_count: int = 25
    gen_count: int = 25
    gen_gibbs_steps: int = 500   # NOTE: Gibbs steps on the top RBM when generating
    gen_init: str = "random"     # "random" or "data"
    max_filters: int = 25
    save_filters: bool = False   # FIXME: Errors saving second-layer filters (acceptable)
    save_every_epochs: int = 10  # NOTE: only save grids/filters when epoch==1 or epoch%save_every_epochs==0 (>=1)
    data_save: bool = False
    data_save_every: int = 100
    # NOTE: save_steps_divisor is only used when save_steps is None (auto-generated).
    save_steps: tuple[int, ...] | None = (1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500)
    save_steps_divisor: int = 20
    gen_desync_steps: int | None = None

    # Output subdirectories
    sample_root: str = "sample"
    data_subdir: str = "data"
    grid_prob_subdir: str = "grids_prob"
    grid_binary_subdir: str = "grids_binary"
    timeline_prob_subdir: str = "timelines_prob"
    timeline_binary_subdir: str = "timelines_binary"
    metrics_subdir: str = "metrics"

    # Optional GIF settings
    save_gif: bool = True
    gif_fps: int = 2
    gif_subdir: str = "gifs"

    # Optional override for checkpoint filenames
    ckpt1_name: str | None = None
    ckpt2_name: str | None = None


# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    device = try_gpu()
    cfg = Config()
    for cd_k1, cd_k2 in zip(cfg.cd_k1s, cfg.cd_k2s):
        cfg.out_dir = os.path.join(cfg.out_root, f"cd_k1={cd_k1}-cd_k2={cd_k2}")
        cfg.cd_k1 = cd_k1
        cfg.cd_k2 = cd_k2
        train_dbn(cfg, device)
        sampling_dbn(cfg, device)


if __name__ == "__main__":
    main()
