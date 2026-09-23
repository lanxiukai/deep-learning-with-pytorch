import copy
import os
from dataclasses import dataclass
from typing import NotRequired, Protocol, Sequence, TypeAlias, TypedDict

from ..training.metrics import NumericScalar


class RBMUpdateMetrics(TypedDict):
    mse: float
    bce: float
    energy: float
    free_energy: float
    pl: NotRequired[float]


class DBMFineTuneMetrics(TypedDict):
    mse: float


class MCMCMetrics(TypedDict):
    r_hat: float
    ess_bulk: float
    ess_tail: float
    autocorr_lag1: float


class SamplingOutputDirs(TypedDict):
    sample_root: str
    data_dir: str | None
    data_save_enabled: bool
    grid_prob_dir: str
    grid_binary_dir: str
    timeline_prob_dir: str
    timeline_binary_dir: str
    gif_dir: str
    metrics_dir: str


MetricHistory: TypeAlias = dict[str, list[NumericScalar]]
StepMetrics: TypeAlias = MetricHistory
EpochMetrics: TypeAlias = MetricHistory
DBMPretrainMetrics: TypeAlias = MetricHistory


@dataclass(kw_only=True)
class Config:
    """Shared mutable EBM runtime state."""

    out_dir: str = ""


@dataclass(kw_only=True)
class LayerConfig(Config):
    """A lightweight config used only to reuse genai's loading utilities."""

    dataset: str
    n_visible: int
    n_hidden: int
    ckpt_name: str | None = None
    rbm_param_fixed_point: bool = False
    rbm_param_q_frac: int = 12


class _RBMFixedPointConfig(Protocol):
    rbm_param_fixed_point: bool
    rbm_param_q_frac: int


class _LayerCloneConfig(Protocol):
    out_dir: str


class _RBMArtifactConfig(Protocol):
    out_dir: str
    dataset: str
    size_h: int
    size_w: int
    max_filters: int
    recon_vis_count: int


class _RBMTrainingConfig(_RBMArtifactConfig, Protocol):
    monitor_step_metrics: bool
    save_every_epochs: int
    monitor_pseudo_likelihood: bool
    log_every_steps: int
    lr_decay: float
    momentum_final: float
    momentum_anneal_epoch: int
    momentum: float
    weight_decay: float
    max_w_norm: float
    recon_vis_batches: int
    save_filters: bool


class _RBMSetupConfig(_RBMTrainingConfig, _RBMFixedPointConfig, Protocol):
    seed: int | None
    data_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    resize: int | tuple[int, int] | None
    n_visible: int
    init_bv_from_data: bool
    init_bv_eps: float
    binarize_mode: str
    threshold: float


class _RBMRunConfig(_RBMSetupConfig, Protocol):
    n_hidden: int
    epochs: int
    lr: float
    cd_k: int
    use_pcd: bool
    pcd_init: str


class _DBNRunConfig(_RBMSetupConfig, Protocol):
    n_hidden1: int
    n_hidden2: int
    epochs1: int
    epochs2: int
    lr1: float
    lr2: float
    cd_k1: int
    cd_k2: int
    use_pcd1: bool
    use_pcd2: bool
    pcd_init: str
    init_bv_hidden: bool


class _RBMLoadConfig(_RBMFixedPointConfig, Protocol):
    out_dir: str
    dataset: str
    n_visible: int
    n_hidden: int
    ckpt_name: str | None


class _InitialVisibleConfig(Protocol):
    dataset: str
    data_dir: str
    batch_size: int
    resize: int | tuple[int, int] | None
    num_workers: int
    pin_memory: bool
    gen_count: int
    n_visible: int
    gen_init: str
    binarize_mode: str
    threshold: float


class _SamplingOutputConfig(Protocol):
    out_dir: str
    sample_root: str
    data_save: bool
    data_subdir: str
    grid_prob_subdir: str
    grid_binary_subdir: str
    timeline_prob_subdir: str
    timeline_binary_subdir: str
    gif_subdir: str
    metrics_subdir: str


class _RBMSamplingConfig(_InitialVisibleConfig, _SamplingOutputConfig, _RBMLoadConfig, Protocol):
    seed: int | None
    gen_desync_steps: int | None
    save_steps_divisor: int
    gen_gibbs_steps: int
    data_save_every: int
    size_h: int
    size_w: int
    save_gif: bool
    gif_fps: int

    @property
    def save_steps(self) -> Sequence[int] | None: ...


class _DBNSamplingConfig(_InitialVisibleConfig, _SamplingOutputConfig, _RBMFixedPointConfig, Protocol):
    seed: int | None
    gen_desync_steps: int | None
    save_steps_divisor: int
    gen_gibbs_steps: int
    data_save_every: int
    size_h: int
    size_w: int
    save_gif: bool
    gif_fps: int
    n_hidden1: int
    n_hidden2: int
    ckpt1_name: str | None
    ckpt2_name: str | None

    @property
    def save_steps(self) -> Sequence[int] | None: ...


class _DBMConfig(_RBMSetupConfig, Protocol):
    n_hidden1: int
    n_hidden2: int
    pretrain_epochs1: int
    pretrain_lr1: float
    pretrain_cd_k1: int
    pretrain_use_pcd: bool
    pretrain_epochs2: int
    pretrain_lr2: float
    pretrain_cd_k2: int
    ckpt_pt1_name: str
    ckpt_pt2_name: str
    ckpt_ft_name: str
    metrics_subdir: str
    finetune_epochs: int
    finetune_lr: float
    mf_steps: int
    finetune_pcd_steps: int
    finetune_use_adam: bool
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    max_grad_norm: float
    finetune_use_pcd: bool
    gen_count: int
    gen_gibbs_steps: int
    gen_init: str
    gen_seed: int | None
    gen_desync_steps: int
    save_steps_divisor: int
    sample_root: str
    grid_prob_subdir: str
    grid_binary_subdir: str
    gif_subdir: str
    save_gif: bool
    gif_fps: int
    mcmc_diagnostics: bool

    @property
    def save_steps(self) -> Sequence[int] | None: ...


def layer_cfg[T: _LayerCloneConfig](cfg: T, *, layer_idx: int) -> T:
    """
    Clone cfg with a subdirectory for the given layer.

    The shared ``out_dir`` field is declared by ``Config``; copying retains the
    concrete dataclass type while changing only the layer-local path.
    """
    if not cfg.out_dir:
        raise ValueError("layer_cfg: cfg.out_dir is missing/empty; set cfg.out_dir before calling train_dbn().")

    cloned_cfg = copy.copy(cfg)
    cloned_cfg.out_dir = os.path.join(cfg.out_dir, f"layer{layer_idx}")
    return cloned_cfg
