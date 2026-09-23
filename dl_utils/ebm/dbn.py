import math
import os

import torch
from tqdm.auto import tqdm

from ..plot.images import save_grid
from ..runtime.randomness import set_seed
from ..training.timing import Timer
from ._ebm_types import LayerConfig, _DBNRunConfig, _DBNSamplingConfig, layer_cfg
from .rbm_model import BinaryRBM
from .rbm_primitives import binarize, rbm_energy_free_energy_fast
from .rbm_training import (
    _configure_rbm_fixed_point,
    _prepare_layer1_rbm,
    init_hidden_visible_bias,
    train_rbm_layer,
)
from .sampling_artifacts import (
    save_gibbs_energy_curves,
    save_gif_from_grids,
    save_sample_data,
    save_timelines_for_samples,
)
from .sampling_common import (
    _desync_gibbs_visible_chains,
    _infer_gen_desync_steps,
    _parse_save_steps,
    _sampling_output_dirs,
    checkpoint_path,
    load_rbm_model_from_ckpt,
    prepare_init_visible,
)


def train_dbn(cfg: _DBNRunConfig, device: torch.device) -> None:
    rbm1, train_iter, test_iter, transform_l1 = _prepare_layer1_rbm(cfg, device, cfg.n_hidden1)
    cfg_l1 = layer_cfg(cfg, layer_idx=1)
    train_rbm_layer(
        rbm1, train_iter, cfg=cfg_l1, device=device, epochs=cfg.epochs1,
        lr_base=cfg.lr1, cd_k=cfg.cd_k1, use_pcd=cfg.use_pcd1, pcd_init=cfg.pcd_init,
        layer_idx=1, transform=transform_l1, gen_iter=test_iter,
    )
    rbm2 = BinaryRBM(cfg.n_hidden1, cfg.n_hidden2, device=device)
    if cfg.init_bv_hidden:
        init_hidden_visible_bias(
            rbm_lower=rbm1, rbm_upper=rbm2, train_iter=train_iter,
            n_visible=cfg.n_hidden1, init_bv_eps=cfg.init_bv_eps, device=device,
            max_batches=20,
        )
    _configure_rbm_fixed_point(rbm2, cfg)
    cfg_l2 = layer_cfg(cfg, layer_idx=2)

    def transform_l2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x.to(device, non_blocking=True).reshape(x.size(0), -1)
        v_bin = binarize(x, mode=cfg.binarize_mode, threshold=cfg.threshold)
        _, h1 = rbm1.sample_h_given_v(v_bin)
        return h1, None

    train_rbm_layer(
        rbm2, train_iter, cfg=cfg_l2, device=device, epochs=cfg.epochs2,
        lr_base=cfg.lr2, cd_k=cfg.cd_k2, use_pcd=cfg.use_pcd2, pcd_init=cfg.pcd_init,
        layer_idx=2, transform=transform_l2,
    )
    print("Done. Outputs written to:", cfg.out_dir, "\n")


def _load_layer_rbm(
    cfg: _DBNSamplingConfig,
    *,
    layer_idx: int,
    n_visible: int,
    n_hidden: int,
    device: torch.device,
    ckpt_name: str | None,
) -> BinaryRBM:
    layer_out_dir = os.path.join(cfg.out_dir, f"layer{layer_idx}")
    ckpt = ckpt_name or f"layer{layer_idx}_rbm_{cfg.dataset}.pt"
    layer_cfg = LayerConfig(
        dataset=cfg.dataset, out_dir=layer_out_dir, n_visible=n_visible,
        n_hidden=n_hidden, ckpt_name=ckpt,
    )
    layer_cfg.rbm_param_fixed_point = cfg.rbm_param_fixed_point
    layer_cfg.rbm_param_q_frac = cfg.rbm_param_q_frac
    if not os.path.isdir(layer_out_dir):
        raise FileNotFoundError(f"layer{layer_idx} output directory not found: {layer_out_dir}")
    ckpt_path = checkpoint_path(layer_cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"layer{layer_idx} checkpoint not found: {ckpt_path}")
    rbm = load_rbm_model_from_ckpt(layer_cfg, device)
    print(f"Loaded layer{layer_idx} RBM from: {ckpt_path}")
    return rbm


def sampling_dbn(cfg: _DBNSamplingConfig, device: torch.device) -> None:
    set_seed(cfg.seed)
    rbm1 = _load_layer_rbm(
        cfg, layer_idx=1, n_visible=cfg.n_visible, n_hidden=cfg.n_hidden1,
        device=device, ckpt_name=cfg.ckpt1_name,
    )
    rbm2 = _load_layer_rbm(
        cfg, layer_idx=2, n_visible=cfg.n_hidden1, n_hidden=cfg.n_hidden2,
        device=device, ckpt_name=cfg.ckpt2_name,
    )
    v0, _ = prepare_init_visible(cfg, device)
    _, h1 = rbm1.sample_h_given_v(v0)
    h1 = _desync_gibbs_visible_chains(rbm2, h1, _infer_gen_desync_steps(cfg))
    dirs = _sampling_output_dirs(cfg)
    sample_root = dirs["sample_root"]
    data_save_enabled = dirs["data_save_enabled"]
    data_dir = dirs["data_dir"]
    grid_prob_dir = dirs["grid_prob_dir"]
    grid_binary_dir = dirs["grid_binary_dir"]
    timeline_prob_dir = dirs["timeline_prob_dir"]
    timeline_binary_dir = dirs["timeline_binary_dir"]
    gif_dir = dirs["gif_dir"]
    metrics_dir = dirs["metrics_dir"]
    save_steps = _parse_save_steps(cfg, divisor_default=True)
    save_steps_set = set(save_steps)
    data_every = max(1, cfg.data_save_every)
    step_to_prob: dict[int, torch.Tensor] = {}
    step_to_binary: dict[int, torch.Tensor] = {}
    gibbs_steps: list[int] = []
    e1_buf = torch.empty(cfg.gen_gibbs_steps, device=device)
    fe1_buf = torch.empty(cfg.gen_gibbs_steps, device=device)
    e2_buf = torch.empty(cfg.gen_gibbs_steps, device=device)
    fe2_buf = torch.empty(cfg.gen_gibbs_steps, device=device)
    step_iter = tqdm(range(1, cfg.gen_gibbs_steps + 1), total=cfg.gen_gibbs_steps,
                     desc="Gibbs (top RBM)", unit="step")
    timer = Timer()
    nrow = max(1, int(math.sqrt(cfg.gen_count)))
    for step in step_iter:
        timer.start()
        _, h2, _, h1 = rbm2.gibbs_vhv(h1)
        p_visible = rbm1.prob_v_given_h(h1)
        v_visible = torch.bernoulli(p_visible)
        timer.stop()
        e1, fe1, _ = rbm_energy_free_energy_fast(rbm1, v_visible, h=h1)
        e2, fe2, _ = rbm_energy_free_energy_fast(rbm2, h1, h=h2)
        e1_buf[step - 1] = e1.mean()
        fe1_buf[step - 1] = fe1.mean()
        e2_buf[step - 1] = e2.mean()
        fe2_buf[step - 1] = fe2.mean()
        gibbs_steps.append(step)
        need_data = data_save_enabled and (step == 1 or step % data_every == 0)
        need_grid = step in save_steps_set
        if need_data or need_grid:
            p_visible_cpu = p_visible.detach().cpu()
            v_visible_cpu = v_visible.detach().cpu()
            if need_data and data_dir is not None:
                save_sample_data(step, p_visible_cpu, v_visible_cpu, data_dir)
            if need_grid:
                step_to_prob[step] = p_visible_cpu
                step_to_binary[step] = v_visible_cpu
                prob_grid_path = os.path.join(grid_prob_dir, f"step{step:04d}_grid.png")
                save_grid(images=p_visible_cpu, path=prob_grid_path, nrow=nrow,
                          title=f"Gibbs step {step} (prob)", hw=(cfg.size_h, cfg.size_w))
                binary_grid_path = os.path.join(grid_binary_dir, f"step{step:04d}_grid.png")
                save_grid(images=v_visible_cpu, path=binary_grid_path, nrow=nrow,
                          title=f"Gibbs step {step} (binary)", hw=(cfg.size_h, cfg.size_w))
    total_time = timer.sum()
    rate = cfg.gen_gibbs_steps / max(1e-8, total_time)
    print(f"Total time: {timer.format_time(total_time, precision=3)} "
          f"| Sampling rate: {rate:.1f} steps/s")
    energy_rbm1_per_step = e1_buf.detach().cpu().tolist()
    free_energy_rbm1_per_step = fe1_buf.detach().cpu().tolist()
    energy_rbm2_per_step = e2_buf.detach().cpu().tolist()
    free_energy_rbm2_per_step = fe2_buf.detach().cpu().tolist()
    save_gibbs_energy_curves(
        gibbs_steps, energy_rbm1_per_step, free_energy_rbm1_per_step, metrics_dir,
        prefix="dbn_rbm1", title_prefix="DBN RBM1 (layer1)",
    )
    save_gibbs_energy_curves(
        gibbs_steps, energy_rbm2_per_step, free_energy_rbm2_per_step, metrics_dir,
        prefix="dbn_rbm2", title_prefix="DBN RBM2 (layer2)",
    )
    if step_to_prob:
        save_timelines_for_samples(save_steps, step_to_prob, cfg, timeline_prob_dir)
    if step_to_binary:
        save_timelines_for_samples(save_steps, step_to_binary, cfg, timeline_binary_dir)
    if cfg.save_gif and step_to_prob:
        save_gif_from_grids(save_steps, grid_prob_dir, gif_dir, cfg.gif_fps, "gibbs_prob")
    if cfg.save_gif and step_to_binary:
        save_gif_from_grids(save_steps, grid_binary_dir, gif_dir, cfg.gif_fps, "gibbs_binary")
    print("Done. Outputs written to:", sample_root, "\n")
