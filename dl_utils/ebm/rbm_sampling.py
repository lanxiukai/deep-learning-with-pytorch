import math
import os

import torch
from tqdm.auto import tqdm

from ..plot.images import save_grid
from ..runtime.randomness import set_seed
from ..training.timing import Timer
from ._ebm_types import _RBMSamplingConfig
from .rbm_primitives import rbm_energy_free_energy_fast
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
    load_rbm_model_from_ckpt,
    prepare_init_visible,
    validate_out_dir_and_ckpt,
)


def sampling_rbm(cfg: _RBMSamplingConfig, device: torch.device) -> None:
    set_seed(cfg.seed)
    ckpt_path = validate_out_dir_and_ckpt(cfg)
    rbm = load_rbm_model_from_ckpt(cfg, device)
    print(f"Loaded RBM checkpoint from: {ckpt_path}")
    v, _ = prepare_init_visible(cfg, device)
    v = _desync_gibbs_visible_chains(rbm, v, _infer_gen_desync_steps(cfg))
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
    energy_buf = torch.empty(cfg.gen_gibbs_steps, device=device)
    free_energy_buf = torch.empty(cfg.gen_gibbs_steps, device=device)
    step_iter = tqdm(range(1, cfg.gen_gibbs_steps + 1), total=cfg.gen_gibbs_steps, desc="Gibbs", unit="step")
    timer = Timer()
    nrow = max(1, int(math.sqrt(cfg.gen_count)))
    for step in step_iter:
        timer.start()
        _, _, p_v, v = rbm.gibbs_vhv(v)
        timer.stop()
        e, fe, _ = rbm_energy_free_energy_fast(rbm, v, h=None)
        energy_buf[step - 1] = e.mean()
        free_energy_buf[step - 1] = fe.mean()
        gibbs_steps.append(step)
        need_data = data_save_enabled and (step == 1 or step % data_every == 0)
        need_grid = step in save_steps_set
        if need_data or need_grid:
            p_v_cpu = p_v.detach().cpu()
            v_cpu = v.detach().cpu()
            if need_data and data_dir is not None:
                save_sample_data(step, p_v_cpu, v_cpu, data_dir)
            if need_grid:
                step_to_prob[step] = p_v_cpu
                step_to_binary[step] = v_cpu
                prob_grid_path = os.path.join(grid_prob_dir, f"step{step:04d}_grid.png")
                save_grid(images=p_v_cpu, path=prob_grid_path, nrow=nrow,
                          title=f"Gibbs step {step} (prob)", hw=(cfg.size_h, cfg.size_w))
                binary_grid_path = os.path.join(grid_binary_dir, f"step{step:04d}_grid.png")
                save_grid(images=v_cpu, path=binary_grid_path, nrow=nrow,
                          title=f"Gibbs step {step} (binary)", hw=(cfg.size_h, cfg.size_w))
    print(f"Total time: {timer.format_time(timer.sum(), precision=3)} "
          f"| Sampling rate: {cfg.gen_gibbs_steps / timer.sum():.1f} steps/s")
    energy_per_step = energy_buf.detach().cpu().tolist()
    free_energy_per_step = free_energy_buf.detach().cpu().tolist()
    save_gibbs_energy_curves(gibbs_steps, energy_per_step, free_energy_per_step, metrics_dir, prefix="rbm")
    if step_to_prob:
        save_timelines_for_samples(save_steps, step_to_prob, cfg, timeline_prob_dir)
    if step_to_binary:
        save_timelines_for_samples(save_steps, step_to_binary, cfg, timeline_binary_dir)
    if cfg.save_gif and step_to_prob:
        save_gif_from_grids(save_steps, grid_prob_dir, gif_dir, cfg.gif_fps, "gibbs_prob")
    if cfg.save_gif and step_to_binary:
        save_gif_from_grids(save_steps, grid_binary_dir, gif_dir, cfg.gif_fps, "gibbs_binary")
    print("Done. Outputs written to:", sample_root, "\n")
