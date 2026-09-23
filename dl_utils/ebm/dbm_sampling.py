import math
import os

import imageio.v2 as imageio
import imageio.v3 as imageio_v3
import torch
from imageio.typing import ArrayLike
from tqdm.auto import tqdm

from ..data.vision import vision_loaders
from ..filesystem.directories import reset_dir
from ..plot.images import save_grid
from ..runtime.randomness import set_seed
from ..training.timing import Timer
from ._ebm_types import _DBMConfig
from .dbm_diagnostics import _compute_mcmc_metrics, _save_mcmc_convergence_plots
from .dbm_model import DBM
from .rbm_primitives import binarize
from .sampling_artifacts import ensure_dir
from .sampling_common import _desync_prepare, _parse_save_steps


@torch.no_grad()
def _desync_dbm_chains(dbm: DBM, v: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, max_offset_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps, offsets = _desync_prepare(v, max_offset_steps)
    if offsets is None:
        return v, h1, h2
    for step in range(1, max_steps + 1):
        v_new, h1_new, h2_new, _, _, _ = dbm.gibbs_step(v, h1, h2)
        mask = offsets >= step
        v = torch.where(mask.unsqueeze(1), v_new, v)
        h1 = torch.where(mask.unsqueeze(1), h1_new, h1)
        h2 = torch.where(mask.unsqueeze(1), h2_new, h2)
    return v, h1, h2


def sampling_dbm(cfg: _DBMConfig, device: torch.device) -> None:
    set_seed(cfg.seed)
    checkpoint_path = os.path.join(cfg.out_dir, cfg.ckpt_ft_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Fine-tuned DBM checkpoint not found: {checkpoint_path}\nRun train_dbm() first.")
    dbm = DBM.load(checkpoint_path, device)
    dbm.eval()
    print(f"Loaded DBM from {checkpoint_path}")
    batch_size = cfg.gen_count
    generator = torch.Generator(device=device)
    if cfg.gen_seed is not None:
        generator.manual_seed(cfg.gen_seed)
    if cfg.gen_init == "random":
        v = torch.bernoulli(0.5 * torch.ones(batch_size, cfg.n_visible, device=device), generator=generator)
        h1 = torch.bernoulli(0.5 * torch.ones(batch_size, cfg.n_hidden1, device=device), generator=generator)
        h2 = torch.bernoulli(0.5 * torch.ones(batch_size, cfg.n_hidden2, device=device), generator=generator)
    else:
        _, test_iter = vision_loaders(dataset=cfg.dataset, data_dir=cfg.data_dir, batch_size=batch_size, resize=cfg.resize, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
        x, _ = next(iter(test_iter))
        v = binarize(x.to(device).reshape(batch_size, -1), mode=cfg.binarize_mode, threshold=cfg.threshold)
        mu1, mu2 = dbm.mean_field(v, n_steps=cfg.mf_steps)
        h1, h2 = torch.bernoulli(mu1, generator=generator), torch.bernoulli(mu2, generator=generator)
    v, h1, h2 = _desync_dbm_chains(dbm, v, h1, h2, max_offset_steps=cfg.gen_desync_steps)
    sample_root = os.path.join(cfg.out_dir, cfg.sample_root)
    reset_dir(sample_root)
    grid_prob_dir = ensure_dir(os.path.join(sample_root, cfg.grid_prob_subdir))
    grid_bin_dir = ensure_dir(os.path.join(sample_root, cfg.grid_binary_subdir))
    gif_dir = ensure_dir(os.path.join(sample_root, cfg.gif_subdir))
    metrics_dir = ensure_dir(os.path.join(sample_root, cfg.metrics_subdir))
    save_steps_set = set(_parse_save_steps(cfg, divisor_default=False))
    nrow = max(1, int(math.sqrt(batch_size)))
    prob_frames: list[str] = []
    binary_frames: list[str] = []
    energy_buf = torch.empty(cfg.gen_gibbs_steps, batch_size, device=device) if cfg.mcmc_diagnostics else None
    step_iter = tqdm(range(1, cfg.gen_gibbs_steps + 1), total=cfg.gen_gibbs_steps, desc="Gibbs sampling", unit="step")
    timer = Timer()
    for step in step_iter:
        timer.start()
        v, h1, h2, p_v, _, _ = dbm.gibbs_step(v, h1, h2)
        timer.stop()
        if energy_buf is not None:
            inter_vh1 = torch.sum((v @ dbm.W1) * h1, dim=1)
            inter_h1h2 = torch.sum((h1 @ dbm.W2) * h2, dim=1)
            v_bias = torch.sum(v * dbm.bv, dim=1)
            h1_bias = torch.sum(h1 * dbm.bh1, dim=1)
            h2_bias = torch.sum(h2 * dbm.bh2, dim=1)
            energy_buf[step - 1] = -(inter_vh1 + inter_h1h2 + v_bias + h1_bias + h2_bias)
        if step in save_steps_set:
            p_v_cpu, v_cpu = p_v.detach().cpu(), v.detach().cpu()
            prob_path = os.path.join(grid_prob_dir, f"step{step:04d}_grid.png")
            save_grid(p_v_cpu, prob_path, nrow=nrow, title=f"DBM sample prob (step {step})", hw=(cfg.size_h, cfg.size_w))
            binary_path = os.path.join(grid_bin_dir, f"step{step:04d}_grid.png")
            save_grid(v_cpu, binary_path, nrow=nrow, title=f"DBM sample binary (step {step})", hw=(cfg.size_h, cfg.size_w))
            prob_frames.append(prob_path)
            binary_frames.append(binary_path)
    total_time = timer.sum()
    rate = cfg.gen_gibbs_steps / max(1e-8, total_time)
    print(f"\nSampling done: {cfg.gen_gibbs_steps} Gibbs steps in {timer.format_time(total_time, precision=3)} ({rate:.1f} steps/s)")
    if energy_buf is not None and cfg.mcmc_diagnostics:
        print("\n--- MCMC Convergence Diagnostics ---")
        _save_mcmc_convergence_plots(energy_buf, list(range(1, cfg.gen_gibbs_steps + 1)), metrics_dir, prefix="dbm")
        final_metrics = _compute_mcmc_metrics(energy_buf[cfg.gen_gibbs_steps // 2:], split_chains=True, n_chains=min(4, batch_size // 2))
        status = "✓ converged" if final_metrics["r_hat"] < 1.1 else "✗ NOT converged — consider more Gibbs steps"
        print(f"  R-hat (Gelman-Rubin):   {final_metrics['r_hat']:.4f}  {status}")
        print(f"  ESS (bulk, approx):     {final_metrics['ess_bulk']:.1f}")
        print(f"  Lag-1 autocorrelation:  {final_metrics['autocorr_lag1']:.4f}")
        print(f"  Diagnostic plots saved to: {metrics_dir}/")
    if cfg.save_gif and prob_frames:
        prob_images: list[ArrayLike] = [imageio_v3.imread(path) for path in prob_frames]
        binary_images: list[ArrayLike] = [imageio_v3.imread(path) for path in binary_frames]
        prob_gif_path = os.path.join(gif_dir, "sampling_prob.gif")
        binary_gif_path = os.path.join(gif_dir, "sampling_binary.gif")
        imageio.mimsave(prob_gif_path, prob_images, fps=cfg.gif_fps, loop=0)
        imageio.mimsave(binary_gif_path, binary_images, fps=cfg.gif_fps, loop=0)
        print(f"GIF (prob):   {prob_gif_path}")
        print(f"GIF (binary): {binary_gif_path}")
    print(f"Sampling outputs → {sample_root}\n")
