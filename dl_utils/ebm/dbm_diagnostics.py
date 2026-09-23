import math
import os

import torch

from dl_utils.plot._backend import pyplot as plt

from ._ebm_types import MCMCMetrics


def _compute_mcmc_metrics(energy_traces: torch.Tensor, *, split_chains: bool = True, n_chains: int = 4) -> MCMCMetrics:
    n_steps, batch_size = energy_traces.shape
    if n_steps < 10 or batch_size < 2:
        return {"r_hat": float("nan"), "ess_bulk": float("nan"), "ess_tail": float("nan"), "autocorr_lag1": float("nan")}
    if batch_size < n_chains:
        n_chains = max(1, batch_size // 2)
    chain_len = batch_size // n_chains
    traces = energy_traces[:, :n_chains * chain_len]
    if split_chains and n_steps >= 4:
        half = n_steps // 2
        reshaped = traces.reshape(n_steps, n_chains, chain_len)
        traces_use = torch.cat([reshaped[:half], reshaped[half:2 * half]], dim=1)
        n_chains_use, chain_len_use = 2 * n_chains, chain_len
    else:
        traces_use = traces.reshape(n_steps, n_chains, chain_len)
        n_chains_use, chain_len_use = n_chains, chain_len
    chain_means = traces_use.mean(dim=0)
    within_variance = traces_use.var(dim=0, unbiased=True).mean()
    between_variance = ((chain_means - chain_means.mean()) ** 2).mean() * traces_use.size(0) / (n_chains_use - 1)
    r_hat = min(torch.sqrt(((traces_use.size(0) - 1) / traces_use.size(0) * within_variance + between_variance) / (within_variance + 1e-12)).item(), 100.0)
    autocorrelations: list[float] = []
    for chain_index in range(n_chains_use):
        for sample_index in range(chain_len_use):
            chain = traces_use[:, chain_index, sample_index]
            centered = chain - chain.mean()
            variance = centered.var()
            if variance >= 1e-10:
                autocorrelations.append(((centered[:-1] * centered[1:]).mean() / variance).item())
    autocorr_lag1 = float(torch.tensor(autocorrelations).mean()) if autocorrelations else float("nan")
    if not math.isnan(autocorr_lag1) and abs(autocorr_lag1) < 1.0:
        total_samples = n_steps * n_chains_use * chain_len_use
        ess_bulk = total_samples * (1 - autocorr_lag1) / (1 + autocorr_lag1) if autocorr_lag1 > -1 else total_samples
    else:
        ess_bulk = float("nan")
    return {"r_hat": r_hat, "ess_bulk": ess_bulk, "ess_tail": float("nan"), "autocorr_lag1": autocorr_lag1}


def _save_mcmc_convergence_plots(energy_traces: torch.Tensor, steps: list[int], out_dir: str, *, prefix: str = "dbm") -> None:
    n_steps, batch_size = energy_traces.shape
    if n_steps < 2:
        return
    energy_np = energy_traces.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n_show = min(8, batch_size)
    rng = torch.Generator().manual_seed(hash(prefix) % (2**31))
    for index in torch.randperm(batch_size, generator=rng)[:n_show].tolist():
        axes[0, 0].plot(steps, energy_np[:, index], alpha=0.6, linewidth=0.5)
    axes[0, 0].set(xlabel="Gibbs step", ylabel="Energy", title=f"{prefix} — Energy traces ({n_show} chains)")
    axes[0, 0].grid(True, alpha=0.3)
    mean_energy, std_energy = energy_np.mean(axis=1), energy_np.std(axis=1)
    axes[0, 1].plot(steps, mean_energy, "b-", linewidth=1.5, label="Mean energy")
    axes[0, 1].fill_between(steps, mean_energy - std_energy, mean_energy + std_energy, alpha=0.2, color="b")
    axes[0, 1].set(xlabel="Gibbs step", ylabel="Energy", title=f"{prefix} — Mean ± 1σ energy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    max_lag = min(100, n_steps - 1)
    centered = mean_energy - mean_energy.mean()
    acf_values = [0.0 if centered.var() < 1e-10 else (centered[lag:] * centered[:-lag]).mean() / centered.var() for lag in range(1, max_lag + 1)]
    axes[1, 0].stem(range(1, max_lag + 1), acf_values, linefmt="gray", markerfmt=".")
    axes[1, 0].axhline(y=0, color="black", linewidth=0.5)
    axes[1, 0].set(xlabel="Lag", ylabel="Autocorrelation", title=f"{prefix} — Energy ACF")
    axes[1, 0].grid(True, alpha=0.3)
    block_size = max(20, n_steps // 20)
    r_hats, centers = [], []
    for start in range(0, n_steps - block_size, block_size // 2):
        block = energy_traces[start:start + block_size]
        if block.size(0) >= 10:
            r_hats.append(_compute_mcmc_metrics(block, split_chains=(block.size(0) >= 20), n_chains=min(4, batch_size // 2))["r_hat"])
            centers.append(start + block_size // 2)
    if r_hats:
        axes[1, 1].plot(centers, r_hats, "r-o", markersize=3, linewidth=1)
        axes[1, 1].axhline(y=1.1, color="gray", linestyle="--", linewidth=1, label="R-hat=1.1 (converged)")
        axes[1, 1].set(xlabel="Gibbs step", ylabel="R-hat", title=f"{prefix} — Running Gelman-Rubin R-hat")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=axes[1, 1].transAxes)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{prefix}_mcmc_diagnostics.png"), dpi=200)
    plt.close(fig)
    final_metrics = _compute_mcmc_metrics(energy_traces[n_steps // 2:], split_chains=True, n_chains=min(4, batch_size // 2))
    summary_path = os.path.join(out_dir, f"{prefix}_mcmc_summary.txt")
    with open(summary_path, "w") as handle:
        handle.write(f"MCMC Convergence Summary for {prefix}\n{'=' * 50}\nTotal Gibbs steps: {n_steps}\nNumber of chains: {batch_size}\n")
        handle.write(f"R-hat (Gelman-Rubin): {final_metrics['r_hat']:.4f}  {'✓ converged' if final_metrics['r_hat'] < 1.1 else '✗ NOT converged'}\nESS (bulk, approx): {final_metrics['ess_bulk']:.1f}\nLag-1 autocorrelation: {final_metrics['autocorr_lag1']:.4f}\n")
        handle.write("\nInterpretation:\n  - R-hat < 1.1  → chains have likely converged (mixed well)\n  - R-hat > 1.1  → need more Gibbs steps or better mixing\n  - Lag-1 AC ≈ 0  → near-independent samples (fast mixing)\n  - Lag-1 AC > 0.9 → highly correlated samples (slow mixing)\n")
    print(f"MCMC summary saved to {summary_path}")
