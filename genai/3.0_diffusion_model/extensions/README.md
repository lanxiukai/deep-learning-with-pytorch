# Optional Diffusion Extensions

These existing lessons sit after the [DDPM, Score-SDE, and CFM foundations](../0.0-ROADMAP.md).
They are retained for later variance, preconditioning, solver, and latent-model
study; they are not prerequisites for the reading guide's current 4.4–4.6 units.

| Entry point | Teaching increment | Former location in the parent directory |
|---|---|---|
| [improved_ddpm.py](improved_ddpm.py) | Cosine schedule, learned variance, hybrid epsilon MSE + VLB | `1.1_improved_ddpm.py` |
| [edm.py](edm.py) | Denoiser preconditioning and weighted log-normal-noise training | `2.1_edm.py` |
| [solver_comparison.py](solver_comparison.py) | Freeze a diffusion checkpoint; compare Euler/Heun and DPM-Solver families | `2.2_solver_comparison.py` |
| [kl_autoencoder.py](kl_autoencoder.py) | Perceptual/adversarial KL first stage with an f=8 spatial bottleneck | `3.0_kl_autoencoder.py` |
| [latent_diffusion.py](latent_diffusion.py) | Freeze the first stage; train DDPM on scaled latents; optional Smiling/CFG | `3.1_latent_diffusion.py` |

All paths to data, outputs, and default checkpoints remain rooted at the
repository. These entries keep 128px-or-larger RGB and the shared quality
protocol. For 128px input, the approximately 4.5M-parameter KL stage produces
`4x16x16` latents. The roughly 31.7M-parameter latent denoiser operates there;
decoding restores 128px RGB. The first-stage identity and latent scale must
match the LDM checkpoint. Latents are not clipped to a pixel range.

The KL first-stage model and VGG feature loss live in
[dl_utils/diffusion/kl_autoencoder.py](../../../dl_utils/diffusion/kl_autoencoder.py).
Shared RGB blocks, PatchGAN, and adaptive adversarial weighting are reused
from [the perceptual image module](../../../dl_utils/vae/perceptual_autoencoder.py).
The first-stage and latent-diffusion scripts import the KL model from its
diffusion module.

## Algorithm boundaries

- Improved DDPM maps its raw variance head with `r=(v+1)/2`, without a sigmoid
  or clamp. Its hybrid loss uses `MSE + lambda*T*VLB_t`, detached mean, finite
  first-step variance, and an 8-bit discretized Gaussian endpoint likelihood.
  Resized pixels are quantized to that bin grid. The logged trainable VLB
  estimate excludes the constant prior KL and is not a full likelihood.
  Ancestral sampling uses learned variance; DDIM ignores that head. Compare
  linear schedules first to isolate variance learning, then cosine separately.
- EDM separates training noise from sampling endpoints. Its preconditioner
  exposes input, output, skip, and noise coefficients. Euler/Heun uses a
  rho-shaped sigma grid; the final interval to zero uses Euler without
  evaluating a singular zero-noise network input.
- The VP solver comparison uses increasing `lambda=log(alpha/sigma)`.
  Discrete models interpolate log-alpha and use fractional U-Net indices;
  continuous VP models invert their time path. DPM-Solver predicts epsilon;
  DPM-Solver++ converts to x0. The 2M methods reuse the previous prediction
  after a first-order start. All use a finite endpoint and one additional
  x0 prediction. Third order, adaptive stepping, and distillation are omitted.
- The score sampler module retains Langevin predictor-corrector and VE
  annealed Langevin with the explicit rule `eta(t)=step_size*sigma(t)^2`.
  These are optional sampling experiments, not the foundation training task
  or an exact NCSN reproduction. The current 2.0 lesson trains continuous time.
- The autoencoder reports reconstruction L1/PSNR/SSIM/perceptual error and KL.
  Its standard-normal decode is an interface diagnostic. Free-generation
  FID/KID/precision/recall resumes after learning the latent prior. CFG metrics
  use the training Smiling label prior and record guidance strength.

## Future commands

```bash
uv run --locked --no-sync python genai/3.0_diffusion_model/extensions/improved_ddpm.py
uv run --locked --no-sync python genai/3.0_diffusion_model/extensions/edm.py
uv run --locked --no-sync python genai/3.0_diffusion_model/extensions/solver_comparison.py \
  --checkpoint output/diffusion/ddpm/latest.pth --steps 25 50
uv run --locked --no-sync python genai/3.0_diffusion_model/extensions/solver_comparison.py \
  --checkpoint output/diffusion/edm/latest.pth --solvers euler heun
uv run --locked --no-sync python genai/3.0_diffusion_model/extensions/kl_autoencoder.py
uv run --locked --no-sync python genai/3.0_diffusion_model/extensions/latent_diffusion.py
```

Use [1.1 DDIM](../1.1_ddim_sampling.py) with an Improved DDPM checkpoint for
ancestral/DDIM comparisons. Use [2.1 SDE/ODE](../2.1_sde_ode_sampling.py) for
the basic continuous-time comparison and [3.1 Flow sampling](../3.1_flow_sampling.py)
for CFM checkpoints. LDM sampling uses `--mode sample` with the matching
`--autoencoder-checkpoint`. No command is launched by importing a script.

Primary references: [Improved DDPM](https://github.com/openai/improved-diffusion),
[EDM](https://github.com/NVlabs/edm),
[DPM-Solver](https://github.com/LuChengTHU/dpm-solver), and
[Latent Diffusion](https://arxiv.org/abs/2112.10752).
