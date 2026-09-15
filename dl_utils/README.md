# `dl_utils`

`dl_utils` is the internal utility package shared by this repository's
PyTorch lessons. It provides reusable data, model, runtime, checkpoint, and
plotting primitives; lesson scripts remain the authoritative place for
training loops, experiment budgets, and model-specific objectives.

## Environment

This package is installed in editable mode by the repository's
[locked project setup](../README.md#quick-start). It does not own a separate
environment or dependency workflow.

## Package map

| Area | Responsibility | Start with |
|---|---|---|
| [d2l/](d2l/) | D2L-style textbook helpers | The relevant lesson call site |
| [data/](data/) | Downloads, datasets, image preparation, and loaders | [celeba.py](data/celeba.py) and [vision.py](data/vision.py) |
| [diffusion/](diffusion/), [ebm/](ebm/), [gan/](gan/), [vae/](vae/) | Model-family building blocks | The importing lesson and focused source module |
| [runtime/](runtime/), [training/](training/) | Devices, precision, checkpoints, metrics, and optimization | [accelerator.py](training/accelerator.py), [checkpoints.py](training/checkpoints.py), and [metrics.py](training/metrics.py) |
| [filesystem/](filesystem/), [plot/](plot/) | Project paths, output directories, and figures | [figures.py](plot/figures.py) and [images.py](plot/images.py) |

## Design boundaries

- [diffusion/ddpm.py](diffusion/ddpm.py) is a compatibility facade. New
  foundation lessons import the DDPM, score-SDE, flow-matching, and U-Net
  modules directly. The
  [diffusion roadmap](../genai/3.0_diffusion_model/0.0-ROADMAP.md) follows the
  128px CelebA main line: discrete denoising, continuous score learning, then
  direct velocity learning. `diffusion/flow_matching.py` owns conditional
  Gaussian paths and Euler/midpoint/Heun integration from noise to data;
  `diffusion/checkpoints.py` keeps score and velocity contracts distinct.
  Improved DDPM, EDM, and DPM solvers serve optional extension lessons.
  `diffusion/lesson_utils.py` shares data, binned losses, and checkpoint
  handling while objectives and optimization remain in scripts;
  `diffusion/quality.py` monitors FID, KID, feature precision/recall, and NFE
  using the existing Inception/MMD/Fréchet primitives.
- [gan/training.py](gan/training.py) owns shared BF16 runtime selection, data
  access, output paths, EMA setup, checkpoints, and sample artifacts for the
  ProGAN-to-StyleGAN2 sequence. Those lesson scripts retain model schedules,
  objectives, regularization, and update order.
- [gan/conditional_training.py](gan/conditional_training.py) owns the repeated
  conditional hinge epoch used by SN-GAN, SAGAN, and BigGAN. Their lesson
  scripts retain lesson-specific hyperparameters, update ratios, regularization,
  EMA, checkpoints, and artifacts.
- [gan/inference.py](gan/inference.py) shares paired class sample grids,
  bounded generation, and EMA normalization-buffer calibration.
- [data/celeba.py](data/celeba.py) loads aligned faces using the official
  partitions and optional binary attributes. Conditional GANs use Smiling
  labels with 64x64 images; the discrete-tokenizer lessons use 128x128 images
  and pass labels only to their second-stage priors.
- [vae/vae.py](vae/vae.py) preserves the 256x256 introductory VAE and the
  comparable standard/beta-VAE training path. Focused modules cover compact
  32x32 hierarchical VAEs plus reusable discrete-tokenizer, token-prior, and
  perceptual-autoencoder blocks for the 128x128 CelebA lessons.
  The KL perceptual autoencoder also supports an f=8, 128px continuous first
  stage for latent diffusion; its 16x16 latent resolution is distinct from the
  decoded RGB image size. The Inception extractor supports both its existing
  projected features and unprojected 2048D features.
- [vae/conditional_vae.py](vae/conditional_vae.py) owns the 256x256 glasses
  CVAE, conditional objective, cache metadata contract, and evaluation figures.
  Its lesson entries reuse [data/loading.py](data/loading.py) and the shared
  model-weight checkpoint helpers. Sampling reuses the fixed class-noise grid
  and bounded inference in [gan/inference.py](gan/inference.py), Gaussian
  reparameterization in [vae/vae_common.py](vae/vae_common.py), and labeled
  grids in [plot/images.py](plot/images.py). The scripts retain optimization
  order, hyperparameters, and artifact timing.
- [training/checkpoints.py](training/checkpoints.py) owns serialization and
  state restoration; [training/session.py](training/session.py) manages output
  lifecycles without owning optimization loops.

## API and side effects

- Source files are canonical. Module docstrings explain purpose, and `__all__`
  identifies explicitly exported names; unprefixed names without it may still
  be lesson-facing helpers.
- Downloads, directory resets, artifact writes, and random-seed operations
  have external effects. Keep them explicit in a lesson or script.
- GAN artifacts default to `output/gan/` unless `DL_OUTPUT_ROOT` is set.
- Update this README only when the package map or cross-module design
  boundaries change; imports and symbol inventories belong to source and type
  checking.
