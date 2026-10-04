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
| [data/](data/) | Downloads, datasets, image preparation, and loaders | [datasets/](data/datasets/) and [vision.py](data/vision.py) |
| [diffusion/](diffusion/), [ebm/](ebm/), [gan/](gan/), [vae/](vae/) | Model-family building blocks | The importing lesson and focused source module |
| [inference/](inference/) | Model-independent batched inference and fixed class-latent grids | [batching.py](inference/batching.py) and [latent_sampling.py](inference/latent_sampling.py) |
| [evaluation/](evaluation/) | Supervised evaluation, image features, distribution metrics, and reconstruction metrics | [supervised.py](evaluation/supervised.py), [image_features.py](evaluation/image_features.py), [distribution_metrics.py](evaluation/distribution_metrics.py), and [reconstruction_metrics.py](evaluation/reconstruction_metrics.py) |
| [runtime/](runtime/), [training/](training/) | Devices, precision, checkpoints, metrics, and optimization | [precision.py](training/precision.py), [checkpoints.py](training/checkpoints.py), [metrics.py](training/metrics.py), and [history.py](training/history.py) |
| [filesystem/](filesystem/), [plot/](plot/) | Project paths, output directories, and figures | [figures.py](plot/figures.py), [curves.py](plot/curves.py), and [images.py](plot/images.py) |

## Design boundaries

- [data/datasets/](data/datasets/) owns dataset readers, labels, preparation
  rules, and the named download catalog. CelebA separates image/attribute
  reading, CPU/CUDA pipelines, and local split preparation. Glasses keeps its
  reviewed correction JSON beside the dataset module and includes it in the
  built package.
  [data/downloads.py](data/downloads.py) owns HTTP/Kaggle downloads and archive
  extraction; [data/preparation.py](data/preparation.py) builds image caches;
  [data/loading.py](data/loading.py) owns loader construction and tensor batches.
  Dataset wrappers retain their shuffle, drop-last, pinning, and worker-sharing
  policies. [data/vision.py](data/vision.py) retains ImageFolder helpers and the
  original D2L/EBM loader imports.

- [runtime/devices.py](runtime/devices.py) owns device selection and explicit
  CUDA backend configuration. [runtime/randomness.py](runtime/randomness.py)
  owns seeding and RNG snapshots; [runtime/timing.py](runtime/timing.py) owns
  the general-purpose timer. Training precision and backward-step policies
  remain in [training/precision.py](training/precision.py), and optimizer
  construction belongs to [training/optimization.py](training/optimization.py).
  Checkpoints retain their existing RNG payload and restoration semantics.
  `training.timing.Timer` remains available for D2L and EBM callers;
  [filesystem](filesystem/__init__.py) exports its two path/directory helpers.

- [training/metrics.py](training/metrics.py) owns scalar accumulators;
  [evaluation/supervised.py](evaluation/supervised.py) owns dataset-level
  accuracy and loss evaluation. [training/history.py](training/history.py)
  aligns metric histories and writes CSV records without running models.
- [plot/curves.py](plot/curves.py) and [plot/images.py](plot/images.py) render
  prepared data. [training/artifacts.py](training/artifacts.py) composes
  history records, inference, and plotting for training outputs, including
  conditional sample grids and optional metric curves. GAN-specific loss
  layouts and BF16 sample grids belong to [gan/artifacts.py](gan/artifacts.py).
  Existing D2L and EBM imports through `training.metrics` and `plot.figures`
  remain available; new callers use the focused modules above.

- Foundation lessons import the [DDPM](diffusion/diffusion_ddpm.py),
  score-SDE, flow-matching, and U-Net modules directly. The
  [diffusion roadmap](../genai/3.0_diffusion_model/0.0-ROADMAP.md) follows the
  128px CelebA main line: discrete denoising, continuous score learning, then
  direct velocity learning. `diffusion/flow_matching.py` owns conditional
  Gaussian paths and Euler/midpoint/Heun integration from noise to data;
  `diffusion/checkpoints.py` keeps score and velocity contracts distinct.
  Improved DDPM, EDM, and DPM solvers serve optional extension lessons.
  `diffusion/lesson_utils.py` shares data, binned losses, and checkpoint
  handling while objectives and optimization remain in scripts;
  `diffusion/quality.py` monitors FID, KID, feature precision/recall, and NFE
  using shared [image features](evaluation/image_features.py) and
  [distribution metrics](evaluation/distribution_metrics.py).
  [gan/continuation.py](gan/continuation.py) retains its CelebA generator evaluation
  protocol and seeded 256D projection; diffusion monitoring retains full
  2048D features, sampling callbacks, and NFE accounting. Sharing primitives
  does not make these evaluation protocols interchangeable.
  [reconstruction_metrics.py](evaluation/reconstruction_metrics.py) provides
  SSIM, including for the latent-diffusion first stage.
- GANs use one module per algorithm. [progan.py](gan/progan.py),
  [stylegan.py](gan/stylegan.py), and [stylegan2.py](gan/stylegan2.py) keep their
  model definitions and continuation adapters together; StyleGAN2 also keeps
  its fixed-resolution options, epoch schedules, and path-length penalty there.
  Existing model imports such as `from dl_utils.gan.progan import ProGANGenerator`
  retain their meaning. Lesson scripts still own experiment budgets,
  objectives, regularization timing, and update order.
- [gan/training.py](gan/training.py) groups shared GAN objectives and update
  steps, progressive schedules, conditional hinge epochs, EMA buffer calibration,
  latent mixing, and R1. It also owns CelebA run setup, model/EMA initialization,
  checkpoint restoration, and kimg histories. It depends on shared utilities,
  without importing concrete GAN model modules.
- [gan/continuation.py](gan/continuation.py) owns bounded continuation,
  checkpoint recovery, the CelebA generator evaluation protocol, and candidate
  selection. Each algorithm module supplies explicit constructors, configuration,
  checkpoint compatibility, sampling settings, and a callback to the lesson
  trainer. Existing `--refine-*` flags and checkpoint keys, including StyleGAN2's
  historical `path_mean` field, retain their meaning.
- [gan/stylegan_layers.py](gan/stylegan_layers.py) shares numerical layers;
  [gan/artifacts.py](gan/artifacts.py) renders GAN loss layouts and fixed-latent
  sample grids. These, `training.py`, and `continuation.py` are the four shared
  support modules alongside the algorithm files.
- [inference/](inference/) shares bounded tensor inference and paired
  class-latent grids across GANs, VAEs, and plotting helpers without depending
  on a model family.
- [data/datasets/celeba/](data/datasets/celeba/) loads aligned faces using the official
  partitions and optional binary attributes. Conditional GANs use Smiling
  labels with 64x64 images.
- [vae/vae.py](vae/vae.py) implements the 256x256 introductory VAE and the
  comparable standard/beta-VAE training path using the RGB encoder/decoder in
  [vae/vae_common.py](vae/vae_common.py). Focused modules cover 256x256 RGB
  hierarchical VAEs on glasses-256 plus reusable discrete-tokenizer, token-prior,
  and perceptual-autoencoder blocks for the 256x256 glasses-256 lessons.
- [diffusion/kl_autoencoder.py](diffusion/kl_autoencoder.py) owns the KL
  perceptual autoencoder and frozen VGG feature loss for latent diffusion.
  Its f=8 default maps 128px RGB to 4x16x16 continuous latents and decodes
  back to 128px. It reuses the RGB encoder/decoder in
  [vae/perceptual_autoencoder.py](vae/perceptual_autoencoder.py); the KL lesson
  also reuses that module's PatchGAN and adaptive adversarial weight.
- [vae/discrete_workflow.py](vae/discrete_workflow.py) composes existing
  checkpoint, weight-loading, DataLoader, and CSV helpers for VQ-VAE/FSQ/VQGAN.
  It owns glasses-256 loading, recorded training subsets, token-usage statistics,
  shared monitoring and reconstruction previews, snapshot-bound token caches,
  epoch recovery, and best/last artifacts. Token caches also track ordered image
  paths, labels, file sizes, and nanosecond modification times; datasets without
  inspectable files are re-encoded. Image and token training loaders retain the
  final partial batch.
  [vae/pixelcnn_training.py](vae/pixelcnn_training.py) owns the shared frozen-token
  PixelCNN training, evaluation, and image-sampling helpers for VQ-VAE/FSQ,
  using the infrastructure in `discrete_workflow.py`. Read its epoch,
  evaluation, and sampling helpers before the complete `train_pixelcnn_prior`
  workflow.
  Training monitors still use token statistics, PSNR, and token-rate estimates;
  the 6.2/7.1 evaluation entries save only reconstruction and generation grids.
  VQ commitment loss is returned once as the quantizer loss; the diagnostics
  dictionary retains its latent quantization MSE.
  Tokenizer optimization and VQGAN objectives remain explicit in the lessons.
  Model modules do not import the workflow. Priors consume cached token grids.
  [vae/token_priors.py](vae/token_priors.py) owns both priors and their direct
  autoregressive samplers. PixelCNN uses one stream of A/B masked convolutions
  with ReLU and recomputes full-grid logits for each sampled position. Its
  receptive field has a blind spot. The Transformer recomputes the full token
  prefix at each step; its layers are independently initialized. Transformer
  sampling temporarily disables dropout and restores the previous mode;
  training and likelihood evaluation use the parallel forward path.
  [vae/quantization.py](vae/quantization.py) supplies the shared quantizers.
  Encoders require at least two downsampling
  steps and image dimensions divisible by their compression factor.
  The VQ-VAE/FSQ PixelCNN prior supports only unconditional generation.
  The Transformer prior supports class conditioning with a positive `num_classes`.
  Models, lesson entries, and weight loaders share
  `TOKENIZER_DOWNSAMPLE_STEPS=4` for 256x256 inputs and outputs; weight loading
  enforces this configuration. They share a 16x16
  grid and 512-code vocabulary. VQ-VAE/FSQ ignore the G/NoG folder labels;
  VQGAN explicitly enables its two-class Transformer prior. Data loading,
  checkpoints, token caches, and evaluation record this conditioning contract.
  VQGAN retains its separate backbone, objective, prior, and training budget.
- [vae/conditional_vae.py](vae/conditional_vae.py) owns the 256x256 glasses
  CVAE, conditional objective, cache metadata contract, and evaluation figures.
  Its RGB backbone is shared with standard VAE, beta-VAE, and HVAE/Ladder through
  [vae/vae_common.py](vae/vae_common.py); all use the existing glasses-256 cache.
  Its lesson entries reuse [data/loading.py](data/loading.py) and the shared
  model-weight checkpoint helpers. Sampling reuses the fixed class-noise grid
  and bounded inference in [inference/](inference/), Gaussian
  reparameterization in [vae/vae_common.py](vae/vae_common.py), and labeled
  grids in [plot/images.py](plot/images.py). The scripts retain optimization
  order, hyperparameters, and artifact timing.
- [training/checkpoints.py](training/checkpoints.py) owns serialization and
  state restoration; [training/session.py](training/session.py) manages output
  lifecycles without owning optimization loops.
- [training/ema.py](training/ema.py) shares parameter averaging across model
  families; [training/validation.py](training/validation.py) checks finite
  model, optimizer, and auxiliary state. Worker-count resolution belongs to
  [data/loading.py](data/loading.py).

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
