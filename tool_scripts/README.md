# Tool Scripts

This README is the operating index for repository-level tools: dataset
preparation, cloud GPU entry points, model tuning, environment inspection,
and standalone visualizations. Exact CLI options remain in each script's
`--help` output.

## Running tools

Run commands from the repository root. Prepare the environment through the
[project quick start](../README.md#quick-start), then use these invocation
patterns:

```bash
uv run --locked --no-sync python tool_scripts/SCRIPT.py
bash tool_scripts/SCRIPT.sh --help
```

`pytorch_test.py` is the read-only runtime check used by the quick start.

## Choose a task

| Goal | Start with | Main effect |
|---|---|---|
| Inspect the local PyTorch/CUDA runtime | `pytorch_test.py` | Read-only |
| Download or prepare a lesson dataset | `download_dataset.py` | Downloads data and may build derived caches |
| Create a local visualization | `plot_fashion_mnist.py`, `sgd_animation.py`, or `word_frequency.py` | Downloads data when needed and writes under `output/` |
| Prepare an existing cloud RTX 5080/5090 host | `setup_cloud_gpu.sh` | Installs host prerequisites and synchronizes the project environment |
| Tune the three 128x128 teaching GANs | `tune_teaching_gans.py` | Sequential continuation, validation-driven parameter changes, retained best checkpoints, and final evaluation |

## GPU targets

GPU tools support exactly these targets:

| Target | GPU | Intended host |
|---|---|---|
| `4070ti` | NVIDIA GeForce RTX 4070 Ti | Local workstation |
| `5080` | NVIDIA GeForce RTX 5080 | Cloud host |
| `5090` | NVIDIA GeForce RTX 5090 | Cloud host |

The Python runtime check and tuning tools detect the active GPU
and reject other models. Pass `--gpu 4070ti`, `--gpu 5080`, or `--gpu 5090`
to require a particular target. This selects a hardware check, not a new
model architecture or training schedule.

Check the local runtime:

```bash
uv run --locked --no-sync python tool_scripts/pytorch_test.py --gpu 4070ti
```

On an existing Ubuntu x86-64 cloud host with a working NVIDIA driver, run
the general setup script from the repository root. It accepts only cloud
targets and reads the Python version and dependencies from the project files:

```bash
bash tool_scripts/setup_cloud_gpu.sh --gpu 5080 --profile core --dry-run
bash tool_scripts/setup_cloud_gpu.sh --gpu 5080 --profile core
```

Use `--gpu 5090` on a 5090 host. Setup installs missing host prerequisites,
uv, the managed Python runtime, and locked project dependencies; it does not
provision a provider instance or download datasets. Training and evaluation
run through the Python lessons and tools directly.

## Teaching GAN refinement

The 7.0, 7.1, and 7.2 lessons accept `--refine-from`, `--refine-output`,
`--refine-kimg`, `--refine-learning-rate`, and `--refine-reg-weight`.
Refinement holds resolution at 128x128 and preserves the original outputs.
Use `--refine-resume` with the same configuration to resume a refinement
checkpoint. Starting another attempt from `best.pth` restores both networks,
EMA, and optimizer state, rather than combining unrelated model states.

Accepted model weights, comparison grids, evaluation figures, metrics, and
review records live in `output/gan/{progan,stylegan,stylegan2}/`. Each directory
also retains `selected_checkpoint.pth`, the complete state matching its selected
EMA weights. Use this checkpoint with `--refine-from` to start another attempt;
it is a refinement checkpoint, not a progressive-phase `--resume-from` checkpoint.

Run another automatic sequence from the accepted checkpoints:

```bash
uv run --locked --no-sync python tool_scripts/tune_teaching_gans.py \
  --progan-source output/gan/progan/selected_checkpoint.pth \
  --stylegan-source output/gan/stylegan/selected_checkpoint.pth \
  --stylegan2-source output/gan/stylegan2/selected_checkpoint.pth \
  --output-dir output/gan/teaching-tuning
```

Completed initial training checkpoints at `output/gan/<model>/checkpoints/latest.pth`
can also be passed through the corresponding `--<model>-source` options.
Without an explicit source, the tool prefers the accepted checkpoint when it
exists and otherwise uses the completed initial training checkpoint.
The sequence trains ProGAN, StyleGAN, then StyleGAN2. Each attempt adds
500 kimg and evaluates approximately every 50 kimg. Learning rates and
regularization weights are tried in a short predefined sequence. A KID
improvement of at least 5% permits more training; a plateau changes the
parameter profile. The final profile can continue while gains remain
material. Numerical failure abandons that profile without replacing the
retained candidate. Other execution failures stop with diagnostics and can
be resumed using the original command.

The default screening target is torchvision Inception KID <= 0.035 and
256-dimensional projected Inception Frechet distance <= 45. These are
practical within-project thresholds, not paper scores or a visual-quality
guarantee. Candidate selection also guards against reduced feature variance
and contradictory Frechet regression. Fixed latents, fixed synthesis noise,
and untruncated sampling keep validation comparisons consistent. Parameter
selection uses only the validation split; a different seed and test split
are used after all three searches finish.

`progress.json` records the active model and each decision. `selected/`
contains the retained weights, complete checkpoints, before/after grids,
test metrics, and the 7.3 lesson's style-mixing, noise, truncation, and
interpolation figures. `summary.json` ends at `awaiting_visual_review`:
inspect those grids before claiming that the models are ready for teaching.
Resuming with the same command reuses finished attempts and resumes the
latest unfinished checkpoint. The original `output/gan/*` model weights
are not replaced.

The shared resampling cache supports evaluation followed by backpropagation.
The encoded CelebA loader selects file-system tensor sharing in its workers
for the local Python/CUDA JPEG runtime. This applies to both fresh training
and refinement, without changing the parent process's sharing strategy.

The evaluation and tuning code is retained for future VAE adaptation. The GAN
quality evaluator already uses feature extraction and moment calculations from
`dl_utils.vae.image_quality`; VAE sampling, objectives, and acceptance criteria
still need model-specific adaptation. Smoke runs, temporary benchmarks, logs,
and superseded experiment checkpoints are disposable after accepted artifacts
have been copied and checked.

## Safety boundaries

- Dataset and visualization commands can write under `data/` and `output/`.
- Cloud commands use an instance that already exists. They do not create, stop,
  or destroy provider resources, so billing remains your responsibility.

## Dataset profiles

Base dependencies cover torchvision datasets and the project's small download
registry. Run the downloader without arguments to prepare every dataset in its
documented order. This includes archive extraction and the required CelebA
and glasses derived data:

```bash
uv sync --extra examples --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py
```

The default sequence is `mnist`, `fashion-mnist`, `house-prices`, `time-machine`,
`celeba`, `anime-face`, `glasses`, `airfoil`, `fra-eng`, `pokemon`.

Select one or several datasets by listing them after `--dataset`. Selections
run in the order given, and duplicates are ignored after their first
appearance:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset mnist celeba glasses
```

SN-GAN, SAGAN, BigGAN, VQ-VAE, FSQ, and VQGAN default to aligned CelebA under
`data/celeba`, using its official train and validation partitions. GANs use
64x64 faces; discrete tokenizers retain their 128x128 setup. Smiling labels
condition GANs and the tokenizers' second-stage priors.

Explicit `--dataset all` is equivalent to omitting the option. The downloader
continues through the selected sequence after individual provider failures and
reports all failed datasets at the end. Selecting CelebA always prepares the
black/blond CycleGAN splits; selecting glasses always classifies, corrects, and
builds the 256-pixel cache.
