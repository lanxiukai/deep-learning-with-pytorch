# Tool Scripts

This README is the operating index for repository-level tools: dataset
preparation, cloud GPU entry points, environment inspection,
and standalone visualizations. Exact CLI options remain in each script's
`--help` output.

## Running tools

Run commands from the repository root. Prepare the environment through the
[project quick start](../README.md#quick-start), then use these invocation
patterns:

```bash
uv run --locked --no-sync python tool_scripts/SCRIPT.py
bash tool_scripts/cloud_gpu/setup_cloud_gpu.sh --help
```

`pytorch_test.py` is the read-only runtime check used by the quick start.

## Choose a task

| Goal | Start with | Main effect |
|---|---|---|
| Inspect the local PyTorch/CUDA runtime | `pytorch_test.py` | Read-only |
| Download or prepare a lesson dataset | `download_dataset.py` | Downloads data and may build derived caches |
| Create a local visualization | `sgd_animation.py` or `word_frequency.py` | Downloads data when needed and writes under `output/` |
| Prepare an existing cloud RTX 5080/5090 host | [Cloud GPU tools](cloud_gpu/README.md) | Installs host prerequisites and synchronizes selected project dependencies |

## GPU targets

The shared runtime check supports RTX 4070 Ti locally and RTX 5080/5090 on
cloud hosts. It reports the PyTorch/CUDA runtime, GPU model, and BF16 support:

```bash
uv run --locked --no-sync python tool_scripts/pytorch_test.py --gpu 4070ti
```

Cloud-specific setup and future cloud utilities belong in
[cloud_gpu/](cloud_gpu/README.md). Its guide covers environment installation,
optional dependencies, prepared datasets, short validation runs, training,
checkpoint recovery, and result transfer. Model-specific training and evaluation
remain in the numbered lesson directories. The
[GAN roadmap](../genai/1.0_generative_adversarial_network/0.0-ROADMAP.md)
describes direct refinement of the style-based GAN lessons.

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
`celeba`, `anime-face`, `glasses`, `airfoil`, `fra-eng`, `pokemon`, `food101`.

Select one or several datasets by listing them after `--dataset`. Selections
run in the order given, and duplicates are ignored after their first
appearance:

```bash
uv sync --no-dev --extra celeba --locked
uv run --locked --no-sync python tool_scripts/download_dataset.py \
  --dataset mnist celeba glasses
```

SN-GAN, SAGAN, and BigGAN default to aligned CelebA under `data/celeba`,
using its official train and validation partitions, 64x64 faces, and Smiling
conditioning. VQ-VAE, FSQ, and VQGAN use 256x256 RGB images from
`data/glasses-256`. VQ-VAE and FSQ use unconditional token priors; VQGAN's
Transformer prior conditions on G=0 (with glasses) and NoG=1 (without glasses).
Their recorded training subsets provide diagnostics; the glasses cache has
no independent test split.

Explicit `--dataset all` is equivalent to omitting the option. The downloader
continues through the selected sequence after individual provider failures and
reports all failed datasets at the end. Selecting CelebA always prepares the
black/blond CycleGAN splits; selecting glasses always classifies, corrects, and
builds the 256-pixel cache.

### Food-101 for diffusion and modern generation

Download and preprocess the shared dataset once:

```bash
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset food101
```

This keeps the official archive and extracted images in `data/food101/` and
writes 101,000 RGB PNG images to `data/food101-256/images/<class>/`. Preparation
uses bicubic short-edge resizing followed by a 256x256 center crop. Lossless
PNG preserves the resized pixels without a second JPEG compression. Repeating
the command verifies existing cache images and prepares missing ones; individual
images and the completed manifest are published atomically.

`data/food101-256/diffusion_manifest.json` retains the official image IDs and
alphabetical 101-class mapping. Split seed 42 gives 70,700 training, 5,050
validation, and 25,250 official test images. All training and evaluation
entries in the [foundation](../genai/3.0_diffusion_model/0.0-ROADMAP.md) and
[modern](../genai/4.0_modern_visual_generation/0.0-ROADMAP.md) series default to
this prepared directory. They read the cached 256x256 images directly; training
adds horizontal flips and normalization. SR conditions and explicit 128px
experiments downsample the same cached images.

For an existing download, custom locations, or a different worker count, use
the shared [preparation lesson](../genai/3.0_diffusion_model/0.1_prepare_data.py):

```bash
uv run --locked --no-sync python genai/3.0_diffusion_model/0.1_prepare_data.py \
  --source-dir data/food101 --data-dir data/food101-256 --workers 8
```

Add `--download` if the source is missing. Only the prepared directory is needed
on the training host; pass its location through `--data-dir` in either series.
