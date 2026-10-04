# Cloud GPU Tools

Cloud-specific scripts live here. [setup_cloud_gpu.sh](setup_cloud_gpu.sh)
prepares this repository on an existing Ubuntu x86_64 RTX 5080/5090 host with
a working NVIDIA driver. It installs missing `curl`, `git`, `rsync`, and `tmux`,
installs uv and the project's managed Python, synchronizes locked dependencies,
and runs the [shared runtime check](../pytorch_test.py).

Instance provisioning, NVIDIA driver installation, data preparation, and model
training are separate steps. Model objectives, training loops, and evaluation
stay in the numbered lessons; these tools do not select hyperparameters or
automatically extend a training budget.

## 1. Prepare the project environment

Place the repository on the host's persistent disk and run commands from its
root. Keep `.python-version`, `pyproject.toml`, and `uv.lock` together; recreate
`.venv` on the host instead of copying a local environment. The setup script
also locates the repository correctly when invoked from another directory.

Preview the plan on a local machine, then run it on the selected cloud host:

```bash
bash tool_scripts/cloud_gpu/setup_cloud_gpu.sh --gpu 5090 --extra diffusion --dry-run

export PATH="$HOME/.local/bin:$PATH"
bash tool_scripts/cloud_gpu/setup_cloud_gpu.sh --gpu 5090 --extra diffusion
```

`--dry-run` checks argument syntax and required project files, prints the exact
uv commands, and exits without hardware checks, downloads, or filesystem writes.
An actual run checks Linux/x86_64 and the first GPU before installation;
`--gpu 5080` or `--gpu 5090` enforces the expected model. Omitting it detects
either supported cloud GPU. Other GPU models require updating the setup and
shared runtime checks before use.

Select optional dependencies from [pyproject.toml](../../pyproject.toml):

```bash
# Download CelebA/glasses and run diffusion metrics in the same environment.
bash tool_scripts/cloud_gpu/setup_cloud_gpu.sh --gpu 5090 --extra celeba --extra diffusion
```

`--extra NAME` is repeatable; uv validates the names during synchronization.
The `core` profile is the default and excludes development tools. `--profile
examples` adds the examples extra without development tools; `--profile full`
installs every extra and the development tools. Explicit extras can be added
to any profile. Repeat the complete desired selection on each setup run,
because synchronization can remove packages outside the selected dependency set.

The default managed Python and uv cache directory is `.cache/cloud-gpu/` under
the repository. Use `--state-dir PATH` for another persistent location; relative
paths are resolved from the repository root. Keep this directory while using
the environment, since `.venv` uses its managed Python installation.
`--skip-system-packages` requires the host commands to be present already and
skips APT installation. Python and uv versions are managed through the project
and installer; this script does not install a CUDA Toolkit or replace drivers.

## 2. Prepare or transfer data

Use the [dataset tools](../README.md#dataset-profiles). For the diffusion and
modern generation series, copy the complete `data/food101-256/` directory,
including its manifest, or prepare it on the host:

```bash
uv run --locked --no-sync python tool_scripts/download_dataset.py --dataset food101
```

The prepared cache is sufficient for training; the original Food-101 archive
is not needed there. Set each lesson's `--data-dir` when storing the cache
elsewhere. VAE and GAN lessons use their own documented datasets and CLI options.

## 3. Validate, train, and evaluate

The setup's runtime check reports CUDA availability and BF16 support. Follow it
with a short real-data run to exercise model forward/backward passes and data
loading. For example, check DDPM without periodic quality evaluation:

```bash
uv run --locked --no-sync python genai/3.0_diffusion_model/1.0_ddpm.py \
  --precision bf16 --max-steps 5 --sample-every 0 --eval-every 0 \
  --output-dir output/diffusion/ddpm-smoke
```

Then start a persistent terminal with `tmux new -s training`, and launch the
chosen lesson with an explicit training budget and a fresh experiment directory:

```bash
uv run --locked --no-sync python genai/3.0_diffusion_model/1.0_ddpm.py \
  --precision bf16 --epochs 100 --output-dir output/diffusion/ddpm-run01
```

Choose the budget and batch size for the model; the example is not a convergence
claim. Keep outputs on persistent storage. Detach with `Ctrl-b`, then `d`, and
reconnect with `tmux attach -t training`. The terminal session survives an SSH
disconnect, but training still depends on the instance remaining alive.

Use the corresponding independent evaluation entry after training. The
[foundation roadmap](../../genai/3.0_diffusion_model/0.0-ROADMAP.md) and
[modern roadmap](../../genai/4.0_modern_visual_generation/0.0-ROADMAP.md) describe
metrics, validation/test usage, and codec/teacher prerequisites. Use each
series' documented resume option; foundation and modern lessons accept
`--resume-from PATH`. A short smoke checkpoint is only a runtime check.

## 4. Preserve the run

Record the source commit, launch arguments, GPU/runtime information, and data
manifest alongside checkpoints, training logs, evaluation JSON, and image grids.
Keep the exact frozen codec or teacher checkpoint when a model depends on it.
Copy required artifacts back to persistent storage or the local project and
verify the transfer before releasing the instance. This directory's scripts
do not stop instances or manage provider billing.
