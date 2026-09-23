# Deep Learning with PyTorch

An educational PyTorch repository for introductory deep learning and generative
AI. Lessons are organized in a suggested reading order; shared utilities and
operational workflows live outside the lesson directories so their behavior
remains explicit and reusable.

## Quick start

The project uses a locked, project-local uv environment:

```bash
uv sync --locked
uv run --locked --no-sync python tool_scripts/pytorch_test.py
```

The second command is a read-only check of the installed PyTorch/CUDA runtime.
Python is pinned by [`.python-version`](.python-version), direct dependencies
by [`pyproject.toml`](pyproject.toml), and exact resolution by
[`uv.lock`](uv.lock).

## Repository guide

| Goal | Authoritative entry point |
|---|---|
| Follow the core deep-learning lessons | [`deep_learning/`](deep_learning/) |
| Follow the generative-model lessons | [`genai/`](genai/), then the [GAN](genai/1.0_generative_adversarial_network/0.0-ROADMAP.md), [VAE](genai/2.0_variational_autoencoder/0.0-ROADMAP.md), or [diffusion](genai/3.0_diffusion_model/0.0-ROADMAP.md) roadmap |
| Find or modify shared utilities | [`dl_utils/README.md`](dl_utils/README.md) |
| Prepare datasets or run repository tools | [`tool_scripts/README.md`](tool_scripts/README.md) |
| Check supported GPUs and prepare a cloud host | [`tool_scripts/README.md`](tool_scripts/README.md#gpu-targets) |
| Inspect dependencies and tooling | [`pyproject.toml`](pyproject.toml) |

Lesson directories and files are numbered in suggested reading order; their
numbers do not correspond directly to book chapters.

## Development checks

Run checks from the locked project environment:

```bash
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv run --locked --no-sync pyright
```

## References

- [Dive into Deep Learning](https://d2l.ai) — Aston Zhang et al.
- [Deep Learning](https://www.deeplearningbook.org) — Ian Goodfellow, Yoshua
  Bengio, and Aaron Courville.
- [Deep Generative Modeling](https://link.springer.com/book/10.1007/978-3-031-64087-2) —
  Jakub M. Tomczak.
- [Learn Generative AI with PyTorch](https://www.manning.com/books/learn-generative-ai-with-pytorch) —
  Mark Liu.
