"""Small utilities; targets and optimizer updates stay in each lesson script."""

import argparse
import json
import os
import random
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root

PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data/food101-256"
OUTPUT_ROOT = Path(os.environ.get("DL_OUTPUT_ROOT", PROJECT_ROOT / "output/diffusion"))


def training_parser(name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / name)
    parser.add_argument("--image-size", type=int, choices=(128, 256), default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[64, 128, 256, 384]
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--max-steps", type=int, help="Stop after this many total optimizer steps."
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Epoch interval; 0 disables previews.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Epoch interval for validation KID; 0 disables.",
    )
    parser.add_argument("--eval-examples", type=int, default=2020)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--autoencoder-checkpoint", type=Path)
    return parser


def setup(args):
    args.run_started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if (
        args.epochs < 1
        or args.batch_size < 1
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        raise ValueError("Epochs, batch size and max steps must be positive.")
    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.resume_from
    ):
        raise FileExistsError(
            "Use a new --output-dir or explicitly --resume-from; runs are not erased."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return torch.device(args.device)


def autocast(args):
    return (
        torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )


def append_record(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def preview(path, images, nrow=8):
    if not torch.isfinite(images).all():
        raise FloatingPointError("Non-finite generated images.")
    save_image(images.detach().float().cpu().add(1).div(2).clamp(0, 1), path, nrow=nrow)


class BinnedLoss:
    def __init__(self):
        self.total = torch.zeros(3, dtype=torch.float64)
        self.count = torch.zeros(3, dtype=torch.long)

    def update(self, loss, coordinate):
        buckets = (coordinate.detach().cpu().float() * 3).long().clamp(0, 2)
        values = loss.detach().cpu().double()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite training objective.")
        self.total.scatter_add_(0, buckets, values)
        self.count.scatter_add_(0, buckets, torch.ones_like(buckets))

    def result(self):
        return {
            "loss": (self.total.sum() / self.count.sum()).item(),
            **{
                f"bin_{i}_loss": (self.total[i] / self.count[i]).item()
                for i in range(3)
                if self.count[i]
            },
        }


def record_epoch(args, epoch, step, metrics):
    record = {
        "epoch": epoch,
        "step": step,
        "run_wall_seconds_including_monitoring": time.perf_counter() - args.run_started,
        **metrics,
    }
    append_record(args.output_dir / "training.jsonl", record)
    print(json.dumps(record))
    import matplotlib.pyplot as plt

    rows = [
        json.loads(line)
        for line in (args.output_dir / "training.jsonl").read_text().splitlines()
    ]
    names = [
        k
        for k in metrics
        if isinstance(metrics[k], (int, float)) and not k.startswith("bin_")
    ]
    if names:
        figure, axes = plt.subplots(
            len(names), 1, figsize=(7, 2.3 * len(names)), squeeze=False
        )
        for axis, name in zip(axes[:, 0], names, strict=True):
            axis.plot(
                [r["step"] for r in rows], [r.get(name, float("nan")) for r in rows]
            )
            axis.set(xlabel="Optimizer updates", ylabel=name)
        figure.tight_layout()
        figure.savefig(args.output_dir / "training_curves.png")
        plt.close(figure)


def save_training(path, model, ema, optimizers, epoch, step, metadata, **extra):
    numpy_state = np.random.get_state()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 4,
            "checkpoint_id": str(uuid.uuid4()),
            **metadata,
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            **extra,
        },
        temporary,
    )
    temporary.replace(path)


def resume_training(args, model, ema, optimizers, metadata):
    metadata["training_config"] = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "resume_policy": "next epoch; a max-steps partial epoch is not replayed",
    }
    if args.resume_from is None:
        return 1, 0, {}
    state = torch.load(args.resume_from, map_location="cpu", weights_only=True)
    for key in ("format_version", "kind", "model_config", "data_config", "algorithm"):
        expected = 4 if key == "format_version" else metadata[key]
        actual = state.get(key)
        if key == "algorithm":
            locations = {"codec_checkpoint", "teacher_checkpoint"}
            expected = {k: v for k, v in expected.items() if k not in locations}
            actual = {k: v for k, v in actual.items() if k not in locations}
        if actual != expected:
            raise ValueError(f"Checkpoint {key} differs from the current run.")
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    for optimizer, saved in zip(optimizers, state["optimizers"], strict=True):
        optimizer.load_state_dict(saved)
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    s = state["numpy_rng"]
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), *s[2:]))
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["epoch"] + 1, state["step"], state
