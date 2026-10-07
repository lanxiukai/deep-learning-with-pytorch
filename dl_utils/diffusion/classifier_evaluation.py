"""Held-out category accuracy, confusion and noise-stratified guidance diagnostics."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from dl_utils.diffusion.checkpoints import load_model
from dl_utils.diffusion.data import data_config, make_loader
from dl_utils.diffusion.diffusion_score_sde import VPSDE
from dl_utils.diffusion.lesson_utils import DATA_DIR, OUTPUT_ROOT


@torch.no_grad()
def classifier_metrics(model, loader, device, sde=None):
    was_training = model.training
    model.eval()
    results = {}
    for time in [0.001, 0.2, 0.5, 0.9] if model.noisy else [0.0]:
        confusion = torch.zeros(101, 101, dtype=torch.long)
        total, loss_sum = 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            t = torch.full((len(images),), time, device=device)
            if model.noisy:
                images = sde.marginal_sample(images, t)[0]
            logits = model(images, t * 1000).float()
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            predicted = logits.argmax(-1)
            confusion += torch.bincount(
                (labels * 101 + predicted).cpu(), minlength=101**2
            ).reshape(101, 101)
            total += len(images)
        per_class = confusion.diag() / confusion.sum(1).clamp_min(1)
        results[str(time)] = {
            "cross_entropy": loss_sum / total,
            "accuracy": confusion.diag().sum().item() / total,
            "macro_accuracy": per_class.mean().item(),
            "per_class_accuracy": per_class.tolist(),
            "confusion": confusion.tolist(),
            "examples": total,
        }
    model.train(was_training)
    return results


def main(noisy=False):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    name = "noisy_classifier" if noisy else "class_evaluator"
    parser.add_argument(
        "--checkpoint", type=Path, default=OUTPUT_ROOT / name / "latest.pth"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--examples", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    model, state = load_model(args.checkpoint, args.device)
    if state["kind"] != "classifier" or model.noisy != noisy:
        raise ValueError("Wrong classifier purpose.")
    args.image_size = state["data_config"]["image_size"]
    if data_config(args) != state["data_config"]:
        raise ValueError("Different data protocol.")
    output = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(123)
    result = classifier_metrics(
        model,
        make_loader(args, args.split, limit=args.examples),
        args.device,
        VPSDE(**state["algorithm"]["sde"]),
    )
    (output / "metrics.json").write_text(
        json.dumps({"split": args.split, "metrics": result}, indent=2) + "\n"
    )
    print(
        {
            k: {
                n: v
                for n, v in row.items()
                if n not in ("confusion", "per_class_accuracy")
            }
            for k, row in result.items()
        }
    )
