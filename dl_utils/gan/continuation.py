"""Bounded, resumable 128x128 continuation for the three compact GAN lessons."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torchvision.utils import save_image

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.celeba_evaluation import CelebAGeneratorEvaluator
from dl_utils.gan.celeba_runtime import GANRun, initialize_gan_models, prepare_gan_run
from dl_utils.gan.stylegan_layers import denormalize
from dl_utils.training.accelerator import make_fused_adam
from dl_utils.training.checkpoints import (
    TrainingCheckpoint,
    load_model_weights,
    load_training_checkpoint,
    save_model_weights,
)
from dl_utils.training.validation import validate_finite_training_state


@dataclass(frozen=True)
class ContinuationContext:
    """Models and runtime passed to one model-specific training callback."""

    generator: nn.Module
    discriminator: nn.Module
    ema: nn.Module
    run: GANRun
    optimizer_g: Optimizer
    optimizer_d: Optimizer
    batch_size: int


@dataclass(frozen=True)
class ContinuationPlan:
    """Explicit model construction, checkpoint contract, and chunk callback.

    Lesson scripts supply model configurations and their training functions;
    each model package resolves its regularization and sampling settings.
    """

    model_name: str
    generator_class: type[nn.Module]
    discriminator_class: type[nn.Module]
    model_config: Mapping[str, Any]
    discriminator_config: Mapping[str, Any]
    initial_unit: str
    initial_completed_units: int | None
    legacy_units: tuple[str, ...]
    generator_kwargs: Mapping[str, Any]
    d_reg_every: int
    reg_batch_shrink: int
    path_batch_shrink: int
    optimizer_ratios: Mapping[str, float]
    train_chunk: Callable[
        [ContinuationContext, int, dict[str, Any]], Mapping[str, float]
    ]


def add_refinement_arguments(
    parser, *, batch_size: int, learning_rate: float, regularization_help: str
):
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refine-from", type=Path, metavar="CHECKPOINT")
    modes.add_argument("--refine-resume", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--refine-output", type=Path)
    parser.add_argument("--refine-kimg", type=int, default=500)
    parser.add_argument("--refine-batch-size", type=int, default=batch_size)
    parser.add_argument(
        "--refine-learning-rate",
        type=float,
        default=learning_rate,
    )
    parser.add_argument(
        "--refine-reg-weight",
        type=float,
        default=10.0,
        help=regularization_help,
    )
    parser.add_argument("--refine-checkpoint-kimg", type=int, default=50)
    parser.add_argument("--refine-eval-samples", type=int, default=2048)
    parser.add_argument("--refine-review-samples", type=int, default=4096)
    parser.add_argument(
        "--refine-skip-review",
        action="store_true",
        help="Reserve test-set review for the end of an automatic search.",
    )


def continuation_batches(total_kimg, checkpoint_kimg, batch_size):
    """Partition the image budget into resumable whole-batch checkpoints."""
    if min(total_kimg, checkpoint_kimg, batch_size) < 1:
        raise ValueError("Refinement budgets and batch size must be positive.")
    remaining = math.ceil(total_kimg * 1000 / batch_size)
    interval = math.ceil(checkpoint_kimg * 1000 / batch_size)
    chunks = []
    while remaining:
        count = min(interval, remaining)
        chunks.append(count)
        remaining -= count
    return tuple(chunks)


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def continue_gan(args, *, plan: ContinuationPlan):
    """Keep baseline artifacts intact and retain the best validation candidate."""
    model_name = plan.model_name
    if args.refine_output is None:
        raise ValueError("Refinement requires a separate --refine-output directory.")
    if not math.isfinite(args.refine_learning_rate) or args.refine_learning_rate <= 0:
        raise ValueError("Refinement learning rate must be finite and positive.")
    if min(args.refine_eval_samples, args.refine_review_samples) < 2:
        raise ValueError("Quality evaluation needs at least two samples.")
    batch_size = args.refine_batch_size
    chunks = continuation_batches(
        args.refine_kimg,
        args.refine_checkpoint_kimg,
        batch_size,
    )
    if not math.isfinite(args.refine_reg_weight) or args.refine_reg_weight <= 0:
        raise ValueError("Regularization weight must be finite and positive.")
    d_reg_every = plan.d_reg_every
    reg_batch_shrink = plan.reg_batch_shrink
    path_batch_shrink = plan.path_batch_shrink
    if min(d_reg_every, reg_batch_shrink, path_batch_shrink) < 1:
        raise ValueError(
            "Regularization intervals and shrink factors must be positive."
        )
    source = args.refine_resume or args.refine_from
    output = args.refine_output.resolve()
    if args.refine_resume is None and source.resolve().is_relative_to(output):
        raise ValueError(
            "The refinement directory must not contain its source checkpoint."
        )
    torch.set_num_threads(4)
    root = Path(infer_project_root())
    torch.hub.set_dir(str(root / ".cache" / "torch" / "hub"))
    run = prepare_gan_run(
        model_name,
        seed=42,
        data_pipeline=args.data_pipeline,
        num_workers=4 if args.num_workers is None else args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    generator_class = plan.generator_class
    model_config = dict(plan.model_config)
    generator, discriminator, ema = initialize_gan_models(
        generator_class(**model_config),
        plan.discriminator_class(**plan.discriminator_config),
        run.device,
    )
    optimizer_g = make_fused_adam(
        generator.parameters(),
        device=run.device,
        lr=args.refine_learning_rate,
        betas=(0.0, 0.99),
    )
    optimizer_d = make_fused_adam(
        discriminator.parameters(),
        device=run.device,
        lr=args.refine_learning_rate,
        betas=(0.0, 0.99),
    )
    models = {
        "online_generator": generator,
        f"{model_name}_generator": ema,
        "discriminator": discriminator,
    }
    optimizers = {"generator": optimizer_g, "discriminator": optimizer_d}
    refinement_unit = f"{model_name}-128-refinement-v2"
    loaded = load_training_checkpoint(
        source,
        models=models,
        optimizers=optimizers,
        restore_random_state=args.refine_resume is not None,
    )
    source_unit = loaded["metadata"].get("unit")
    initial_unit = plan.initial_unit
    allowed_units = {initial_unit, refinement_unit, *plan.legacy_units}
    if source_unit not in allowed_units:
        raise ValueError("Source checkpoint is not a supported lesson or refinement.")
    if args.refine_resume and source_unit != refinement_unit:
        raise ValueError(
            "Use --refine-from to start a new round from older checkpoints."
        )
    config = {
        "resolution": 128,
        "model_name": model_name,
        "total_kimg": args.refine_kimg,
        "batch_size": batch_size,
        "learning_rate": args.refine_learning_rate,
        "checkpoint_kimg": args.refine_checkpoint_kimg,
        "eval_samples": args.refine_eval_samples,
        "review_samples": args.refine_review_samples,
        "d_reg_every": d_reg_every,
        "reg_batch_shrink": reg_batch_shrink,
        "regularization_weight": args.refine_reg_weight,
        "path_batch_shrink": path_batch_shrink,
        "skip_review": args.refine_skip_review,
        "ema_half_life_kimg": 10.0,
        "precision": run.precision.name,
        "data_pipeline": run.pipeline,
        "training_seed": 42,
    }
    weight_metadata = {"model_name": model_name, "model_config": model_config}
    if args.refine_resume:
        state = loaded["training_state"]
        if state["refinement_config"] != config:
            raise ValueError("Resume must use the saved refinement configuration.")
        completed = loaded["epoch"]
        if not 0 < completed <= len(chunks):
            raise ValueError("Refinement checkpoint exceeds its configured budget.")
        if state["added_images"] != sum(chunks[:completed]) * batch_size:
            raise ValueError(
                "Refinement checkpoint image count disagrees with its schedule."
            )
        if not (output / "baseline_generator.pth").is_file():
            raise ValueError(
                "Resume must use the original refinement output directory."
            )
    else:
        if (
            plan.initial_completed_units is not None
            and source_unit == initial_unit
            and loaded["epoch"] != plan.initial_completed_units
        ):
            raise ValueError(
                f"Refinement requires a completed {plan.initial_completed_units}-phase "
                "128x128 checkpoint."
            )
        output.mkdir(parents=True, exist_ok=False)
        completed = 0
        state = {
            "status": "training",
            "refinement_config": config,
            "source_checkpoint": str(source),
            "source_run_config": loaded["training_state"].get(
                "refinement_config", loaded["training_state"].get("run_config")
            ),
            "source_global_step": loaded["training_state"]["global_step"],
            "global_step": loaded["training_state"]["global_step"],
            "added_images": 0,
            "training_seconds": 0.0,
            "history": [],
            "path_mean": float(loaded["training_state"].get("path_mean", 0.0)),
        }
        save_model_weights(
            ema, output / "baseline_generator.pth", metadata=weight_metadata
        )
        save_model_weights(ema, output / "best_generator.pth", metadata=weight_metadata)
        shutil.copy2(source, output / "best.pth")
        _write_json(output / "config.json", config)
    if state["global_step"] != state["source_global_step"] + sum(chunks[:completed]):
        raise ValueError(
            "Refinement checkpoint step count disagrees with its schedule."
        )
    for name, optimizer in optimizers.items():
        ratio = plan.optimizer_ratios[name]
        for group in optimizer.param_groups:
            group["lr"] = args.refine_learning_rate * ratio
            group["betas"] = (0.0, 0.99**ratio)
    context = ContinuationContext(
        generator, discriminator, ema, run, optimizer_g, optimizer_d, batch_size
    )
    # Release CPU copies of the loaded parameters and optimizer state.
    del loaded
    output.joinpath("samples").mkdir(exist_ok=True)
    print(json.dumps({"event": "refinement_setup", **config}), flush=True)
    evaluator = CelebAGeneratorEvaluator(
        root / "data" / "celeba",
        device=run.device,
        examples=args.refine_eval_samples,
        generator_kwargs=plan.generator_kwargs,
    )
    if not args.refine_resume:
        baseline, samples = evaluator.evaluate(ema)
        state["baseline"] = baseline
        state["best"] = {"added_images": 0, "quality": baseline}
        save_image(denormalize(samples), output / "samples" / "baseline.png", nrow=8)
        _write_json(output / "baseline.json", baseline)
        _write_json(output / "progress.json", state)
        print(json.dumps({"event": "baseline", **baseline}), flush=True)
    checkpoint = TrainingCheckpoint(
        output / "latest.pth",
        unit=refinement_unit,
        models=models,
        optimizers=optimizers,
    )
    for index, count in enumerate(chunks[completed:], start=completed + 1):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        metrics = plan.train_chunk(context, count, state)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        state["training_seconds"] += seconds
        state["added_images"] += count * batch_size
        validate_finite_training_state(models, optimizers)
        quality, samples = evaluator.evaluate(ema)
        record = {
            "added_images": state["added_images"],
            "training_seconds": seconds,
            "sec_per_kimg": seconds * 1000 / (count * batch_size),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "training_metrics": metrics,
            "quality": quality,
        }
        state["history"].append(record)
        improved = (
            quality["projected_inception_frechet_256"]
            < state["best"]["quality"]["projected_inception_frechet_256"]
            and quality["projected_inception_frechet_256"]
            <= state["baseline"]["projected_inception_frechet_256"] * 1.03
            and quality["generated_feature_variance"]
            >= state["baseline"]["generated_feature_variance"] * 0.8
        )
        if improved:
            state["best"] = {"added_images": state["added_images"], "quality": quality}
            save_model_weights(
                ema, output / "best_generator.pth", metadata=weight_metadata
            )
        save_image(
            denormalize(samples),
            output / "samples" / f"added_{state['added_images']:07d}.png",
            nrow=8,
        )
        checkpoint.save(index, state)
        if improved:
            TrainingCheckpoint(
                output / "best.pth",
                unit=refinement_unit,
                models=models,
                optimizers=optimizers,
            ).save(index, state)
        _write_json(output / "progress.json", state)
        print(
            json.dumps(
                {
                    "event": "checkpoint",
                    **record,
                    "best_added_images": state["best"]["added_images"],
                }
            ),
            flush=True,
        )
    save_model_weights(ema, output / "final_generator.pth", metadata=weight_metadata)
    if args.refine_skip_review:
        summary = {
            **state,
            "status": "completed",
            "review": None,
            "original_artifacts_replaced": False,
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "progress.json", summary)
        print(json.dumps({"event": "completed", "best": state["best"]}), flush=True)
        return summary
    # A different split and latent seed are used only after candidate selection.
    review = CelebAGeneratorEvaluator(
        root / "data" / "celeba",
        device=run.device,
        examples=args.refine_review_samples,
        seed=20260907,
        split="test",
        feature_extractor=evaluator.features,
        generator_kwargs=evaluator.generator_kwargs,
    )
    review_model, _ = load_model_weights(
        output / "baseline_generator.pth", generator_class, device=run.device
    )
    baseline_review, baseline_samples = review.evaluate(review_model)
    del review_model
    review_model, _ = load_model_weights(
        output / "best_generator.pth", generator_class, device=run.device
    )
    best_review, best_samples = review.evaluate(review_model)
    save_image(
        denormalize(baseline_samples),
        output / "samples" / "review_baseline.png",
        nrow=8,
    )
    save_image(
        denormalize(best_samples), output / "samples" / "review_best.png", nrow=8
    )
    summary = {
        **state,
        "review": {"baseline": baseline_review, "selected": best_review},
        "status": "completed",
        "original_artifacts_replaced": False,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "progress.json", summary)
    print(
        json.dumps(
            {
                "event": "completed",
                "added_images": state["added_images"],
                "best_added_images": state["best"]["added_images"],
                "review": summary["review"],
            }
        ),
        flush=True,
    )
    return summary


__all__ = [
    "ContinuationContext",
    "ContinuationPlan",
    "add_refinement_arguments",
    "continuation_batches",
    "continue_gan",
]
