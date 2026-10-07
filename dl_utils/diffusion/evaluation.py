"""Independent post-training evaluation, sampler comparisons and class/SR evidence."""

import argparse
import copy
import json
from pathlib import Path

import torch

from dl_utils.diffusion.checkpoints import load_model
from dl_utils.diffusion.data import data_config, low_resolution, make_loader
from dl_utils.diffusion.lesson_utils import DATA_DIR, OUTPUT_ROOT, preview
from dl_utils.diffusion.quality import CleanFeatures, evaluate_generation
from dl_utils.diffusion.sampling import make_sampler


def parser_for(default_name, *, modern=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=OUTPUT_ROOT / default_name / "latest.pth"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    if modern:
        parser.add_argument("--autoencoder-checkpoint", type=Path)
    else:
        parser.set_defaults(autoencoder_checkpoint=None)
    parser.add_argument(
        "--class-evaluator",
        type=Path,
        help="Independent clean-image classifier for category compliance.",
    )
    parser.add_argument(
        "--noisy-classifier",
        type=Path,
        help="VP guidance classifier, separate from the evaluator.",
    )
    if modern:
        parser.add_argument(
            "--low-checkpoint",
            type=Path,
            help="128px conditional DDPM for a generated-condition SR cascade.",
        )
    else:
        parser.set_defaults(low_checkpoint=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--examples", dest="eval_examples", type=int, default=25250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--solvers", nargs="+")
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0])
    if modern:
        parser.add_argument("--augmentation-level", type=float, default=0.0)
    else:
        parser.set_defaults(augmentation_level=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--nearest-examples",
        type=int,
        default=2020,
        help="Training reference subset; 0 disables nearest-neighbor panels.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Images only, without pretrained metrics or a quality claim.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def evaluate(
    args,
    expected_tasks=None,
    *,
    load_checkpoint=load_model,
    codec_loader=None,
    sampler_factory=make_sampler,
    solver_choices=None,
    quality_evaluator=evaluate_generation,
):
    """Common evaluation loop with constructors selected by the lesson family."""
    device = torch.device(args.device)
    model, state = load_checkpoint(args.checkpoint, device)
    task = state["algorithm"]["task"]
    if expected_tasks is not None and task not in expected_tasks:
        raise ValueError(f"Expected {expected_tasks}, received {task}.")
    args.image_size = state["data_config"]["image_size"]
    if data_config(args) != state["data_config"]:
        raise ValueError("Evaluation dataset differs from the training contract.")
    root = args.output_dir or args.checkpoint.parent / f"evaluation_{args.split}"
    root.mkdir(parents=True, exist_ok=True)
    codec, scale = (
        codec_loader(state, device, args.autoencoder_checkpoint)
        if codec_loader
        else (None, 1.0)
    )
    classifier = evaluator = None
    if args.noisy_classifier:
        classifier, classifier_state = load_checkpoint(args.noisy_classifier, device)
        if (
            classifier_state["algorithm"]["task"] != "noisy_classifier"
            or classifier_state["data_config"] != state["data_config"]
        ):
            raise ValueError(
                "The VP guidance classifier has the wrong purpose or data protocol."
            )
        if (
            task != "vp"
            or classifier_state["algorithm"]["sde"] != state["algorithm"]["sde"]
        ):
            raise ValueError(
                "Noisy classifier and score must use the same VP schedule."
            )
    if args.class_evaluator:
        evaluator, evaluator_state = load_checkpoint(args.class_evaluator, device)
        if (
            evaluator_state["algorithm"]["task"] != "class_evaluator"
            or evaluator_state["data_config"] != state["data_config"]
        ):
            raise ValueError(
                "Use an independent clean-image evaluator on the same Food-101 protocol."
            )
    needs_labels = state["algorithm"].get("conditional", False) or (
        task == "vp" and classifier is not None
    )
    if needs_labels and not args.preview_only and evaluator is None:
        raise ValueError(
            "Supply --class-evaluator from 0.2; first check its held-out accuracy with 0.3."
        )
    if task == "vp" and classifier is None:
        args.guidance = [0.0]
    sampler = sampler_factory(
        model, state, codec=codec, latent_scale=scale, classifier=classifier
    )
    low_sampler, low_model = None, None
    if args.low_checkpoint:
        if task not in ("sr3", "sr_regression"):
            raise ValueError("A low-resolution model is only used by SR lessons.")
        low_model, low_state = load_checkpoint(args.low_checkpoint, device)
        expected = copy.deepcopy(state["data_config"])
        expected["image_size"] //= 2
        if (
            low_state["data_config"] != expected
            or low_state["algorithm"]["task"] != "ddpm"
            or not low_state["algorithm"]["conditional"]
        ):
            raise ValueError(
                "Train the conditional DDPM on the same sample IDs at half resolution."
            )
        if evaluator is None and not args.preview_only:
            raise ValueError(
                "Category cascades require --class-evaluator to measure class preservation."
            )
        base_sampler = sampler_factory(low_model, low_state)
        low_sampler = lambda count, labels: base_sampler(
            count, labels=labels, steps=50, solver="ddim"
        )
    allowed = solver_choices or {
        "ddpm": ["ddpm", "ddim"],
        "vp": ["reverse_sde", "euler", "heun"],
        "cfm": ["euler", "heun"],
    }
    solvers = args.solvers or allowed[task]
    if any(s not in allowed[task] for s in solvers):
        raise ValueError(f"Valid solvers for {task}: {allowed[task]}")
    if task == "dmd2":
        default_steps = [len(state["algorithm"]["student_sigmas"])]
    elif task in ("imf", "consistency"):
        default_steps = [1, 2, 4]
    elif task == "sr_regression":
        default_steps = [1]
    else:
        default_steps = [25, 50, 100]
    counts = args.steps or default_steps
    features = None if args.preview_only else CleanFeatures(device)
    records = []
    for solver in solvers:
        steps_for_solver = (
            [state["algorithm"]["diffusion"]["num_steps"]]
            if solver == "ddpm"
            else counts
        )
        for steps in steps_for_solver:
            for guidance in args.guidance:
                torch.manual_seed(args.seed)
                current = copy.copy(args)
                current.output_dir = root / f"{solver}_{steps}_guidance_{guidance:g}"
                current.output_dir.mkdir(parents=True, exist_ok=True)
                if args.preview_only:
                    source, labels = next(
                        iter(
                            make_loader(
                                args, args.split, limit=min(args.batch_size, 44)
                            )
                        )
                    )
                    source, labels = source.to(device), labels.to(device)
                    low = (
                        low_resolution(source)
                        if task in ("sr3", "sr_regression")
                        else None
                    )
                    if low_sampler:
                        low = low_sampler(len(source), labels)
                    images = sampler(
                        len(source),
                        labels=labels if needs_labels or low is not None else None,
                        steps=steps,
                        solver=solver,
                        guidance=guidance,
                        low=low,
                        augmentation_level=args.augmentation_level,
                    )
                    preview(current.output_dir / "samples.png", images)
                    record = {
                        "mode": "preview_only",
                        "solver": solver,
                        "steps": steps,
                        "guidance": guidance,
                    }
                else:
                    counted = {"generator": model}
                    if codec:
                        counted["decoder"] = codec.decoder
                    if classifier:
                        counted["classifier_forward_plus_input_gradient"] = classifier
                    if low_model:
                        counted["low_generator"] = low_model
                    record = quality_evaluator(
                        current,
                        model,
                        state,
                        sampler,
                        split=args.split,
                        steps=steps,
                        solver=solver,
                        guidance=guidance,
                        class_evaluator=evaluator,
                        features=features,
                        low_sampler=low_sampler,
                        augmentation_level=args.augmentation_level,
                        count_modules=counted,
                    )
                record.update(
                    checkpoint_id=state["checkpoint_id"],
                    task=task,
                    seed=args.seed,
                    class_evaluator_id=evaluator_state["checkpoint_id"]
                    if evaluator
                    else None,
                )
                (current.output_dir / "metrics.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                records.append(record)
                print(
                    {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
                )
    (root / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    if not args.preview_only:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        for solver in solvers:
            rows = [r for r in records if r["sampler"] == solver]
            axis.plot(
                [r["seconds_per_image"] for r in rows],
                [r["fid"] for r in rows],
                "o-",
                label=solver,
            )
        axis.set(xlabel="Seconds per image (batch throughput)", ylabel="Clean FID")
        axis.legend()
        figure.savefig(root / "quality_cost.png", bbox_inches="tight")
        plt.close(figure)
    return records


def main(default_name, expected_tasks):
    evaluate(parser_for(default_name).parse_args(), expected_tasks)
