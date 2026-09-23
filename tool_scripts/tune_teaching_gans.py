"""Sequential, validation-guided tuning of the three 128x128 GAN lessons."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.runtime.gpu_targets import GPU_TARGETS, resolve_gpu_target

ROOT = Path(infer_project_root())
LESSONS = ROOT / "genai" / "1.0_generative_adversarial_network"
SCRIPTS = {
    "progan": "7.0_progan.py",
    "stylegan": "7.1_stylegan.py",
    "stylegan2": "7.2_stylegan2.py",
}
PROFILES = {
    "progan": [(5e-4, 10.0), (2.5e-4, 10.0), (2.5e-4, 5.0)],
    "stylegan": [(1e-3, 10.0), (1e-3, 2.0), (5e-4, 2.0)],
    "stylegan2": [(1e-3, 10.0), (1e-3, 2.0), (5e-4, 2.0)],
}
FRECHET = "projected_inception_frechet_256"
VARIANCE = "generated_feature_variance"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def assess_round(baseline, candidate, *, target_frechet, reference=None):
    """Screen candidates; these metric thresholds do not certify visual quality."""
    reference = baseline if reference is None else reference
    diverse = candidate[VARIANCE] >= reference[VARIANCE] * 0.8
    consistent = candidate[FRECHET] <= min(baseline[FRECHET], reference[FRECHET]) * 1.03
    gain = (baseline[FRECHET] - candidate[FRECHET]) / max(abs(baseline[FRECHET]), 1e-8)
    if not diverse or not consistent:
        return {
            "action": "change_profile",
            "relative_frechet_gain": gain,
            "reason": "diversity_or_frechet_regression",
        }
    if candidate[FRECHET] <= target_frechet:
        return {
            "action": "stop",
            "relative_frechet_gain": gain,
            "reason": "teaching_metric_target",
        }
    if gain >= 0.05:
        return {
            "action": "continue",
            "relative_frechet_gain": gain,
            "reason": "material_validation_gain",
        }
    return {
        "action": "change_profile",
        "relative_frechet_gain": gain,
        "reason": "validation_plateau",
    }


def run_attempt(args, model, source, directory, learning_rate, reg_weight):
    """Reuse finished attempts, resume checkpoints, and preserve interrupted setup."""
    summary = directory / "summary.json"
    if summary.is_file():
        return json.loads(summary.read_text())
    checkpoint = directory / "latest.pth"
    if directory.exists() and not checkpoint.exists():
        directory.rename(
            directory.with_name(f"{directory.name}-interrupted-{time.time_ns()}")
        )
    command = [
        sys.executable,
        str(LESSONS / SCRIPTS[model]),
        "--refine-resume" if checkpoint.exists() else "--refine-from",
        str(checkpoint if checkpoint.exists() else source),
        "--refine-output",
        str(directory),
        "--refine-kimg",
        str(args.round_kimg),
        "--refine-learning-rate",
        str(learning_rate),
        "--refine-reg-weight",
        str(reg_weight),
        "--refine-eval-samples",
        str(args.validation_samples),
        "--refine-review-samples",
        str(args.review_samples),
        "--refine-checkpoint-kimg",
        str(args.checkpoint_kimg),
        "--refine-skip-review",
    ]
    with directory.with_suffix(".log").open("a", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    return json.loads(summary.read_text())


def tune_model(args, model, output, suite):
    model_dir = output / model
    model_dir.mkdir(exist_ok=True)
    source = getattr(args, f"{model}_source").resolve()
    result = {"status": "training", "initial_checkpoint": str(source), "attempts": []}
    suite["models"][model] = result
    suite["active_model"] = model
    stop = False
    for profile, (learning_rate, reg_weight) in enumerate(PROFILES[model]):
        repetition = 0
        while True:
            directory = model_dir / f"profile-{profile + 1}-round-{repetition + 1}"
            result["active_attempt"] = str(directory)
            write_json(output / "progress.json", suite)
            print(
                json.dumps(
                    {
                        "event": "attempt_start",
                        "model": model,
                        "directory": str(directory),
                        "learning_rate": learning_rate,
                        "regularization_weight": reg_weight,
                    }
                ),
                flush=True,
            )
            try:
                summary = run_attempt(
                    args, model, source, directory, learning_rate, reg_weight
                )
            except subprocess.CalledProcessError:
                tail = directory.with_suffix(".log").read_text()[-5000:]
                if "FloatingPointError" not in tail:
                    raise
                result["attempts"].append(
                    {"directory": str(directory), "status": "numerical_failure"}
                )
                break
            candidate = summary["best"]["quality"]
            result.setdefault("baseline_validation", summary["baseline"])
            decision = assess_round(
                summary["baseline"],
                candidate,
                target_frechet=args.target_frechet,
                reference=result["baseline_validation"],
            )
            if decision["reason"] != "diversity_or_frechet_regression":
                source = directory / "best.pth"
                result["selected_checkpoint"] = str(source)
                result["selected_weights"] = str(directory / "best_generator.pth")
                result["selected_validation"] = candidate
            result["attempts"].append(
                {
                    "directory": str(directory),
                    "status": "completed",
                    "config": summary["refinement_config"],
                    "added_images": summary["added_images"],
                    "best_added_images": summary["best"]["added_images"],
                    "training_seconds": summary["training_seconds"],
                    "decision": decision,
                    "quality": candidate,
                }
            )
            write_json(output / "progress.json", suite)
            print(
                json.dumps(
                    {
                        "event": "attempt_decision",
                        "model": model,
                        **decision,
                        "quality": candidate,
                    }
                ),
                flush=True,
            )
            if decision["action"] == "stop":
                result["stop_reason"] = decision["reason"]
                stop = True
                break
            if decision["action"] == "change_profile":
                break
            repetition += 1
            # Try the next profile after a small number of productive rounds.
            # The last profile can continue while gains remain material: an
            # arbitrary round count must not label an improving model finished.
            if (
                profile < len(PROFILES[model]) - 1
                and repetition >= args.rounds_per_profile
            ):
                break
        if stop:
            break
    if "selected_checkpoint" not in result:
        raise RuntimeError(
            f"All {model} profiles failed numerically; no candidate was selected."
        )
    if not stop:
        last_decision = next(
            a["decision"] for a in reversed(result["attempts"]) if "decision" in a
        )
        result["stop_reason"] = (
            "tested_profiles_plateaued"
            if last_decision["action"] == "change_profile"
            else "numerical_failure_in_final_profile"
        )
    result["status"] = "awaiting_final_review"
    selected_dir = output / "selected" / model
    selected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result["selected_weights"], selected_dir / f"{model}_generator.pth")
    shutil.copy2(
        result["selected_checkpoint"], selected_dir / "selected_checkpoint.pth"
    )
    write_json(selected_dir / "selection.json", result)
    write_json(output / "progress.json", suite)


def final_review(args, output, suite):
    """Use the test partition once selection is over; retain visual review artifacts."""
    import torch
    from torchvision.utils import save_image

    from dl_utils.gan.celeba_evaluation import CelebAGeneratorEvaluator
    from dl_utils.gan.progan import ProGANGenerator
    from dl_utils.gan.stylegan import StyleGANGenerator
    from dl_utils.gan.stylegan2 import StyleGenerator
    from dl_utils.gan.stylegan_layers import denormalize
    from dl_utils.runtime.devices import configure_device
    from dl_utils.training.checkpoints import load_model_weights

    torch.set_num_threads(4)
    device = torch.device("cuda")
    configure_device(device)
    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    evaluator = CelebAGeneratorEvaluator(
        ROOT / "data" / "celeba",
        device=device,
        examples=args.review_samples,
        seed=20260907,
        split="test",
    )
    classes = {
        "progan": ProGANGenerator,
        "stylegan": StyleGANGenerator,
        "stylegan2": StyleGenerator,
    }
    for model in SCRIPTS:
        selected = output / "selected" / model
        evaluator.generator_kwargs = (
            {} if model == "progan" else {"noise_mode": "fixed", "truncation_psi": 1.0}
        )
        review = {}
        for label, weights in [
            ("baseline", ROOT / "output" / "gan" / model / f"{model}_generator.pth"),
            ("selected", selected / f"{model}_generator.pth"),
        ]:
            generator, _ = load_model_weights(weights, classes[model], device=device)
            metrics, samples = evaluator.evaluate(generator)
            review[label] = metrics
            save_image(denormalize(samples), selected / f"review_{label}.png", nrow=8)
            del generator
        suite["models"][model]["test_review"] = review
        # A metric target or a finished search never substitutes for inspecting faces.
        suite["models"][model]["status"] = "awaiting_visual_review"
        write_json(selected / "review.json", review)
        write_json(output / "progress.json", suite)
        print(
            json.dumps({"event": "test_review", "model": model, **review}), flush=True
        )
    del evaluator
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            str(LESSONS / "7.3_progan_stylegan_stylegan2_evaluation.py"),
            "--weights-root",
            str(output / "selected"),
        ],
        cwd=ROOT,
        check=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=tuple(GPU_TARGETS))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/gan/teaching-tuning")
    )
    for model in SCRIPTS:
        source = Path("output/gan") / model / "selected_checkpoint.pth"
        if not (ROOT / source).is_file():
            source = Path("output/gan") / model / "checkpoints/latest.pth"
        parser.add_argument(
            f"--{model}-source",
            type=Path,
            default=source,
        )
    parser.add_argument("--round-kimg", type=int, default=500)
    parser.add_argument("--checkpoint-kimg", type=int, default=50)
    parser.add_argument("--rounds-per-profile", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--review-samples", type=int, default=4096)
    parser.add_argument("--target-frechet", type=float, default=45.0)
    return parser.parse_args()


def main(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A supported CUDA GPU is required.")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_target = resolve_gpu_target(gpu_name, args.gpu)
    if min(args.round_kimg, args.checkpoint_kimg, args.rounds_per_profile) < 1:
        raise ValueError("Training budgets must be positive.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "gpu"
    }
    config["profiles"] = PROFILES
    # Normalize tuples to JSON lists before comparing a resumed search.
    config = json.loads(json.dumps(config))
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("A resumed search must keep its original configuration.")
    write_json(config_path, config)
    suite = {
        "status": "training",
        "gpu": gpu_name,
        "gpu_target": gpu_target,
        "models": {},
        "config": config,
        "acceptance": "Metric targets screen candidates; final image grids require visual review.",
        "scope": "128x128 teaching examples, not paper reproduction",
    }
    try:
        for model in SCRIPTS:
            tune_model(args, model, output, suite)
        suite["active_model"] = None
        suite["status"] = "final_evaluation"
        write_json(output / "progress.json", suite)
        final_review(args, output, suite)
        suite["status"] = "awaiting_visual_review"
        write_json(output / "summary.json", suite)
        write_json(output / "progress.json", suite)
        print(
            json.dumps(
                {
                    "event": "search_and_evaluation_finished",
                    "summary": str(output / "summary.json"),
                }
            ),
            flush=True,
        )
    except Exception as error:
        suite["status"] = "failed"
        suite["error"] = str(error)
        write_json(output / "progress.json", suite)
        raise


if __name__ == "__main__":
    main(parse_args())
