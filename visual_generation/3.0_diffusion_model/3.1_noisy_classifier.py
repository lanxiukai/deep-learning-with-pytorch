"""Train the time-conditioned VP guidance classifier on Food-101.

Select configurations using validation images; the official test split is held out.
The clean evaluator never shares weights with the guidance classifier.
"""

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.diffusion.classifier_evaluation import classifier_metrics
from dl_utils.diffusion.classifiers import ImageClassifier
from dl_utils.diffusion.data import data_config, make_loader
from dl_utils.diffusion.diffusion_score_sde import VPSDE
from dl_utils.diffusion.lesson_utils import (
    BinnedLoss,
    autocast,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("noisy_classifier")
    parser.add_argument("--width", type=int, default=64)
    return parser.parse_args()


def main(args):
    device = setup(args)
    model = ImageClassifier(width=args.width, noisy=True).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sde = VPSDE()
    metadata = {
        "kind": "classifier",
        "model_config": model.config(),
        "data_config": data_config(args),
        "algorithm": {
            "task": "noisy_classifier",
            "sde": {"beta_min": 0.1, "beta_max": 20.0},
            "time_epsilon": 1e-3,
            "precision": args.precision,
            "ema_decay": args.ema_decay,
        },
    }
    start, step, _ = resume_training(args, model, ema, [optimizer], metadata)
    loader = make_loader(args)
    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        for images, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            time = (
                1e-3 + (1 - 1e-3) * torch.rand(len(images), device=device)
                if model.noisy
                else torch.zeros(len(images), device=device)
            )
            noisy = sde.marginal_sample(images, time)[0] if model.noisy else images
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                logits = model(noisy, time * 1000)
                per_image = F.cross_entropy(logits.float(), labels, reduction="none")
                loss = per_image.mean()
            loss.backward()
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            meter.update(per_image, time)
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        metrics = meter.result()
        if args.eval_every and epoch % args.eval_every == 0:
            validation = make_loader(args, "validation", limit=args.eval_examples)
            metrics["validation"] = classifier_metrics(ema, validation, device, sde)
        record_epoch(args, epoch, step, metrics)
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer],
            epoch,
            step,
            metadata,
        )
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
