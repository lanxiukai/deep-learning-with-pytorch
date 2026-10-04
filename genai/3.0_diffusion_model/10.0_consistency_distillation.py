"""Distill a trained class-conditional EDM into a consistency endpoint map.

Teacher integration and the EMA target are stopped; only the student receives
consistency gradients. Compare against the same teacher at matched NFE.
"""

import copy
from pathlib import Path

import torch
from tqdm import tqdm

from dl_utils.diffusion.checkpoints import load_model
from dl_utils.diffusion.consistency import ConsistencyModel, consistency_loss
from dl_utils.diffusion.data import data_config, make_loader
from dl_utils.diffusion.lesson_utils import (
    BinnedLoss,
    autocast,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.diffusion.monitoring import monitor_generation
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("consistency_distillation")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--intervals", type=int, default=80)
    parser.add_argument("--pair-solver", choices=("euler", "heun"), default="heun")
    return parser.parse_args()


def main(args):
    device = setup(args)
    teacher, teacher_state = load_model(args.teacher_checkpoint, device)
    if (
        teacher_state["algorithm"]["task"] != "edm"
        or not teacher_state["algorithm"]["conditional"]
    ):
        raise ValueError("CD requires the trained class-conditional EDM teacher.")
    data = data_config(args)
    if teacher_state["data_config"] != data:
        raise ValueError("Teacher and student must use the same data and resolution.")
    model = ConsistencyModel(
        teacher.network.config(),
        sigma_data=teacher.sigma_data,
        noise_embedding_scale=teacher.noise_embedding_scale,
    ).to(device)
    model.network.load_state_dict(teacher.network.state_dict())
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    algorithm = {
        "task": "consistency",
        "conditional": True,
        "latent": False,
        "teacher_id": teacher_state["checkpoint_id"],
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "sigma_max": teacher_state["algorithm"]["sigma_max"],
        "intervals": args.intervals,
        "pair_solver": args.pair_solver,
        "distance": "coordinate_mean_mse",
        "ema_decay": args.ema_decay,
        "precision": args.precision,
    }
    metadata = {
        "kind": "consistency",
        "model_config": model.config(),
        "data_config": data,
        "algorithm": algorithm,
    }
    start, step, _ = resume_training(args, model, ema, [optimizer], metadata)
    for epoch in range(start, args.epochs + 1):
        model.train()
        meter = BinnedLoss()
        for images, labels in tqdm(make_loader(args), desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(args):
                per_image, coordinate = consistency_loss(
                    model,
                    ema,
                    teacher,
                    images,
                    labels,
                    sigma_max=algorithm["sigma_max"],
                    intervals=args.intervals,
                    solver=args.pair_solver,
                )
                loss = per_image.float().mean()
            loss.backward()
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            meter.update(per_image, coordinate)
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        record_epoch(args, epoch, step, meter.result())
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer],
            epoch,
            step,
            metadata,
        )
        monitor_generation(args, ema, metadata, epoch)
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
