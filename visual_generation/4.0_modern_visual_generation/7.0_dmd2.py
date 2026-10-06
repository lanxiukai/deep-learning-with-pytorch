"""DMD2: online fake-score, shared GAN head and inference-distribution rollouts.

A one-step and a four-step student are separate training configurations. No
precomputed noise/teacher-image regression dataset is used. The DM scalar is
only a carrier of the stopped, normalized score-difference gradient.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dl_utils.diffusion.data import data_config, make_loader
from dl_utils.diffusion.lesson_utils import (
    autocast,
    record_epoch,
    resume_training,
    save_training,
    setup,
    training_parser,
)
from dl_utils.modern.checkpoints import load_model
from dl_utils.modern.dmd2 import (
    FakeScoreCritic,
    dm_surrogate,
    student_rollout,
)
from dl_utils.modern.edm import edm_noise_grid
from dl_utils.modern.monitoring import monitor_generation
from dl_utils.training.ema import update_ema


def parse_args():
    parser = training_parser("dmd2")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--student-steps", type=int, choices=(1, 4), default=1)
    parser.add_argument("--auxiliary-updates", type=int, default=5)
    parser.add_argument("--gan-weight", type=float, default=0.01)
    parser.add_argument(
        "--real-noised-inputs",
        action="store_true",
        help="Explicit input-mismatch training ablation.",
    )
    return parser.parse_args()


def main(args):
    device = setup(args)
    teacher, teacher_state = load_model(args.teacher_checkpoint, device)
    data = data_config(args)
    if (
        teacher_state["algorithm"]["task"] != "edm"
        or not teacher_state["algorithm"]["conditional"]
        or teacher_state["data_config"] != data
    ):
        raise ValueError(
            "DMD2 needs a class-conditional EDM teacher on the identical data protocol."
        )
    if args.auxiliary_updates < 1:
        raise ValueError("At least one auxiliary update is required.")
    model = copy.deepcopy(teacher).train().requires_grad_(True)
    critic = FakeScoreCritic(teacher).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    optimizer_aux = torch.optim.AdamW(
        critic.parameters(), lr=args.learning_rate, weight_decay=0
    )
    # Positive input noise levels, not a solver grid to be changed after training.
    sigmas = edm_noise_grid(args.student_steps + 1, 0.002, 80, 7, device)[:-2]
    algorithm = {
        "task": "dmd2",
        "conditional": True,
        "latent": False,
        "teacher_id": teacher_state["checkpoint_id"],
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "student_sigmas": sigmas.tolist(),
        "auxiliary_updates": args.auxiliary_updates,
        "gan_weight": args.gan_weight,
        "real_noised_inputs": args.real_noised_inputs,
        "precision": args.precision,
        "ema_decay": args.ema_decay,
    }
    metadata = {
        "kind": "edm",
        "model_config": model.config(),
        "data_config": data,
        "algorithm": algorithm,
    }
    start, step, saved = resume_training(
        args, model, ema, [optimizer, optimizer_aux], metadata
    )
    if saved:
        critic.load_state_dict(saved["critic"])
    for epoch in range(start, args.epochs + 1):
        model.train()
        sums, count = {}, 0
        for real, labels in tqdm(make_loader(args), desc=f"Epoch {epoch}"):
            real, labels = real.to(device), labels.to(device)
            batch = len(real)
            # Repeat online auxiliary updates. Both sides use the same label prior.
            for _ in range(args.auxiliary_updates):
                with torch.no_grad(), autocast(args):
                    stage = torch.randint(len(sigmas), ()).item()
                    fake = student_rollout(
                        model, torch.randn_like(real), labels, sigmas, stop_at=stage
                    )
                sigma = (
                    (-1.2 + 1.2 * torch.randn(batch, device=device))
                    .exp()
                    .clamp(0.002, 80)
                )
                s = sigma[:, None, None, None]
                noisy_fake = fake.detach() + s * torch.randn_like(real)
                noisy_real = real + s * torch.randn_like(real)
                critic.requires_grad_(True)
                optimizer_aux.zero_grad(set_to_none=True)
                with autocast(args):
                    fake_prediction, fake_logits = critic(noisy_fake, sigma, labels)
                    _, real_logits = critic(noisy_real, sigma, labels)
                    weight = (sigma.square() + teacher.sigma_data**2) / (
                        sigma * teacher.sigma_data
                    ).square()
                    score_loss = (
                        weight
                        * (fake_prediction.float() - fake.detach())
                        .square()
                        .flatten(1)
                        .mean(1)
                    ).mean()
                    discriminator_loss = (
                        F.softplus(fake_logits.float()).mean()
                        + F.softplus(-real_logits.float()).mean()
                    )
                    auxiliary_loss = score_loss + args.gan_weight * discriminator_loss
                auxiliary_loss.backward()
                optimizer_aux.step()
            # Student update: stop score derivatives, retain the GAN INPUT derivative.
            critic.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            stage = torch.randint(len(sigmas), ()).item()
            with autocast(args):
                if args.real_noised_inputs:
                    noisy_input = real + sigmas[stage] * torch.randn_like(real)
                    generated = model(noisy_input, sigmas[stage].expand(batch), labels)
                else:
                    generated = student_rollout(
                        model,
                        torch.randn_like(real),
                        labels,
                        sigmas,
                        stop_at=stage,
                        track_last=True,
                    )
                sigma = (
                    (-1.2 + 1.2 * torch.randn(batch, device=device))
                    .exp()
                    .clamp(0.002, 80)
                )
                noisy_generated = generated + sigma[
                    :, None, None, None
                ] * torch.randn_like(real)
                with torch.no_grad():
                    real_denoised = teacher(noisy_generated.detach(), sigma, labels)
                    fake_denoised = critic.denoiser(
                        noisy_generated.detach(), sigma, labels
                    )
                dm, direction = dm_surrogate(
                    generated.float(), real_denoised.float(), fake_denoised.float()
                )
                _, logits = critic(noisy_generated, sigma, labels)
                generator_adversarial = F.softplus(-logits.float()).mean()
                loss = dm.mean() + args.gan_weight * generator_adversarial
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DMD2 update.")
            loss.backward()
            optimizer.step()
            critic.requires_grad_(True)
            update_ema(ema, model, args.ema_decay)
            metrics = {
                "fake_score_mse": score_loss,
                "score_difference_rms": direction,
                "discriminator": discriminator_loss,
                "generator_adversarial": generator_adversarial,
            }
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0) + value.item() * batch
            count += batch
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        record_epoch(args, epoch, step, {k: v / count for k, v in sums.items()})
        save_training(
            args.output_dir / "latest.pth",
            model,
            ema,
            [optimizer, optimizer_aux],
            epoch,
            step,
            metadata,
            critic=critic.state_dict(),
        )
        monitor_generation(args, ema, metadata, epoch)
        if args.max_steps is not None and step >= args.max_steps:
            break


if __name__ == "__main__":
    main(parse_args())
