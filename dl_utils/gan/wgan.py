"""Shared 64x64 WGAN/WGAN-GP models and training helpers."""

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.gan.dcgan import (
    DCGANGenerator,
    initialize_dcgan_weights,
    save_dcgan_sample_grid,
)
from dl_utils.plot.curves import save_loss_panels
from dl_utils.runtime.devices import try_gpu
from dl_utils.training.metrics import MetricAccumulator


class WGANGenerator(DCGANGenerator):
    """Generate 64x64 RGB images with the DCGAN generator topology."""


class WGANCriticBlock(nn.Module):
    """Downsample without normalization for the Wasserstein critic."""

    def __init__(
        self,
        out_channels,
        in_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        negative_slope=0.2,
    ):
        super().__init__()
        # Keep the original attribute name for checkpoint compatibility.
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, inputs):
        hidden = self.conv2d(inputs)
        return self.activation(hidden)


class WGANCritic(nn.Sequential):
    """Score 64x64 RGB images without sigmoid or BatchNorm."""

    def __init__(self, image_channels=3, base_channels=64):
        super().__init__(
            WGANCriticBlock(
                in_channels=image_channels,
                out_channels=base_channels,
            ),
            WGANCriticBlock(
                in_channels=base_channels,
                out_channels=base_channels * 2,
            ),
            WGANCriticBlock(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
            ),
            WGANCriticBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 8,
            ),
            nn.Conv2d(
                in_channels=base_channels * 8,
                out_channels=1,
                kernel_size=4,
                bias=False,
            ),
        )


def gradient_penalty(
    critic,
    real_images,
    generated_images,
    lambda_gp=10,
):
    """Return the weighted WGAN-GP penalty on random interpolations."""
    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=real_images.device)
    interpolated_images = (
        epsilon * real_images + (1 - epsilon) * generated_images
    ).requires_grad_(True)
    interpolated_scores = critic(interpolated_images)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def update_wgan_critic(
    real_images,
    noise,
    critic,
    generator,
    optimizer,
    *,
    lipschitz="gp",
    lambda_gp=10,
    clip_value=0.01,
):
    """Run one Wasserstein critic update and return its diagnostics."""
    optimizer.zero_grad(set_to_none=True)
    real_score = critic(real_images).mean()
    with torch.no_grad():
        generated_images = generator(noise)
    fake_score = critic(generated_images).mean()
    wasserstein_gap = real_score - fake_score
    penalty = torch.zeros((), device=real_images.device)
    critic_loss = -wasserstein_gap

    if lipschitz == "gp":
        penalty = gradient_penalty(
            critic,
            real_images,
            generated_images,
            lambda_gp,
        )
        critic_loss = critic_loss + penalty

    critic_loss.backward()
    optimizer.step()
    if lipschitz == "clip":
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.clamp_(-clip_value, clip_value)
    return wasserstein_gap.detach(), penalty.detach()


def update_wgan_generator(noise, critic, generator, optimizer):
    """Run one generator update that maximizes the generated-image score."""
    optimizer.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    try:
        generator_loss = -critic(generator(noise)).mean()
        generator_loss.backward()
        optimizer.step()
    finally:
        critic.requires_grad_(True)
    return generator_loss.detach()


def _create_wgan_optimizers(critic, generator, lipschitz, learning_rate):
    if lipschitz == "clip":
        learning_rate = 5e-5 if learning_rate is None else learning_rate
        critic_optimizer = torch.optim.RMSprop(
            critic.parameters(),
            lr=learning_rate,
        )
        generator_optimizer = torch.optim.RMSprop(
            generator.parameters(),
            lr=learning_rate,
        )
    else:
        learning_rate = 1e-4 if learning_rate is None else learning_rate
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9),
        )
        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9),
        )
    return critic_optimizer, generator_optimizer


def train_wgan(
    generator,
    critic,
    data_loader,
    *,
    num_epochs,
    z_dim,
    output_dir,
    lipschitz="gp",
    learning_rate=None,
    n_critic=5,
    lambda_gp=10,
    clip_value=0.01,
    samples_to_display=18,
    sample_grid_columns=6,
    sample_every_epochs=6,
    device=None,
):
    """Train WGAN or WGAN-GP and save samples, losses, and generator weights."""
    if lipschitz not in {"gp", "clip"}:
        raise ValueError("lipschitz must be 'gp' or 'clip'.")
    if min(num_epochs, n_critic, sample_every_epochs) < 1:
        raise ValueError("training counts must be positive.")

    output_dir = Path(output_dir)
    training_dir = output_dir / "training"
    reset_dir(str(training_dir))
    device = try_gpu() if device is None else device
    initialize_dcgan_weights(generator)
    initialize_dcgan_weights(critic)
    generator = generator.to(device)
    critic = critic.to(device)
    critic_optimizer, generator_optimizer = _create_wgan_optimizers(
        critic,
        generator,
        lipschitz,
        learning_rate,
    )

    fixed_noise = torch.randn(samples_to_display, z_dim, 1, 1, device=device)
    epochs, wasserstein_gaps, penalties, generator_losses = [], [], [], []
    critic_steps = 0

    with tqdm(
        total=num_epochs * len(data_loader),
        desc=f"Epoch 1/{num_epochs}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(
                f"Epoch {epoch}/{num_epochs}",
                refresh=False,
            )
            critic_metrics = MetricAccumulator(
                ("wasserstein_gap", "gradient_penalty"),
                device=device,
            )
            generator_metrics = MetricAccumulator(
                ("generator_loss",),
                device=device,
            )
            generator_updated = False

            for real_images, _ in data_loader:
                real_images = real_images.to(device, non_blocking=True)
                batch_size = real_images.shape[0]
                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                wasserstein_gap, penalty = update_wgan_critic(
                    real_images,
                    noise,
                    critic,
                    generator,
                    critic_optimizer,
                    lipschitz=lipschitz,
                    lambda_gp=lambda_gp,
                    clip_value=clip_value,
                )
                critic_metrics.add_batch_means(
                    (wasserstein_gap, penalty),
                    num_examples=batch_size,
                )
                critic_steps += 1

                if critic_steps % n_critic == 0:
                    noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                    generator_loss = update_wgan_generator(
                        noise,
                        critic,
                        generator,
                        generator_optimizer,
                    )
                    generator_metrics.add_batch_means(
                        (generator_loss,),
                        num_examples=batch_size,
                    )
                    generator_updated = True
                progress_bar.update(1)

            critic_epoch_metrics = critic_metrics.compute_weighted_means()
            wasserstein_gap = critic_epoch_metrics["wasserstein_gap"]
            penalty = critic_epoch_metrics["gradient_penalty"]
            generator_loss = (
                generator_metrics.compute_weighted_means()["generator_loss"]
                if generator_updated
                else 0.0
            )
            epochs.append(epoch)
            wasserstein_gaps.append(wasserstein_gap)
            penalties.append(penalty)
            generator_losses.append(generator_loss)
            progress_bar.set_postfix(
                gap=f"{wasserstein_gap:.3f}",
                penalty=f"{penalty:.3f}",
                loss_G=f"{generator_loss:.3f}",
            )

            if epoch == 1 or epoch % sample_every_epochs == 0:
                save_dcgan_sample_grid(
                    generator,
                    fixed_noise,
                    training_dir / f"epoch_{epoch:03d}.png",
                    columns=sample_grid_columns,
                )

    print(
        f"lipschitz={lipschitz}: wasserstein_gap {wasserstein_gap:.3f}, "
        f"gradient_penalty {penalty:.3f}, loss_gen {generator_loss:.3f}, "
        f"on {device}"
    )
    checkpoint_name = "wgan_gp.pth" if lipschitz == "gp" else "wgan.pth"
    torch.save(generator.state_dict(), output_dir / checkpoint_name)
    save_loss_panels(
        epochs,
        {
            "Wasserstein gap": {"Wasserstein gap": wasserstein_gaps},
            "Gradient penalty": {"Gradient penalty": penalties},
            "Generator loss": {"G loss": generator_losses},
        },
        output_dir / "loss_curves.png",
    )


__all__ = [
    "WGANCritic",
    "WGANCriticBlock",
    "WGANGenerator",
    "gradient_penalty",
    "train_wgan",
    "update_wgan_critic",
    "update_wgan_generator",
]
