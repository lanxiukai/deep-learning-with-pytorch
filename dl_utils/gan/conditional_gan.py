"""Conditional WGAN-GP models and training for the glasses lesson."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from tqdm import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.gan.wgan import gradient_penalty
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import save_loss_panels
from dl_utils.runtime.devices import try_gpu
from dl_utils.training.metrics import MetricAccumulator


class ConditionalGenerator(nn.Module):
    """Generate 256x256 RGB images from noise and condition channels."""

    def __init__(self, noise_channels, image_channels, base_channels):
        super().__init__()
        self.net = nn.Sequential(
            self._block(noise_channels, base_channels * 64, 4, 1, 0),
            self._block(base_channels * 64, base_channels * 32, 4, 2, 1),
            self._block(base_channels * 32, base_channels * 16, 4, 2, 1),
            self._block(base_channels * 16, base_channels * 8, 4, 2, 1),
            self._block(base_channels * 8, base_channels * 4, 4, 2, 1),
            self._block(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                base_channels * 2,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class ConditionalCritic(nn.Module):
    """Score images concatenated with spatial condition channels."""

    def __init__(self, image_channels, base_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                image_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            self._block(base_channels, base_channels * 2, 4, 2, 1),
            self._block(base_channels * 2, base_channels * 4, 4, 2, 1),
            self._block(base_channels * 4, base_channels * 8, 4, 2, 1),
            self._block(base_channels * 8, base_channels * 16, 4, 2, 1),
            self._block(base_channels * 16, base_channels * 32, 4, 2, 1),
            nn.Conv2d(
                base_channels * 32,
                1,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
        )

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, inputs):
        return self.net(inputs)


def initialize_conditional_gan_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


@torch.inference_mode()
def save_conditional_sample_grids(
    generator,
    noise,
    epoch,
    training_dir,
    *,
    num_classes,
    rows,
    columns,
):
    """Save one fixed-noise generated grid for every condition class."""
    if len(noise) != rows * columns:
        raise ValueError("noise must contain rows * columns samples.")

    was_training = generator.training
    generator.eval()
    try:
        for class_index in range(num_classes):
            labels = torch.zeros(
                len(noise),
                num_classes,
                1,
                1,
                device=noise.device,
            )
            labels[:, class_index] = 1
            conditioned_noise = torch.cat([noise, labels], dim=1)
            generated_images = (
                generator(conditioned_noise).permute(0, 2, 3, 1).cpu().mul(0.5).add(0.5)
            )
            image_grid = torch.cat(
                [
                    torch.cat(
                        generated_images[
                            row_index * columns : (row_index + 1) * columns
                        ].unbind(),
                        dim=1,
                    )
                    for row_index in range(rows)
                ],
                dim=0,
            )
            prefix = (
                ("G", "NoG")[class_index]
                if num_classes == 2
                else f"class_{class_index}"
            )
            plt.imsave(
                training_dir / f"{prefix}{epoch}.png",
                image_grid.clamp(0, 1).numpy(),
            )
    finally:
        generator.train(was_training)


def update_conditional_critic(
    real_conditioned_images,
    noise,
    one_hot_labels,
    critic,
    generator,
    optimizer,
    *,
    lambda_gp=10,
):
    """Run one conditional critic update and return its diagnostics."""
    batch_size, num_classes = one_hot_labels.shape
    optimizer.zero_grad(set_to_none=True)
    real_score = critic(real_conditioned_images).mean()
    conditioned_noise = torch.cat(
        [noise, one_hot_labels.reshape(batch_size, num_classes, 1, 1)],
        dim=1,
    )
    with torch.no_grad():
        generated_images = generator(conditioned_noise)
    condition_channels = real_conditioned_images[:, -num_classes:]
    generated_conditioned_images = torch.cat(
        [generated_images, condition_channels],
        dim=1,
    )
    generated_score = critic(generated_conditioned_images).mean()
    wasserstein_gap = real_score - generated_score
    penalty = gradient_penalty(
        critic,
        real_conditioned_images,
        generated_conditioned_images,
        lambda_gp,
    )
    critic_loss = -wasserstein_gap + penalty
    critic_loss.backward()
    optimizer.step()
    return wasserstein_gap.detach(), penalty.detach()


def update_conditional_generator(
    noise,
    one_hot_labels,
    real_conditioned_images,
    critic,
    generator,
    optimizer,
):
    """Run one conditional generator update."""
    batch_size, num_classes = one_hot_labels.shape
    optimizer.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    try:
        conditioned_noise = torch.cat(
            [noise, one_hot_labels.reshape(batch_size, num_classes, 1, 1)],
            dim=1,
        )
        generated_images = generator(conditioned_noise)
        condition_channels = real_conditioned_images[:, -num_classes:]
        generated_conditioned_images = torch.cat(
            [generated_images, condition_channels],
            dim=1,
        )
        generator_loss = -critic(generated_conditioned_images).mean()
        generator_loss.backward()
        optimizer.step()
    finally:
        critic.requires_grad_(True)
    return generator_loss.detach()


def train_conditional_gan(
    generator,
    critic,
    data_loader,
    *,
    num_epochs,
    z_dim,
    num_classes,
    output_dir,
    n_critic=5,
    lambda_gp=10,
    generator_lr=1e-4,
    critic_lr=1e-4,
    sample_grid_rows=4,
    sample_grid_columns=8,
    sample_every_epochs=6,
    device=None,
):
    """Train the conditional WGAN-GP and save samples, losses, and weights."""
    if min(num_epochs, n_critic, num_classes, sample_every_epochs) < 1:
        raise ValueError("training counts must be positive.")

    output_dir = Path(output_dir)
    training_dir = output_dir / "training"
    reset_dir(str(training_dir))
    device = try_gpu() if device is None else device
    generator.apply(initialize_conditional_gan_weights)
    critic.apply(initialize_conditional_gan_weights)
    generator = generator.to(device)
    critic = critic.to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_lr,
        betas=(0.0, 0.9),
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=critic_lr,
        betas=(0.0, 0.9),
    )

    samples_to_display = sample_grid_rows * sample_grid_columns
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

            for images, labels in data_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                batch_size, _, height, width = images.shape
                one_hot_labels = functional.one_hot(
                    labels,
                    num_classes=num_classes,
                ).to(images.dtype)
                condition_channels = one_hot_labels[:, :, None, None].expand(
                    -1,
                    -1,
                    height,
                    width,
                )
                real_conditioned_images = torch.cat(
                    [images, condition_channels],
                    dim=1,
                )
                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                wasserstein_gap, penalty = update_conditional_critic(
                    real_conditioned_images,
                    noise,
                    one_hot_labels,
                    critic,
                    generator,
                    critic_optimizer,
                    lambda_gp=lambda_gp,
                )
                critic_metrics.add_batch_means(
                    (wasserstein_gap, penalty),
                    num_examples=batch_size,
                )
                critic_steps += 1

                if critic_steps % n_critic == 0:
                    noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
                    generator_loss = update_conditional_generator(
                        noise,
                        one_hot_labels,
                        real_conditioned_images,
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
                save_conditional_sample_grids(
                    generator,
                    fixed_noise,
                    epoch,
                    training_dir,
                    num_classes=num_classes,
                    rows=sample_grid_rows,
                    columns=sample_grid_columns,
                )

    print(
        f"wasserstein_gap {wasserstein_gap:.3f}, "
        f"gradient_penalty {penalty:.3f}, loss_gen {generator_loss:.3f}, "
        f"on {device}"
    )
    torch.save(generator.state_dict(), output_dir / "cgan.pth")
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
    "ConditionalCritic",
    "ConditionalGenerator",
    "initialize_conditional_gan_weights",
    "save_conditional_sample_grids",
    "train_conditional_gan",
    "update_conditional_critic",
    "update_conditional_generator",
]
