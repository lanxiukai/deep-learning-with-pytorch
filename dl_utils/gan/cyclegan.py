"""CycleGAN models and training helpers for unpaired image translation."""

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from dl_utils.data.images import load_rgb_image
from dl_utils.plot._backend import pyplot as plt

LAMBDA_CYCLE = 10


def save_cyclegan_translation_snapshot(
    epoch,
    batch_index,
    domain_a_images,
    domain_b_images,
    generated_a_images,
    generated_b_images,
    output_dir,
    domain_a_label,
    domain_b_label,
):
    """Save a titled 2x2 grid for both CycleGAN translation directions."""
    comparison_images = [
        domain_a_images[0],
        generated_b_images[0],
        domain_b_images[0],
        generated_a_images[0],
    ]
    titles = [
        f"Real: {domain_a_label}",
        f"Generated: {domain_b_label}",
        f"Real: {domain_b_label}",
        f"Generated: {domain_a_label}",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(7, 7), dpi=150)
    for axis, image, title in zip(
        axes.flat,
        comparison_images,
        titles,
        strict=True,
    ):
        display_image = image.detach().float().mul(0.5).add(0.5).clamp(0, 1)
        axis.imshow(display_image.permute(1, 2, 0).cpu().numpy())
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"CycleGAN translations — epoch {epoch}, step {batch_index}")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(
        output_dir / f"comparison_epoch_{epoch:02d}_step_{batch_index:05d}.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def train_cyclegan_epoch(
    discriminator_a,
    discriminator_b,
    generator_a,
    generator_b,
    data_loader,
    discriminator_optimizer,
    generator_optimizer,
    reconstruction_loss,
    adversarial_loss,
    discriminator_scaler,
    generator_scaler,
    device,
    output_dir,
    *,
    epoch=1,
    loss_callback=None,
    loss_updates_per_epoch=20,
    snapshot_updates_per_epoch=10,
    domain_a_label="Domain A",
    domain_b_label="Domain B",
    progress_bar=None,
):
    """Train one epoch and report weighted generator-loss components."""
    if loss_updates_per_epoch <= 0:
        raise ValueError("loss_updates_per_epoch must be positive.")
    if snapshot_updates_per_epoch <= 0:
        raise ValueError("snapshot_updates_per_epoch must be positive.")

    batches = (
        data_loader
        if progress_bar is not None
        else tqdm(data_loader, leave=True, mininterval=1.0)
    )
    accumulated_losses = torch.zeros(2, device=device)
    window_loss_sums = [0.0] * 5
    num_examples = 0
    window_examples = 0
    num_batches = len(data_loader)
    loss_update_interval = max(
        1,
        (num_batches + loss_updates_per_epoch - 1) // loss_updates_per_epoch,
    )
    snapshot_interval = max(
        1,
        (num_batches + snapshot_updates_per_epoch - 1) // snapshot_updates_per_epoch,
    )

    for batch_index, (domain_a_images, domain_b_images) in enumerate(
        batches,
        start=1,
    ):
        domain_a_images = domain_a_images.to(device)
        domain_b_images = domain_b_images.to(device)
        batch_size = domain_a_images.shape[0]

        with torch.amp.autocast(device.type):
            generated_a_images = generator_a(domain_b_images)
            real_a_outputs = discriminator_a(domain_a_images)
            generated_a_outputs = discriminator_a(generated_a_images.detach())
            discriminator_a_loss = adversarial_loss(
                real_a_outputs,
                torch.ones_like(real_a_outputs),
            ) + adversarial_loss(
                generated_a_outputs,
                torch.zeros_like(generated_a_outputs),
            )

            generated_b_images = generator_b(domain_a_images)
            real_b_outputs = discriminator_b(domain_b_images)
            generated_b_outputs = discriminator_b(generated_b_images.detach())
            discriminator_b_loss = adversarial_loss(
                real_b_outputs,
                torch.ones_like(real_b_outputs),
            ) + adversarial_loss(
                generated_b_outputs,
                torch.zeros_like(generated_b_outputs),
            )
            discriminator_loss = (discriminator_a_loss + discriminator_b_loss) / 2

        discriminator_optimizer.zero_grad()
        discriminator_scaler.scale(discriminator_loss).backward()
        discriminator_scaler.step(discriminator_optimizer)
        discriminator_scaler.update()

        with torch.amp.autocast(device.type):
            generated_a_outputs = discriminator_a(generated_a_images)
            generated_b_outputs = discriminator_b(generated_b_images)
            generator_a_adversarial_loss = adversarial_loss(
                generated_a_outputs,
                torch.ones_like(generated_a_outputs),
            )
            generator_b_adversarial_loss = adversarial_loss(
                generated_b_outputs,
                torch.ones_like(generated_b_outputs),
            )

            reconstructed_b_images = generator_b(generated_a_images)
            reconstructed_a_images = generator_a(generated_b_images)
            weighted_cycle_a_loss = LAMBDA_CYCLE * reconstruction_loss(
                domain_a_images,
                reconstructed_a_images,
            )
            weighted_cycle_b_loss = LAMBDA_CYCLE * reconstruction_loss(
                domain_b_images,
                reconstructed_b_images,
            )
            generator_loss = (
                generator_a_adversarial_loss
                + generator_b_adversarial_loss
                + weighted_cycle_a_loss
                + weighted_cycle_b_loss
            )

        generator_optimizer.zero_grad()
        generator_scaler.scale(generator_loss).backward()
        generator_scaler.step(generator_optimizer)
        generator_scaler.update()

        if batch_index % snapshot_interval == 0 or batch_index == num_batches:
            save_cyclegan_translation_snapshot(
                epoch,
                batch_index,
                domain_a_images,
                domain_b_images,
                generated_a_images,
                generated_b_images,
                output_dir,
                domain_a_label,
                domain_b_label,
            )

        batch_losses = torch.stack(
            [
                discriminator_loss,
                generator_loss,
                generator_a_adversarial_loss,
                generator_b_adversarial_loss,
                weighted_cycle_a_loss,
                weighted_cycle_b_loss,
            ]
        ).detach()
        accumulated_losses += batch_losses[:2] * batch_size
        num_examples += batch_size
        (
            discriminator_loss_value,
            generator_loss_value,
            generator_a_loss_value,
            generator_b_loss_value,
            weighted_cycle_a_loss_value,
            weighted_cycle_b_loss_value,
        ) = batch_losses.tolist()

        if loss_callback is not None:
            loss_values = (
                discriminator_loss_value,
                generator_a_loss_value,
                generator_b_loss_value,
                weighted_cycle_a_loss_value,
                weighted_cycle_b_loss_value,
            )
            for loss_index, loss_value in enumerate(loss_values):
                window_loss_sums[loss_index] += loss_value * batch_size
            window_examples += batch_size
            if batch_index % loss_update_interval == 0 or batch_index == num_batches:
                window_losses = [
                    loss_sum / window_examples for loss_sum in window_loss_sums
                ]
                window_discriminator_loss, *window_generator_components = window_losses
                loss_callback(
                    batch_index / num_batches,
                    window_discriminator_loss,
                    sum(window_generator_components),
                    *window_generator_components,
                )
                window_loss_sums = [0.0] * 5
                window_examples = 0

        loss_postfix = {
            "loss_D": discriminator_loss_value,
            "loss_G": generator_loss_value,
        }
        if progress_bar is None:
            batches.set_postfix(**loss_postfix, refresh=False)
        else:
            progress_bar.set_postfix(**loss_postfix, refresh=False)
            progress_bar.update(1)

    if num_examples == 0:
        raise ValueError("Cannot train CycleGAN with an empty data loader.")
    discriminator_loss_sum, generator_loss_sum = accumulated_losses.tolist()
    return (
        discriminator_loss_sum / num_examples,
        generator_loss_sum / num_examples,
    )


class CycleGANConvBlock(nn.Module):
    """Convolution, instance normalization, and optional activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        downsample=True,
        use_activation=True,
        **convolution_kwargs,
    ):
        super().__init__()
        convolution = (
            nn.Conv2d(
                in_channels,
                out_channels,
                padding_mode="reflect",
                **convolution_kwargs,
            )
            if downsample
            else nn.ConvTranspose2d(
                in_channels,
                out_channels,
                **convolution_kwargs,
            )
        )
        self.conv = nn.Sequential(
            convolution,
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_activation else nn.Identity(),
        )

    def forward(self, inputs):
        return self.conv(inputs)


class CycleGANResidualBlock(nn.Module):
    """Two-convolution residual block used by the CycleGAN generator."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            CycleGANConvBlock(channels, channels, kernel_size=3, padding=1),
            CycleGANConvBlock(
                channels,
                channels,
                use_activation=False,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, inputs):
        return inputs + self.block(inputs)


class CycleGANGenerator(nn.Module):
    """ResNet generator for one CycleGAN translation direction."""

    def __init__(
        self,
        image_channels,
        base_channels=64,
        num_residual_blocks=9,
    ):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                image_channels,
                base_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                CycleGANConvBlock(
                    base_channels,
                    base_channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                CycleGANConvBlock(
                    base_channels * 2,
                    base_channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[
                CycleGANResidualBlock(base_channels * 4)
                for _ in range(num_residual_blocks)
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                CycleGANConvBlock(
                    base_channels * 4,
                    base_channels * 2,
                    downsample=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                CycleGANConvBlock(
                    base_channels * 2,
                    base_channels,
                    downsample=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )
        self.last = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, inputs):
        hidden = self.initial(inputs)
        for down_block in self.down_blocks:
            hidden = down_block(hidden)
        hidden = self.res_blocks(hidden)
        for up_block in self.up_blocks:
            hidden = up_block(hidden)
        return torch.tanh(self.last(hidden))


class CycleGANDiscriminatorBlock(nn.Module):
    """PatchGAN downsampling block."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs):
        return self.conv(inputs)


class CycleGANDiscriminator(nn.Module):
    """PatchGAN discriminator for one CycleGAN image domain."""

    def __init__(
        self,
        in_channels=3,
        feature_channels=(64, 128, 256, 512),
    ):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                feature_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers = []
        previous_channels = feature_channels[0]
        for channels in feature_channels[1:]:
            layers.append(
                CycleGANDiscriminatorBlock(
                    previous_channels,
                    channels,
                    stride=1 if channels == feature_channels[-1] else 2,
                )
            )
            previous_channels = channels
        layers.append(
            nn.Conv2d(
                previous_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.model(self.initial(inputs))
        return torch.sigmoid(outputs)


class UnpairedImageDataset(Dataset):
    """Sample two image domains independently for unpaired translation."""

    def __init__(self, domain_a_roots, domain_b_roots, transform=None):
        super().__init__()
        self.transform = transform
        self.domain_a_paths = [
            os.path.join(root, filename)
            for root in domain_a_roots
            for filename in os.listdir(root)
        ]
        self.domain_b_paths = [
            os.path.join(root, filename)
            for root in domain_b_roots
            for filename in os.listdir(root)
        ]
        self.dataset_length = max(
            len(self.domain_a_paths),
            len(self.domain_b_paths),
        )
        self.domain_a_length = len(self.domain_a_paths)
        self.domain_b_length = len(self.domain_b_paths)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        domain_a_path = self.domain_a_paths[index % self.domain_a_length]
        domain_b_path = self.domain_b_paths[index % self.domain_b_length]
        domain_a_image = np.array(load_rgb_image(domain_a_path))
        domain_b_image = np.array(load_rgb_image(domain_b_path))
        if self.transform:
            transformed = self.transform(
                image=domain_b_image,
                image0=domain_a_image,
            )
            domain_b_image = transformed["image"]
            domain_a_image = transformed["image0"]
        return domain_a_image, domain_b_image


def initialize_cyclegan_weights(module):
    """Apply the CycleGAN normal initialization to learned affine layers."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


__all__ = [
    "LAMBDA_CYCLE",
    "CycleGANConvBlock",
    "CycleGANDiscriminator",
    "CycleGANDiscriminatorBlock",
    "CycleGANGenerator",
    "CycleGANResidualBlock",
    "UnpairedImageDataset",
    "initialize_cyclegan_weights",
    "save_cyclegan_translation_snapshot",
    "train_cyclegan_epoch",
]
