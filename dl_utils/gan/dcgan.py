"""Shared DCGAN models, training, and sample rendering."""

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.gan.training import update_discriminator, update_generator
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.curves import save_loss_panels
from dl_utils.plot.figures import plot
from dl_utils.runtime.devices import try_gpu
from dl_utils.training.metrics import MetricAccumulator


class DCGANGeneratorBlock(nn.Module):
    """Upsample with transposed convolution, BatchNorm, and ReLU."""

    def __init__(
        self,
        out_channels,
        in_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
    ):
        super().__init__()
        # Keep the original attribute name for checkpoint compatibility.
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        hidden = self.conv2d_trans(inputs)
        return self.activation(self.batch_norm(hidden))


class DCGANDiscriminatorBlock(nn.Module):
    """Downsample with convolution, BatchNorm, and LeakyReLU."""

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
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, inputs):
        hidden = self.conv2d(inputs)
        return self.activation(self.batch_norm(hidden))


class DCGANGenerator(nn.Sequential):
    """Generate 64 x 64 RGB images from ``N x z_dim x 1 x 1`` noise."""

    def __init__(self, z_dim=100, image_channels=3, base_channels=64):
        super().__init__(
            DCGANGeneratorBlock(
                in_channels=z_dim,
                out_channels=base_channels * 8,
                stride=1,
                padding=0,
            ),
            DCGANGeneratorBlock(
                in_channels=base_channels * 8,
                out_channels=base_channels * 4,
            ),
            DCGANGeneratorBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 2,
            ),
            DCGANGeneratorBlock(
                in_channels=base_channels * 2,
                out_channels=base_channels,
            ),
            nn.ConvTranspose2d(
                in_channels=base_channels,
                out_channels=image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )


class DCGANDiscriminator(nn.Sequential):
    """Score 64 x 64 RGB images with a DCGAN discriminator."""

    def __init__(self, image_channels=3, base_channels=64):
        super().__init__(
            DCGANDiscriminatorBlock(
                in_channels=image_channels,
                out_channels=base_channels,
            ),
            DCGANDiscriminatorBlock(
                in_channels=base_channels,
                out_channels=base_channels * 2,
            ),
            DCGANDiscriminatorBlock(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
            ),
            DCGANDiscriminatorBlock(
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


def initialize_dcgan_weights(model):
    """Initialize every trainable parameter from N(0, 0.02)."""
    for parameter in model.parameters():
        nn.init.normal_(parameter, 0, 0.02)


def save_leaky_relu_curves(output_path):
    """Save the discriminator activation family used by both lessons."""
    negative_slopes = [0, 0.2, 0.4, 0.6, 0.8, 1]
    activation_inputs = torch.arange(-2, 1, 0.1)
    activation_outputs = [
        nn.LeakyReLU(slope)(activation_inputs).detach().numpy()
        for slope in negative_slopes
    ]
    plot(
        activation_inputs.numpy(),
        activation_outputs,
        "x",
        "y",
        negative_slopes,
    )
    plt.savefig(output_path, dpi=300)
    plt.close()


@torch.inference_mode()
def save_dcgan_sample_grid(generator, noise, output_path, *, columns):
    """Save generated images in the compact untitled grid used by the lessons."""
    if columns < 1 or len(noise) == 0 or len(noise) % columns != 0:
        raise ValueError("noise must contain a positive multiple of columns.")

    was_training = generator.training
    generator.eval()
    try:
        generated_images = generator(noise).permute(0, 2, 3, 1).mul(0.5).add(0.5).cpu()
    finally:
        generator.train(was_training)

    image_grid = torch.cat(
        [
            torch.cat(
                [
                    generated_images[row_index * columns + column_index]
                    for column_index in range(columns)
                ],
                dim=1,
            )
            for row_index in range(len(generated_images) // columns)
        ],
        dim=0,
    )
    plt.imsave(output_path, image_grid.clamp(0, 1).numpy())


def train_dcgan(
    generator,
    discriminator,
    data_loader,
    *,
    num_epochs,
    generator_lr,
    discriminator_lr,
    z_dim,
    output_dir,
    samples_to_display,
    sample_grid_columns,
    sample_every_epochs,
    device=None,
):
    """Train a DCGAN and save fixed-z samples, loss panels, and final weights."""
    if num_epochs < 1 or sample_every_epochs < 1:
        raise ValueError("epoch counts must be positive.")

    output_dir = Path(output_dir)
    training_dir = output_dir / "training"
    reset_dir(str(training_dir))
    device = try_gpu() if device is None else device
    initialize_dcgan_weights(generator)
    initialize_dcgan_weights(discriminator)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_lr,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=(0.5, 0.999),
    )
    fixed_noise = torch.randn(samples_to_display, z_dim, 1, 1, device=device)
    epochs, discriminator_losses, generator_losses = [], [], []

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
            metrics = MetricAccumulator(
                ("discriminator", "generator"),
                device=device,
            )
            for real_images, _ in data_loader:
                real_images = real_images.to(device, non_blocking=True)
                batch_size = real_images.shape[0]
                noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

                discriminator_loss = update_discriminator(
                    real_images,
                    noise,
                    discriminator,
                    generator,
                    loss_function,
                    discriminator_optimizer,
                    real_label=0.9,
                )
                generator_loss = update_generator(
                    noise,
                    discriminator,
                    generator,
                    loss_function,
                    generator_optimizer,
                )
                metrics.add_batch_means(
                    (
                        discriminator_loss / batch_size,
                        generator_loss / batch_size,
                    ),
                    num_examples=batch_size,
                )
                progress_bar.update(1)

            epoch_metrics = metrics.compute_weighted_means()
            discriminator_loss = epoch_metrics["discriminator"]
            generator_loss = epoch_metrics["generator"]
            epochs.append(epoch)
            discriminator_losses.append(discriminator_loss)
            generator_losses.append(generator_loss)
            progress_bar.set_postfix(
                loss_D=f"{discriminator_loss:.3f}",
                loss_G=f"{generator_loss:.3f}",
            )

            if epoch == 1 or epoch % sample_every_epochs == 0:
                save_dcgan_sample_grid(
                    generator,
                    fixed_noise,
                    training_dir / f"epoch_{epoch:03d}.png",
                    columns=sample_grid_columns,
                )

    print(f"loss_D {discriminator_loss:.3f}, loss_G {generator_loss:.3f} on {device}")
    save_loss_panels(
        epochs,
        {
            "Discriminator loss": {"D loss": discriminator_losses},
            "Generator loss": {"G loss": generator_losses},
        },
        output_dir / "loss_curves.png",
    )
    torch.save(discriminator.state_dict(), output_dir / "discriminator.pth")
    torch.save(generator.state_dict(), output_dir / "generator.pth")


__all__ = [
    "DCGANDiscriminator",
    "DCGANDiscriminatorBlock",
    "DCGANGenerator",
    "DCGANGeneratorBlock",
    "initialize_dcgan_weights",
    "save_dcgan_sample_grid",
    "save_leaky_relu_curves",
    "train_dcgan",
]
