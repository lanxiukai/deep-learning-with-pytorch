"""
Generative Adversarial Networks
"""

import torch
from torch import nn
from tqdm import tqdm

from dl_utils.data.vision import load_array
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.gan import update_discriminator, update_generator
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.curves import save_loss_panels
from dl_utils.plot.figures import set_figsize
from dl_utils.training.metrics import MetricAccumulator

PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "gan"

NUM_EPOCHS = 20
BATCH_SIZE = 8
Z_DIM = 2
LEARNING_RATE_D = 0.05
LEARNING_RATE_G = 0.005
NUM_DATA_POINTS = 1000
NUM_SAMPLES_TO_DISPLAY = 100


def train(
    discriminator,
    generator,
    data_loader,
    num_epochs,
    discriminator_lr,
    generator_lr,
    z_dim,
    real_samples,
    output_dir,
):
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    # Parameter initialization
    for parameter in discriminator.parameters():
        nn.init.normal_(parameter, 0, 0.02)
    for parameter in generator.parameters():
        nn.init.normal_(parameter, 0, 0.02)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=discriminator_lr
    )
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_lr)
    epochs, discriminator_losses, generator_losses = [], [], []

    with tqdm(
        total=num_epochs * len(data_loader),
        desc=f"Epoch 1/{num_epochs}",
        unit="batch",
        dynamic_ncols=True,
    ) as progress_bar:
        for epoch in range(1, num_epochs + 1):
            progress_bar.set_description(f"Epoch {epoch}/{num_epochs}", refresh=False)
            # Train one epoch
            metric = MetricAccumulator(
                ("discriminator", "generator"),
                device=next(discriminator.parameters()).device,
            )
            for (real_batch,) in data_loader:
                batch_size = real_batch.shape[0]
                noise = torch.normal(0, 1, size=(batch_size, z_dim))
                discriminator_loss = update_discriminator(
                    real_batch,
                    noise,
                    discriminator,
                    generator,
                    loss,
                    discriminator_optimizer,
                )
                generator_loss = update_generator(
                    noise,
                    discriminator,
                    generator,
                    loss,
                    generator_optimizer,
                )
                metric.add_batch_means(
                    (
                        discriminator_loss / batch_size,
                        generator_loss / batch_size,
                    ),
                    num_examples=batch_size,
                )
                progress_bar.update(1)
            # Show the losses
            epoch_metrics = metric.compute_weighted_means()
            discriminator_loss = epoch_metrics["discriminator"]
            generator_loss = epoch_metrics["generator"]
            epochs.append(epoch)
            discriminator_losses.append(discriminator_loss)
            generator_losses.append(generator_loss)
            progress_bar.set_postfix(
                loss_D=f"{discriminator_loss:.3f}",
                loss_G=f"{generator_loss:.3f}",
            )
    print(f"loss_D {discriminator_loss:.3f}, loss_G {generator_loss:.3f}")

    # --- Save loss curve ---
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

    # --- Save scatter comparison (final generated vs real) ---
    noise = torch.normal(0, 1, size=(NUM_SAMPLES_TO_DISPLAY, z_dim))
    generated_samples = generator(noise).detach().numpy()
    set_figsize((4, 4))
    plt.scatter(real_samples[:, 0], real_samples[:, 1], label="real", alpha=0.5)
    plt.scatter(
        generated_samples[:, 0],
        generated_samples[:, 1],
        label="generated",
        alpha=0.5,
    )
    plt.legend()
    plt.savefig(output_dir / "scatter_comparison.png", dpi=300)
    plt.close()


def main():
    standard_samples = torch.normal(0.0, 1, (NUM_DATA_POINTS, 2))
    transformation_matrix = torch.tensor([[1, 2], [-0.1, 0.5]])
    bias = torch.tensor([1, 2])
    transformed_samples = torch.matmul(standard_samples, transformation_matrix) + bias

    reset_dir(str(OUT_DIR))

    set_figsize()
    plt.scatter(
        transformed_samples[:NUM_SAMPLES_TO_DISPLAY, 0].detach().numpy(),
        transformed_samples[:NUM_SAMPLES_TO_DISPLAY, 1].detach().numpy(),
    )
    plt.savefig(OUT_DIR / "data_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    covariance_matrix = torch.matmul(
        transformation_matrix.T,
        transformation_matrix,
    )
    print(f"The covariance matrix is\n{covariance_matrix}")

    data_loader = load_array((transformed_samples,), BATCH_SIZE)

    generator = nn.Sequential(nn.Linear(2, 2))

    discriminator = nn.Sequential(
        nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 1)
    )

    train(
        discriminator,
        generator,
        data_loader,
        NUM_EPOCHS,
        LEARNING_RATE_D,
        LEARNING_RATE_G,
        Z_DIM,
        transformed_samples[:NUM_SAMPLES_TO_DISPLAY].detach().numpy(),
        OUT_DIR,
    )


if __name__ == "__main__":
    main()
