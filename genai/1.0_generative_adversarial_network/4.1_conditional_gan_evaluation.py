"""
Evaluate a previously trained Conditional GAN (cGAN) checkpoint — sample
faces with/without glasses, then interpolate in latent (z) and label space.

Requires output/gan/cgan/cgan.pth — run 4.0_conditional_gan.py first.

Outputs:
    output/gan/cgan/evaluation/, reset at the start of every run.
"""

import torch

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.conditional_gan import ConditionalGenerator
from dl_utils.plot._backend import pyplot as plt
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "cgan" / "evaluation"
CHECKPOINT_PATH = OUT_DIR.parent / "cgan.pth"

Z_DIM = 100
IMAGE_CHANNELS = 3
FEATURES = 16
SAMPLE_GRID_ROWS = 4
SAMPLE_GRID_COLUMNS = 8
NUM_SAMPLES = SAMPLE_GRID_ROWS * SAMPLE_GRID_COLUMNS
INTERPOLATION_WEIGHTS = (0, 0.25, 0.5, 0.75, 1)
SEED = 0


def main():
    """Load a trained cGAN checkpoint and run the full evaluation pipeline."""
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run 4.0_conditional_gan.py first."
        )
    # ensure a clean output directory before each run
    reset_dir(str(OUT_DIR))

    # determine the device automatically
    device = try_gpu()

    torch.manual_seed(SEED)

    generator = ConditionalGenerator(Z_DIM + 2, IMAGE_CHANNELS, FEATURES).to(device)
    generator.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    generator.eval()

    def _savefig(name):
        """Save current figure to OUT_DIR and close it."""
        path = OUT_DIR / f"{name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # ── Helper definitions ──────────────────────────────────────────
    # One‑hot label constants used throughout the evaluation.
    label_g = torch.zeros(2, 1, 1)
    label_g[0, :, :] = 1  # glasses
    label_ng = torch.zeros(2, 1, 1)
    label_ng[1, :, :] = 1  # no glasses

    def _denorm(image):
        """(C,H,W) tensor in [-1,1] → (H,W,C) numpy array in [0,1]."""
        return (image / 2 + 0.5).permute(1, 2, 0).numpy()

    def _show(image):
        """Display a single denormalised image tensor."""
        plt.imshow(_denorm(image))
        plt.xticks([])
        plt.yticks([])

    def _gen(noise, label):
        """Concatenate noise + label, run generator, return CPU tensor."""
        conditioned_noise = torch.cat([noise, label], dim=1).to(device)
        return generator(conditioned_noise).cpu().detach()

    def _plot_grid(images, rows, cols, name, figsize=None):
        """Plot (N,C,H,W) batch of images in a rows×cols grid."""
        if figsize is None:
            figsize = (cols * 2.5, rows * 2.5)
        plt.figure(figsize=figsize, dpi=300)
        for sample_index in range(len(images)):
            plt.subplot(rows, cols, sample_index + 1)
            _show(images[sample_index])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    def _sample_grid(
        num_samples,
        active_channel,
        name,
        rows=SAMPLE_GRID_ROWS,
        cols=SAMPLE_GRID_COLUMNS,
    ):
        """Generate labeled images, plot them, and return the sampled noise."""
        noise = torch.randn(num_samples, Z_DIM, 1, 1)
        labels = torch.zeros(num_samples, 2, 1, 1)
        labels[:, active_channel, :, :] = 1
        generated_images = _gen(noise, labels)
        _plot_grid(generated_images, rows, cols, name)
        return noise

    def _interp_1d(
        start_noise,
        end_noise,
        start_label,
        end_label,
        weights,
        name,
    ):
        """Plot 1×N row: interpolate z (a→b) AND label (a→b) together."""
        plt.figure(figsize=(len(weights) * 4, 4), dpi=300)
        for sample_index, weight in enumerate(weights):
            plt.subplot(1, len(weights), sample_index + 1)
            interpolated_noise = weight * end_noise + (1 - weight) * start_noise
            interpolated_label = weight * end_label + (1 - weight) * start_label
            generated_image = _gen(
                interpolated_noise.reshape(1, -1, 1, 1),
                interpolated_label.reshape(1, 2, 1, 1),
            )
            _show(generated_image[0])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    def _combo_2x2(first_noise, second_noise, first_label, second_label, name):
        """Plot 2×2: every combination of {z0, z1} × {label0, label1}."""
        plt.figure(figsize=(20, 5), dpi=300)
        for sample_index in range(4):
            plt.subplot(1, 4, sample_index + 1)
            noise_choice = sample_index // 2
            label_choice = sample_index % 2
            selected_noise = second_noise * noise_choice + first_noise * (
                1 - noise_choice
            )
            selected_label = second_label * label_choice + first_label * (
                1 - label_choice
            )
            generated_image = _gen(
                selected_noise.reshape(1, -1, 1, 1),
                selected_label.reshape(1, 2, 1, 1),
            )
            _show(generated_image[0])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    def _interp_grid_6x6(
        start_noise,
        end_noise,
        start_label,
        end_label,
        name,
    ):
        """Plot 6×6 grid: jointly interpolate z (a→b) and label (a→b)."""
        plt.figure(figsize=(20, 20), dpi=300)
        for sample_index in range(36):
            plt.subplot(6, 6, sample_index + 1)
            noise_weight = (sample_index // 6) / 5
            label_weight = (sample_index % 6) / 5
            interpolated_noise = end_noise * noise_weight + start_noise * (
                1 - noise_weight
            )
            interpolated_label = end_label * label_weight + start_label * (
                1 - label_weight
            )
            generated_image = _gen(
                interpolated_noise.reshape(1, -1, 1, 1),
                interpolated_label.reshape(1, 2, 1, 1),
            )
            _show(generated_image[0])
        plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
        _savefig(name)

    # ── 1. Sample faces with glasses / no‑glasses ──────────────────

    noise_g = _sample_grid(
        NUM_SAMPLES, active_channel=0, name="01_sample_glasses"
    )  # glasses
    noise_ng = _sample_grid(
        NUM_SAMPLES, active_channel=1, name="02_sample_no_glasses"
    )  # no glasses

    # Pick representative z vectors (indices from original notebook).
    z_male_g, z_female_g = noise_g[0], noise_g[20]
    z_male_ng, z_female_ng = noise_ng[8], noise_ng[12]

    # ── 2. Label interpolation (fix z, glasses → no glasses) ───────

    _interp_1d(
        z_female_g,
        z_female_g,
        label_g,
        label_ng,
        INTERPOLATION_WEIGHTS,
        "03_label_interp_female",
    )  # fix z (female), vary label
    _interp_1d(
        z_male_g,
        z_male_g,
        label_g,
        label_ng,
        INTERPOLATION_WEIGHTS,
        "04_label_interp_male",
    )  # fix z (male),   vary label — Exercise 5.1

    # ── 3. Latent interpolation (fix label, male → female) ──────────

    _interp_1d(
        z_male_ng,
        z_female_ng,
        label_ng,
        label_ng,
        INTERPOLATION_WEIGHTS,
        "05_latent_interp_no_glasses",
    )  # fix label (no glasses), vary z
    _interp_1d(
        z_male_ng,
        z_female_ng,
        label_g,
        label_g,
        INTERPOLATION_WEIGHTS,
        "06_latent_interp_glasses",
    )  # fix label (glasses),    vary z — Exercise 5.2

    # ── 4. 2×2 combinations (vary z × label) ────────────────────────

    _combo_2x2(
        z_male_g, z_female_g, label_g, label_ng, "07_combo_glasses_z"
    )  # z sampled w/ glasses prompt
    _combo_2x2(
        z_male_ng, z_female_ng, label_g, label_ng, "08_combo_no_glasses_z"
    )  # z sampled w/ no‑glasses prompt — Exercise 5.3

    # ── 5. 6×6 interpolation grid (vary z + label jointly) ──────────

    _interp_grid_6x6(z_male_ng, z_female_ng, label_g, label_ng, "09_interp_grid_6x6")


if __name__ == "__main__":
    main()
