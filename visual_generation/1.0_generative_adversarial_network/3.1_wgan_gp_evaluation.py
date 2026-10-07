"""Generate evaluation samples from the WGAN-GP generator checkpoint.

Input:
    output/gan/wgan_gp/wgan_gp.pth, created by 3.0_wgan_gp.py

Output:
    output/gan/wgan_gp/evaluation/generated_samples.png
"""

import torch

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.dcgan import save_dcgan_sample_grid
from dl_utils.gan.wgan import WGANGenerator
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
OUT_DIR = PROJECT_ROOT / "output" / "gan" / "wgan_gp"
EVALUATION_DIR = OUT_DIR / "evaluation"

LIPSCHITZ = "gp"
Z_DIM = 100
GENERATOR_BASE_CHANNELS = 64
SAMPLES_TO_DISPLAY = 18
SAMPLE_GRID_COLUMNS = 6
EVALUATION_SEED = 10_043


@torch.inference_mode()
def evaluate_generator(checkpoint_path, output_path, device):
    """Load one WGAN generator and save deterministic fresh samples."""
    generator = WGANGenerator(
        z_dim=Z_DIM,
        base_channels=GENERATOR_BASE_CHANNELS,
    ).to(device)
    generator.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )
    generator.eval()

    random_generator = torch.Generator().manual_seed(EVALUATION_SEED)
    noise = torch.randn(
        SAMPLES_TO_DISPLAY,
        Z_DIM,
        1,
        1,
        generator=random_generator,
    ).to(device)
    save_dcgan_sample_grid(
        generator,
        noise,
        output_path,
        columns=SAMPLE_GRID_COLUMNS,
    )


def main():
    if LIPSCHITZ not in {"gp", "clip"}:
        raise ValueError("LIPSCHITZ must be 'gp' or 'clip'.")
    checkpoint_name = "wgan_gp.pth" if LIPSCHITZ == "gp" else "wgan.pth"
    checkpoint_path = OUT_DIR / checkpoint_name
    reset_dir(str(EVALUATION_DIR))
    if not checkpoint_path.is_file():
        print(
            f"Skipping WGAN-{LIPSCHITZ.upper()} evaluation: "
            f"{checkpoint_path} is missing. Run 3.0_wgan_gp.py first."
        )
        return

    output_path = EVALUATION_DIR / "generated_samples.png"
    evaluate_generator(checkpoint_path, output_path, try_gpu())
    print(f"Saved WGAN-{LIPSCHITZ.upper()} samples: {output_path}")


if __name__ == "__main__":
    main()
