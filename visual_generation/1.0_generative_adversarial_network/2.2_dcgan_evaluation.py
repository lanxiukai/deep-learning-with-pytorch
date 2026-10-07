"""Generate fresh evaluation samples from both DCGAN checkpoints.

Inputs:
    output/gan/dcgan/anime_face/generator.pth, created by
    2.0_dcgan_anime_face.py
    output/gan/dcgan/pokemon/generator.pth, created by
    2.1_dcgan_pokemon.py

Outputs:
    output/gan/dcgan/anime_face/evaluation/generated_samples.png
    output/gan/dcgan/pokemon/evaluation/generated_samples.png

Missing checkpoints are reported and skipped so either model can be evaluated
independently.
"""

import torch

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.dcgan import DCGANGenerator, save_dcgan_sample_grid
from dl_utils.runtime.devices import try_gpu

PROJECT_ROOT = infer_project_root()
DCGAN_OUT_DIR = PROJECT_ROOT / "output" / "gan" / "dcgan"

Z_DIM = 100
GENERATOR_BASE_CHANNELS = 64
SAMPLES_TO_DISPLAY = 18
SAMPLE_GRID_COLUMNS = 6
EVALUATION_SEED = 10_042

MODEL_SPECS = (
    (
        "Anime-face DCGAN",
        DCGAN_OUT_DIR / "anime_face" / "generator.pth",
        DCGAN_OUT_DIR / "anime_face" / "evaluation",
        EVALUATION_SEED + 1,
        "2.0_dcgan_anime_face.py",
    ),
    (
        "Pokémon DCGAN",
        DCGAN_OUT_DIR / "pokemon" / "generator.pth",
        DCGAN_OUT_DIR / "pokemon" / "evaluation",
        EVALUATION_SEED + 2,
        "2.1_dcgan_pokemon.py",
    ),
)


@torch.inference_mode()
def evaluate_generator(checkpoint_path, evaluation_dir, seed, device):
    """Load one generator and save a deterministic fresh sample grid."""
    generator = DCGANGenerator(
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

    random_generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        SAMPLES_TO_DISPLAY,
        Z_DIM,
        1,
        1,
        generator=random_generator,
    ).to(device)
    output_path = evaluation_dir / "generated_samples.png"
    save_dcgan_sample_grid(
        generator,
        noise,
        output_path,
        columns=SAMPLE_GRID_COLUMNS,
    )
    return output_path


def main():
    for _, _, evaluation_dir, _, _ in MODEL_SPECS:
        reset_dir(str(evaluation_dir))

    available_specs = []
    for spec in MODEL_SPECS:
        display_name, checkpoint_path, _, _, script_name = spec
        if checkpoint_path.is_file():
            available_specs.append(spec)
        else:
            print(
                f"Skipping {display_name}: {checkpoint_path} is missing. "
                f"Run {script_name} first."
            )
    if not available_specs:
        print("No DCGAN generator checkpoints are available; nothing to evaluate.")
        return

    device = try_gpu()
    for display_name, checkpoint_path, evaluation_dir, seed, _ in available_specs:
        output_path = evaluate_generator(
            checkpoint_path,
            evaluation_dir,
            seed,
            device,
        )
        print(f"Saved {display_name} samples: {output_path}")


if __name__ == "__main__":
    main()
