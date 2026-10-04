"""Visually compare Gaussian VAE, VQ-VAE and FSQ on glasses-256.

Run 1.0, 6.0 and 6.1 first. All three models use the same 256px training
images and unconditional generation. Two labeled grids compare posterior-mean
VAE / discrete reconstructions and independent prior samples. Columns in the
generation grid do not represent matched identities.

Only eight seeded training images are reconstructed, and each model generates
eight samples. No quantitative metrics or separate per-model grids are saved.
Loading checks the dataset, architecture, conditioning and tokenizer/prior pair.
Outputs remain in output/vae/evaluation/discrete_tokenizer/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights
from dl_utils.vae.discrete_workflow import (
    glasses_loader,
    load_prior_weights,
    load_tokenizer_weights,
)
from dl_utils.vae.quantization import (
    TOKENIZER_DOWNSAMPLE_STEPS,
    VQVAE,
    FSQAutoencoder,
)
from dl_utils.vae.token_priors import PixelCNNPrior
from dl_utils.vae.vae import VAE

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
OUTPUT_DIR = OUTPUT_ROOT / "evaluation" / "discrete_tokenizer"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
IMAGE_SIZE = 256
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
RECONSTRUCTION_SAMPLES = 8
GENERATION_SAMPLES = 8
GENERATION_BATCH_SIZE = 8
TEMPERATURE = 1.0
WORKERS = 0
SEED = 123
VAE_CHECKPOINT = OUTPUT_ROOT / "vae" / "vae.pth"
VQ_VAE_TOKENIZER = OUTPUT_ROOT / "vq_vae" / "vq_vae.pth"
VQ_VAE_PRIOR = OUTPUT_ROOT / "vq_vae" / "pixelcnn_prior.pth"
FSQ_TOKENIZER = OUTPUT_ROOT / "fsq" / "fsq.pth"
FSQ_PRIOR = OUTPUT_ROOT / "fsq" / "pixelcnn_prior.pth"


@dataclass
class DiscreteSystem:
    name: str
    tokenizer: VQVAE | FSQAutoencoder
    prior: PixelCNNPrior
    image_size: int

    @property
    def latent_grid_size(self) -> int:
        return self.image_size // (2**self.tokenizer.downsample_steps)

    def reconstruct(self, images: Tensor) -> Tensor:
        return self.tokenizer(images)[0]

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        temperature: float,
    ) -> Tensor:
        indices = self.prior.sample(
            count,
            self.latent_grid_size,
            self.latent_grid_size,
            device=device,
            temperature=temperature,
        )
        return self.tokenizer.decode_indices(indices)


def make_evaluation_loader(device: torch.device):
    loader, _ = glasses_loader(
        DATA_DIR,
        IMAGE_SIZE,
        RECONSTRUCTION_SAMPLES,
        device,
        max_examples=RECONSTRUCTION_SAMPLES,
        seed=SEED,
        num_workers=WORKERS,
    )
    return loader


def load_single_level_system(
    *,
    name: str,
    tokenizer_path: Path,
    prior_path: Path,
    device: torch.device,
) -> DiscreteSystem:
    model_class = VQVAE if name == "vq_vae" else FSQAutoencoder
    tokenizer, payload = load_tokenizer_weights(
        tokenizer_path,
        model_class,
        name=f"{name}_tokenizer",
        image_size=IMAGE_SIZE,
        device=device,
        downsample_steps=DOWNSAMPLE_STEPS,
    )
    prior = load_prior_weights(
        prior_path,
        PixelCNNPrior,
        name=f"{name}_pixelcnn_prior",
        image_size=IMAGE_SIZE,
        tokenizer=tokenizer,
        tokenizer_payload=payload,
        device=device,
    )
    return DiscreteSystem(name, tokenizer, prior, IMAGE_SIZE)


def load_systems(device: torch.device) -> list[DiscreteSystem]:
    systems = [
        load_single_level_system(
            name="vq_vae",
            tokenizer_path=VQ_VAE_TOKENIZER,
            prior_path=VQ_VAE_PRIOR,
            device=device,
        ),
        load_single_level_system(
            name="fsq",
            tokenizer_path=FSQ_TOKENIZER,
            prior_path=FSQ_PRIOR,
            device=device,
        ),
    ]
    return systems


@torch.inference_mode()
def save_vae_comparisons(
    vae: VAE,
    originals: Tensor,
    reconstructions: dict[str, Tensor],
    generations: dict[str, Tensor],
    *,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Save two labeled grids; the Gaussian baseline keeps its [0, 1] input."""
    vae.eval()
    _, _, vae_reconstruction = vae.reconstruct(
        originals.mul(0.5).add(0.5), sample=False
    )
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        vae_samples = vae.decoder(
            torch.randn(GENERATION_SAMPLES, vae.z_dim, device=device)
        )
    names = list(reconstructions)
    labels = [{"vq_vae": "VQ-VAE", "fsq": "FSQ"}[name] for name in names]
    save_image_row_grid(
        [originals, vae_reconstruction.mul(2).sub(1)]
        + [reconstructions[name] for name in names],
        ["Original", "VAE (mean)"] + labels,
        out_dir / "vae_vq_vae_fsq_reconstructions.png",
        title="Same training images: reconstruction",
        dpi=160,
    )
    save_image_row_grid(
        [vae_samples.mul(2).sub(1)] + [generations[name] for name in names],
        ["VAE"] + labels,
        out_dir / "vae_vq_vae_fsq_samples.png",
        title="Independent unconditional prior samples",
        dpi=160,
    )


@torch.inference_mode()
def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    images, _ = next(iter(make_evaluation_loader(device)))
    originals = images.to(device, non_blocking=True)
    systems = load_systems(device)
    vae, _ = load_model_weights(
        VAE_CHECKPOINT,
        VAE,
        device=device,
        expected_metadata={
            "model_name": "vae",
            "backbone": VAE.backbone,
            "dataset": "glasses-256",
            "value_range": [0.0, 1.0],
            "beta": 1.0,
        },
    )
    vae = cast(VAE, vae)
    out_dir = OUTPUT_DIR
    reset_dir(str(out_dir))
    reconstructions: dict[str, Tensor] = {}
    generations: dict[str, Tensor] = {}
    for system in systems:
        reconstructions[system.name] = system.reconstruct(originals).cpu()
        with torch.random.fork_rng():
            torch.manual_seed(SEED)
            generations[system.name] = generate_in_batches(
                torch.arange(GENERATION_SAMPLES, device=device),
                GENERATION_BATCH_SIZE,
                lambda batch, system=system: system.sample(
                    len(batch), device=device, temperature=TEMPERATURE
                ),
            )
    save_vae_comparisons(
        vae, originals, reconstructions, generations, device=device, out_dir=out_dir
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
