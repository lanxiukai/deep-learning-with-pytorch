"""Visually inspect VQGAN reconstructions and conditional samples on glasses-256.

Run 7.0 first. Only eight seeded training images are reconstructed, and four
independent samples are generated per class: G (with glasses) and NoG (without
glasses). Two labeled grids are saved; no quantitative metrics, LPIPS network,
or PatchGAN discriminator are needed. These previews do not measure held-out
performance or distributional sample quality.

Loading requires matching tokenizer/prior snapshot IDs, image preprocessing,
conditioning and the current four-step spatial compression. Different
conditioning, backbones, objectives, priors and training budgets prevent a
controlled algorithm ranking.
Outputs remain in output/vae/vqgan/evaluation/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from dl_utils.data.datasets.glasses import GLASSES_CLASS_NAMES
from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.discrete_workflow import (
    glasses_loader,
    load_prior_weights,
    load_tokenizer_weights,
)
from dl_utils.vae.perceptual_autoencoder import VQPerceptualAutoencoder
from dl_utils.vae.quantization import TOKENIZER_DOWNSAMPLE_STEPS
from dl_utils.vae.token_priors import CausalTransformerPrior

PROJECT_ROOT = infer_project_root()
OUTPUT_ROOT = PROJECT_ROOT / "output" / "vae"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
IMAGE_SIZE = 256
NUM_CLASSES = len(GLASSES_CLASS_NAMES)
DOWNSAMPLE_STEPS = TOKENIZER_DOWNSAMPLE_STEPS


# Edit these defaults to explore the lesson.
DATA_DIR = DEFAULT_DATA_DIR
RECONSTRUCTION_SAMPLES = 8
SAMPLES_PER_CLASS = 4
GENERATION_BATCH_SIZE = 8
TEMPERATURE = 1.0
WORKERS = 0
SEED = 123
VQGAN_TOKENIZER = OUTPUT_ROOT / "vqgan" / "vqgan.pth"
OUTPUT_DIR = VQGAN_TOKENIZER.parent / "evaluation"
PRIOR_CHECKPOINT_NAME = "transformer_prior.pth"


@dataclass
class EvaluatedSystem:
    tokenizer: VQPerceptualAutoencoder
    prior: CausalTransformerPrior

    def reconstruct(self, images: Tensor) -> Tensor:
        return self.tokenizer(images)[0]

    @torch.inference_mode()
    def sample(
        self,
        count: int,
        *,
        device: torch.device,
        labels: Tensor,
        temperature: float,
    ) -> Tensor:
        side = IMAGE_SIZE // (2**self.tokenizer.downsample_steps)
        indices = self.prior.sample(
            count,
            device=device,
            labels=labels,
            temperature=temperature,
        ).reshape(count, side, side)
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
        conditional=True,
    )
    return loader


def load_vqgan_system(tokenizer_path: Path, device: torch.device) -> EvaluatedSystem:
    tokenizer, payload = load_tokenizer_weights(
        tokenizer_path,
        VQPerceptualAutoencoder,
        name="vqgan_tokenizer",
        image_size=IMAGE_SIZE,
        device=device,
        downsample_steps=DOWNSAMPLE_STEPS,
        conditional=True,
    )
    prior = load_prior_weights(
        tokenizer_path.with_name(PRIOR_CHECKPOINT_NAME),
        CausalTransformerPrior,
        name="vqgan_transformer_prior",
        image_size=IMAGE_SIZE,
        tokenizer=tokenizer,
        tokenizer_payload=payload,
        device=device,
        conditional=True,
    )
    return EvaluatedSystem(tokenizer, prior)


@torch.inference_mode()
def evaluate() -> None:
    set_seed(SEED)
    device = try_gpu()
    system = load_vqgan_system(VQGAN_TOKENIZER, device)
    images, _ = next(iter(make_evaluation_loader(device)))
    originals = images.to(device, non_blocking=True)
    reconstruction = system.reconstruct(originals).cpu()
    sample_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    samples = generate_in_batches(
        sample_labels,
        GENERATION_BATCH_SIZE,
        lambda labels: system.sample(
            len(labels), device=device, labels=labels, temperature=TEMPERATURE
        ),
    )
    reset_dir(str(OUTPUT_DIR))
    save_image_row_grid(
        [originals, reconstruction],
        ["Original", "VQGAN"],
        OUTPUT_DIR / "vqgan_real_and_reconstruction.png",
        title="Same training images: reconstruction",
        dpi=160,
    )
    save_image_row_grid(
        samples.split(SAMPLES_PER_CLASS),
        ["G (with glasses)", "NoG (without glasses)"],
        OUTPUT_DIR / "vqgan_prior_samples.png",
        title="Independent class-conditional prior samples",
        dpi=160,
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
