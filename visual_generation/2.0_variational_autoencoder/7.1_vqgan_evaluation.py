"""Inspect VQGAN reconstructions and unconditional generation after training.

Data:
    data/glasses-256, prepared by tool_scripts/download_dataset.py --dataset glasses.
    Read the first 8 training images without shuffling.
    Resize to 256x256 RGB and normalize from [0, 1] to [-1, 1].
    Class labels are ignored for both reconstruction and generation.

Checkpoint:
    output/vae/vqgan/model.pth: final tokenizer/unconditional Transformer pair.
    Run 7.0 first. Model dimensions are loaded and checked.
    Recovery files tokenizer_latest.pth and prior_latest.pth are not used here.

Outputs:
    output/vae/vqgan/evaluation/vqgan_real_and_reconstruction.png
    output/vae/vqgan/evaluation/vqgan_prior_samples_temperature_<temperature>.png
    output/vae/vqgan/evaluation/vqgan_prior_beam_width_comparison.png
    output/vae/vqgan/evaluation/vqgan_prior_top_k_<k>_temperature_<temperature>.png
    output/vae/vqgan/evaluation/vqgan_prior_top_p_<p>_temperature_<temperature>.png
    The evaluation directory is reset before writing its 20 default grids.
    Greedy and beam results are saved only in the combined comparison grid.

Evaluation defaults:
    Reconstruction images: 8 training inputs; one batch.
    Generated images:      8 per stochastic run; one greedy image; best beam per width.
    Layout:                At most 4 images per sampling row; 256x256 RGB.
    Beam comparison:       Greedy (width 1), then one column per beam width.
    Token grid / sequence: 16x16 / 256 tokens with default four downsampling stages.
    Prior temperatures:    0.2, 0.5, 1.0, 2.0, 4.0, 8.0 (sharp to very random).
    Beam widths:           1 (greedy), 2, 4, 8, 16, 32; best image from each.
    Top-k / p:             50 / 0.9.
    Seed:                  123 for repeatable unconditional sampling.
    Model dimensions:      loaded from the checkpoint produced by 7.0.

Run without arguments after training 7.0; edit the constants below to change
the checkpoint, image counts, generation methods, beam widths, top-k/top-p
thresholds, or sampling temperatures. Greedy and beam search are deterministic
for this unconditional fixed-length prior and do not apply temperature.
"""

import math
from typing import Literal

import torch
from tqdm.auto import tqdm

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.vae.causal_transformer import CausalTransformerPrior
from dl_utils.vae.discrete_workflow import glasses_loader, load_pair

# Paths and checkpoints
PROJECT_ROOT = infer_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "glasses-256"
CHECKPOINT = PROJECT_ROOT / "output" / "vae" / "vqgan" / "model.pth"
OUTPUT_DIR = CHECKPOINT.parent / "evaluation"

# Image configuration
IMAGE_SIZE = 256
RECONSTRUCTION_SAMPLES = 8
NUM_SAMPLES = 8
SAMPLES_PER_ROW = 4

# Sampling configuration
GenerationMethod = Literal["multinomial", "greedy", "beam", "top_k", "top_p"]
GENERATION_METHODS: tuple[GenerationMethod, ...] = (
    "multinomial",
    "greedy",
    "beam",
    "top_k",
    "top_p",
)
TEMPERATURES = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)
# Width 1 is supplied by greedy decoding in the comparison grid.
BEAM_WIDTHS = (2, 4, 8, 16, 32)
TOP_K = 50
TOP_P = 0.9
SEED = 123


@torch.inference_mode()
def evaluate():
    if not GENERATION_METHODS or any(
        method not in {"multinomial", "greedy", "beam", "top_k", "top_p"}
        for method in GENERATION_METHODS
    ):
        raise ValueError(
            "GENERATION_METHODS must contain supported generation methods."
        )
    stochastic = any(
        method in {"multinomial", "top_k", "top_p"} for method in GENERATION_METHODS
    )
    if stochastic and (
        not TEMPERATURES
        or any(
            not math.isfinite(temperature) or temperature <= 0
            for temperature in TEMPERATURES
        )
    ):
        raise ValueError("TEMPERATURES must contain finite, positive values.")
    if min(RECONSTRUCTION_SAMPLES, NUM_SAMPLES, SAMPLES_PER_ROW) < 1:
        raise ValueError("Image counts and row width must be positive.")
    if "beam" in GENERATION_METHODS and (
        not BEAM_WIDTHS or any(width < 1 for width in BEAM_WIDTHS)
    ):
        raise ValueError("BEAM_WIDTHS must contain positive widths.")
    if "top_p" in GENERATION_METHODS and (
        not math.isfinite(TOP_P) or not 0 < TOP_P <= 1
    ):
        raise ValueError("TOP_P must be finite and in (0, 1].")
    generation_runs: list[tuple[GenerationMethod, float, int]] = [
        (method, temperature, beam_width)
        for method in GENERATION_METHODS
        for temperature in (
            TEMPERATURES if method in {"multinomial", "top_k", "top_p"} else (1.0,)
        )
        for beam_width in (BEAM_WIDTHS if method == "beam" else (1,))
    ]
    beam_images: list[torch.Tensor] = []
    greedy_image: torch.Tensor | None = None
    set_seed(SEED)
    device = try_gpu()
    with tqdm(
        total=1
        + len(generation_runs)
        + int(any(method in {"greedy", "beam"} for method in GENERATION_METHODS)),
        desc="Evaluate VQGAN",
        unit="step",
    ) as progress:
        tokenizer, prior = load_pair(CHECKPOINT, "vqgan", device, image_size=IMAGE_SIZE)
        assert isinstance(prior, CausalTransformerPrior)
        if "top_k" in GENERATION_METHODS and not 1 <= TOP_K <= prior.vocabulary_size:
            raise ValueError(
                "TOP_K must be between 1 and the checkpoint vocabulary size."
            )
        reset_dir(OUTPUT_DIR)
        loader = glasses_loader(DATA_DIR, IMAGE_SIZE, RECONSTRUCTION_SAMPLES, device)
        originals = next(iter(loader))[0].to(device)
        reconstruction = tokenizer(originals)[0]
        side = IMAGE_SIZE // (2**tokenizer.downsample_steps)
        save_image_row_grid(
            [originals, reconstruction],
            ["Original", "VQGAN"],
            OUTPUT_DIR / "vqgan_real_and_reconstruction.png",
            title="Same training images: reconstruction",
            dpi=160,
        )
        progress.update(1)
        for method, temperature, beam_width in generation_runs:
            progress.set_postfix(method=method, refresh=False)
            with torch.random.fork_rng():
                torch.manual_seed(SEED)
                if method == "beam":
                    indices = prior.beam_search(beam_width, device=device)[:1]
                elif method == "greedy":
                    indices = prior.sample(1, device=device, method="greedy")
                else:
                    indices = prior.sample(
                        NUM_SAMPLES,
                        device=device,
                        method=method,
                        temperature=temperature,
                        top_k=TOP_K,
                        top_p=TOP_P,
                    )
                samples = tokenizer.decode_indices(
                    indices.reshape(len(indices), side, side)
                )
            if method == "beam":
                beam_images.append(samples.cpu())
            elif method == "greedy":
                greedy_image = samples.cpu()
            if method in {"greedy", "beam"}:
                progress.update(1)
                continue
            if method == "top_k":
                suffix = f"top_k_{TOP_K}_temperature_{temperature}"
                title = f"Top-k sampling: k={TOP_K}, temperature={temperature}"
            elif method == "top_p":
                suffix = f"top_p_{TOP_P}_temperature_{temperature}"
                title = f"Top-p sampling: p={TOP_P}, temperature={temperature}"
            else:
                suffix = f"samples_temperature_{temperature}"
                title = f"Unconditional prior samples: temperature={temperature}"
            # The row-grid helper requires equal row lengths.
            columns = min(SAMPLES_PER_ROW, len(samples))
            while len(samples) % columns:
                columns -= 1
            rows = samples.split(columns)
            save_image_row_grid(
                rows,
                ["VQGAN"] * len(rows),
                OUTPUT_DIR / f"vqgan_prior_{suffix}.png",
                title=title,
                dpi=160,
            )
            progress.update(1)
        if beam_images or greedy_image is not None:
            comparison_images = beam_images
            comparison_labels = (
                [f"Width {width}" for width in BEAM_WIDTHS] if beam_images else []
            )
            if greedy_image is not None:
                comparison_images = [greedy_image, *comparison_images]
                comparison_labels = ["Greedy (width 1)", *comparison_labels]
            save_image_row_grid(
                [torch.cat(comparison_images)],
                ["VQGAN"],
                OUTPUT_DIR / "vqgan_prior_beam_width_comparison.png",
                title="Beam width comparison: best candidate from each search",
                column_labels=comparison_labels,
                dpi=160,
            )
            progress.update(1)


if __name__ == "__main__":
    evaluate()
