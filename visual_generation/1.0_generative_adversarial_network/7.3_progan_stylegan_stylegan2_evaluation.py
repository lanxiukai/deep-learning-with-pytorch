"""Qualitatively evaluate the ProGAN to StyleGAN2 lesson sequence.

The evaluation separates the mechanisms introduced across the three lessons:
    - all generators receive shared fixed z samples;
    - StyleGAN and StyleGAN2 vary stochastic noise while holding W fixed;
    - their 4-32 structure styles and 64-128 detail styles are mixed;
    - their learned ``w_avg`` buffers drive truncation comparisons;
    - spherical Z interpolation is compared with linear W interpolation.

Shared inputs fix random conditions only.  The independently trained latent
spaces do not have sample-wise semantic correspondence.  This compact lesson
uses current metadata-rich checkpoints and intentionally omits legacy formats,
FID reproduction, and a large evaluation framework.

Inputs:
    output/gan/progan/progan_generator.pth, created by 7.0_progan.py
    output/gan/stylegan/stylegan_generator.pth, created by 7.1_stylegan.py
    output/gan/stylegan2/stylegan2_generator.pth, created by 7.2_stylegan2.py

Outputs:
    Each evaluation directory is reset after inputs have been validated.
    output/gan/{progan,stylegan,stylegan2}/evaluation/fixed_samples.png
    output/gan/{stylegan,stylegan2}/evaluation/noise_variations.png
    output/gan/stylegan/evaluation/stylegan_style_mixing.png
    output/gan/stylegan2/evaluation/stylegan2_style_mixing.png
    output/gan/{stylegan,stylegan2}/evaluation/truncation.png
    output/gan/{progan,stylegan,stylegan2}/evaluation/latent_interpolation.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dl_utils.filesystem.directories import reset_dir
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.gan.progan import ProGANGenerator
from dl_utils.gan.stylegan import StyleGANGenerator
from dl_utils.gan.stylegan2 import StyleGenerator
from dl_utils.inference.batching import generate_in_batches
from dl_utils.plot.images import save_image_row_grid
from dl_utils.runtime.devices import try_gpu
from dl_utils.runtime.randomness import set_seed
from dl_utils.training.checkpoints import load_model_weights

PROJECT_ROOT = infer_project_root()
MODEL_OUT_DIRS = {
    model_name: PROJECT_ROOT / "output" / "gan" / model_name
    for model_name in ("progan", "stylegan", "stylegan2")
}
EVALUATION_DIRS = {
    model_name: output_dir / "evaluation"
    for model_name, output_dir in MODEL_OUT_DIRS.items()
}

SEED = 42
NUM_FIXED_SAMPLES = 8
NUM_NOISE_VARIATIONS = 8
NUM_STYLE_SOURCES = 4
NUM_INTERPOLATION_STEPS = 9
FINAL_RESOLUTION = 128
STRUCTURE_MAX_RESOLUTION = 32
INFERENCE_BATCH_SIZE = 4
TRUNCATION_PSIS = (1.0, 0.7, 0.5, 0.0)

MODEL_SPECS = (
    (
        "progan",
        ProGANGenerator,
        MODEL_OUT_DIRS["progan"] / "progan_generator.pth",
        "7.0_progan.py",
    ),
    (
        "stylegan",
        StyleGANGenerator,
        MODEL_OUT_DIRS["stylegan"] / "stylegan_generator.pth",
        "7.1_stylegan.py",
    ),
    (
        "stylegan2",
        StyleGenerator,
        MODEL_OUT_DIRS["stylegan2"] / "stylegan2_generator.pth",
        "7.2_stylegan2.py",
    ),
)


@dataclass(frozen=True)
class GeneratorAdapter:
    """Present the three lesson generators through one evaluation API."""

    model_name: str
    generator: nn.Module

    @property
    def num_ws(self):
        return self.generator.num_ws

    def mapping(self, z):
        return self.generator.mapping(z)

    def num_ws_for_resolution(self, resolution):
        return self.generator.num_ws_for_resolution(resolution)

    def generate(self, z, *, noise_mode="fixed", truncation_psi=None):
        if self.model_name == "progan":
            if truncation_psi is not None:
                raise ValueError("ProGAN does not support truncation.")
            return self.generator(
                z,
                resolution=FINAL_RESOLUTION,
                alpha=1.0,
            )

        kwargs = {"noise_mode": noise_mode}
        if truncation_psi is not None:
            kwargs["truncation_psi"] = truncation_psi
        if self.model_name == "stylegan":
            return self.generator(
                z,
                resolution=FINAL_RESOLUTION,
                alpha=1.0,
                **kwargs,
            )
        if self.model_name == "stylegan2":
            return self.generator(z, **kwargs)
        raise ValueError(f"Unknown generator family: {self.model_name}.")

    def synthesize(self, ws, *, noise_mode):
        if self.model_name == "stylegan":
            return self.generator.synthesize(
                ws,
                resolution=FINAL_RESOLUTION,
                alpha=1.0,
                noise_mode=noise_mode,
            )
        if self.model_name == "stylegan2":
            return self.generator.synthesize(ws, noise_mode=noise_mode)
        raise ValueError(f"{self.model_name} does not expose style synthesis.")


def load_generator(
    model_name,
    model_class,
    checkpoint_path,
    script_name,
    device,
):
    """Load one EMA generator produced by the current lesson scripts."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{model_name} checkpoint not found: {checkpoint_path}. "
            f"Run {script_name} first."
        )

    try:
        return load_model_weights(
            checkpoint_path,
            model_class,
            device=device,
            expected_metadata={"model_name": model_name},
        )
    except ValueError as error:
        raise ValueError(f"{error} Retrain it with {script_name}.") from error


def synthesize_styles_in_chunks(adapter, ws, noise_mode):
    """Synthesize a W stack without exceeding the training microbatch."""
    return generate_in_batches(
        ws,
        INFERENCE_BATCH_SIZE,
        lambda batch: adapter.synthesize(batch, noise_mode=noise_mode),
        module=adapter.generator,
    )


def spherical_interpolation(start, end, num_steps):
    """Interpolate directions on Z's hypersphere and radii linearly."""
    if start.shape != end.shape or start.ndim != 2 or start.shape[0] != 1:
        raise ValueError("start and end must both have shape [1, z_dim].")
    if num_steps < 2:
        raise ValueError("num_steps must be at least two.")

    fractions = torch.linspace(
        0,
        1,
        num_steps,
        device=start.device,
        dtype=start.dtype,
    )[:, None]
    start_direction = F.normalize(start, dim=1)
    end_direction = F.normalize(end, dim=1)
    cosine = (start_direction * end_direction).sum().clamp(-1, 1)

    if cosine.item() < -0.9995:
        raise ValueError("Slerp endpoints must not be nearly antipodal.")
    # Near-parallel endpoints make the trigonometric form ill-conditioned.
    if cosine.item() > 0.9995:
        directions = F.normalize(
            torch.lerp(start, end, fractions),
            dim=1,
        )
    else:
        angle = torch.acos(cosine)
        denominator = torch.sin(angle)
        directions = (
            torch.sin((1 - fractions) * angle)
            / denominator
            * start_direction
            + torch.sin(fractions * angle)
            / denominator
            * end_direction
        )

    radii = torch.lerp(
        start.norm(dim=1, keepdim=True),
        end.norm(dim=1, keepdim=True),
        fractions,
    )
    return directions * radii


@torch.inference_mode()
def generate_fixed_samples(adapters, fixed_z):
    """Generate deterministic final-resolution samples from one shared z."""
    return {
        model_name: generate_in_batches(
            fixed_z,
            INFERENCE_BATCH_SIZE,
            lambda batch, adapter=adapter: adapter.generate(batch),
            module=adapter.generator,
        )
        for model_name, adapter in adapters.items()
    }


@torch.inference_mode()
def generate_noise_variations(adapter, z):
    """Hold one W fixed while drawing independent per-layer noise."""
    w = adapter.mapping(z)
    ws = w[:, None, :].repeat(
        NUM_NOISE_VARIATIONS,
        adapter.num_ws,
        1,
    )
    return synthesize_styles_in_chunks(
        adapter,
        ws,
        noise_mode="random",
    )


@torch.inference_mode()
def generate_style_mixing(adapter, row_z, column_z):
    """Combine low-resolution row styles with high-resolution column styles."""
    row_w = adapter.mapping(row_z)
    column_w = adapter.mapping(column_z)
    row_ws = row_w[:, None, :].repeat(1, adapter.num_ws, 1)
    column_ws = column_w[:, None, :].repeat(1, adapter.num_ws, 1)
    row_images = synthesize_styles_in_chunks(
        adapter,
        row_ws,
        noise_mode="fixed",
    )
    column_images = synthesize_styles_in_chunks(
        adapter,
        column_ws,
        noise_mode="fixed",
    )

    cutoff = adapter.num_ws_for_resolution(STRUCTURE_MAX_RESOLUTION)
    mixed_ws = torch.stack(
        [
            torch.cat(
                [
                    row_w[row].repeat(cutoff, 1),
                    column_w[column].repeat(
                        adapter.num_ws - cutoff,
                        1,
                    ),
                ]
            )
            for row in range(NUM_STYLE_SOURCES)
            for column in range(NUM_STYLE_SOURCES)
        ]
    )
    mixed_images = synthesize_styles_in_chunks(
        adapter,
        mixed_ws,
        noise_mode="fixed",
    )
    mixed_images = mixed_images.reshape(
        NUM_STYLE_SOURCES,
        NUM_STYLE_SOURCES,
        *mixed_images.shape[1:],
    )
    return row_images, column_images, mixed_images


@torch.inference_mode()
def generate_truncation_samples(adapter, z):
    """Generate one fixed latent at each requested truncation psi."""
    samples = [
        adapter.generate(z, truncation_psi=psi)[0].cpu()
        for psi in TRUNCATION_PSIS
    ]
    return torch.stack(samples)


@torch.inference_mode()
def generate_interpolations(adapters, start_z, end_z):
    """Expose each style model's intermediate W path beside its Z path."""
    z_path = spherical_interpolation(
        start_z,
        end_z,
        NUM_INTERPOLATION_STEPS,
    )
    results = {
        "progan": [
            generate_in_batches(
                z_path,
                INFERENCE_BATCH_SIZE,
                adapters["progan"].generate,
                module=adapters["progan"].generator,
            )
        ]
    }
    for model_name in ("stylegan", "stylegan2"):
        adapter = adapters[model_name]
        z_images = generate_in_batches(
            z_path,
            INFERENCE_BATCH_SIZE,
            adapter.generate,
            module=adapter.generator,
        )

        start_w = adapter.mapping(start_z)
        end_w = adapter.mapping(end_z)
        fractions = torch.linspace(
            0,
            1,
            NUM_INTERPOLATION_STEPS,
            device=start_z.device,
            dtype=start_z.dtype,
        )[:, None]
        w_path = torch.lerp(start_w, end_w, fractions)
        ws = w_path[:, None, :].repeat(1, adapter.num_ws, 1)
        w_images = synthesize_styles_in_chunks(
            adapter,
            ws,
            noise_mode="fixed",
        )
        results[model_name] = [z_images, w_images]
    return results


def save_shared_comparison(
    destination_names,
    filename,
    image_rows,
    row_labels,
    *,
    title,
    column_labels,
):
    """Save one shared comparison under every participating model."""
    for model_name in destination_names:
        save_image_row_grid(
            image_rows,
            row_labels,
            EVALUATION_DIRS[model_name] / filename,
            title=title,
            column_labels=column_labels,
        )


def save_fixed_samples(fixed_samples):
    """Save the shared-z comparison for all three generator families."""
    names = ("progan", "stylegan", "stylegan2")
    save_shared_comparison(
        names,
        "fixed_samples.png",
        [fixed_samples[name] for name in names],
        ["ProGAN EMA", "StyleGAN EMA", "StyleGAN2 EMA"],
        title=(
            "Shared-z samples from independently trained generators "
            "(no semantic alignment implied)"
        ),
        column_labels=[
            f"Shared z {index + 1}" for index in range(NUM_FIXED_SAMPLES)
        ],
    )


def save_noise_variations(noise_variations):
    """Save stochastic-noise rows while W remains unchanged."""
    save_shared_comparison(
        ("stylegan", "stylegan2"),
        "noise_variations.png",
        [
            noise_variations["stylegan"],
            noise_variations["stylegan2"],
        ],
        ["StyleGAN EMA", "StyleGAN2 EMA"],
        title="One fixed W with independent stochastic layer noise",
        column_labels=[
            f"Noise {index + 1}" for index in range(NUM_NOISE_VARIATIONS)
        ],
    )


def save_style_mixing(model_name, display_name, mixing_data):
    """Save one source-labelled structure/detail style-mixing matrix."""
    row_images, column_images, mixed_images = mixing_data
    corner = torch.zeros_like(row_images[:1])
    image_rows = [torch.cat([corner, column_images], dim=0)]
    image_rows.extend(
        torch.cat([row_images[row : row + 1], mixed_images[row]], dim=0)
        for row in range(NUM_STYLE_SOURCES)
    )
    save_image_row_grid(
        image_rows,
        ["Detail sources"] + [
            f"Structure {index + 1}" for index in range(NUM_STYLE_SOURCES)
        ],
        EVALUATION_DIRS[model_name] / f"{model_name}_style_mixing.png",
        title=(
            f"{display_name}: 4-32 px structure styles from rows, "
            "64-128 px detail styles from columns"
        ),
        column_labels=["Structure source"] + [
            f"Detail {index + 1}" for index in range(NUM_STYLE_SOURCES)
        ],
    )


def save_truncation(truncation_samples):
    """Save learned-W-center interpolation for both style generators."""
    save_shared_comparison(
        ("stylegan", "stylegan2"),
        "truncation.png",
        [
            truncation_samples["stylegan"],
            truncation_samples["stylegan2"],
        ],
        ["StyleGAN EMA", "StyleGAN2 EMA"],
        title="One fixed z interpolated toward each model's learned w_avg",
        column_labels=[f"psi={psi:.1f}" for psi in TRUNCATION_PSIS],
    )


def save_interpolations(interpolations):
    """Save Z paths for every model and the added W paths where available."""
    column_labels = [
        f"t={index / (NUM_INTERPOLATION_STEPS - 1):.2f}"
        for index in range(NUM_INTERPOLATION_STEPS)
    ]
    for model_name in ("progan", "stylegan", "stylegan2"):
        row_labels = (
            ["Z (slerp)"]
            if model_name == "progan"
            else ["Z (slerp)", "W (linear)"]
        )
        save_image_row_grid(
            interpolations[model_name],
            row_labels,
            EVALUATION_DIRS[model_name] / "latent_interpolation.png",
            title=(
                f"{model_name.upper()} latent interpolation "
                "(fixed synthesis noise where applicable)"
            ),
            column_labels=column_labels,
        )


def main(weights_root=None):
    global MODEL_OUT_DIRS, EVALUATION_DIRS, MODEL_SPECS
    if weights_root is not None:
        MODEL_OUT_DIRS = {name: Path(weights_root) / name for name in MODEL_OUT_DIRS}
        EVALUATION_DIRS = {name: directory / "evaluation" for name, directory in MODEL_OUT_DIRS.items()}
        MODEL_SPECS = tuple(
            (name, model_class, MODEL_OUT_DIRS[name] / f"{name}_generator.pth", script_name)
            for name, model_class, _, script_name in MODEL_SPECS
        )
    set_seed(SEED)
    device = try_gpu()

    adapters = {}
    configurations = {}
    for (
        model_name,
        model_class,
        path,
        script_name,
    ) in MODEL_SPECS:
        generator, model_config = load_generator(
            model_name,
            model_class,
            path,
            script_name,
            device,
        )
        adapters[model_name] = GeneratorAdapter(model_name, generator)
        configurations[model_name] = model_config

    latent_dimensions = {
        int(config["z_dim"]) for config in configurations.values()
    }
    if len(latent_dimensions) != 1:
        raise ValueError(
            "The three checkpoints need the same z_dim for shared-z sampling."
        )
    z_dim = latent_dimensions.pop()

    torch.manual_seed(SEED)
    fixed_z = torch.randn(NUM_FIXED_SAMPLES, z_dim, device=device)
    fixed_samples = generate_fixed_samples(adapters, fixed_z)

    torch.manual_seed(SEED + 1)
    noise_z = torch.randn(1, z_dim, device=device)
    noise_variations = {
        model_name: generate_noise_variations(
            adapters[model_name],
            noise_z,
        )
        for model_name in ("stylegan", "stylegan2")
    }

    torch.manual_seed(SEED + 2)
    row_z = torch.randn(NUM_STYLE_SOURCES, z_dim, device=device)
    column_z = torch.randn(NUM_STYLE_SOURCES, z_dim, device=device)
    style_mixing = {
        model_name: generate_style_mixing(
            adapters[model_name],
            row_z,
            column_z,
        )
        for model_name in ("stylegan", "stylegan2")
    }

    torch.manual_seed(SEED + 3)
    truncation_z = torch.randn(1, z_dim, device=device)
    truncation_samples = {
        model_name: generate_truncation_samples(
            adapters[model_name],
            truncation_z,
        )
        for model_name in ("stylegan", "stylegan2")
    }

    torch.manual_seed(SEED + 4)
    interpolation_endpoints = torch.randn(2, z_dim, device=device)
    interpolations = generate_interpolations(
        adapters,
        interpolation_endpoints[:1],
        interpolation_endpoints[1:],
    )

    for evaluation_dir in EVALUATION_DIRS.values():
        reset_dir(str(evaluation_dir))

    save_fixed_samples(fixed_samples)
    save_noise_variations(noise_variations)
    save_style_mixing(
        "stylegan",
        "StyleGAN EMA (AdaIN)",
        style_mixing["stylegan"],
    )
    save_style_mixing(
        "stylegan2",
        "StyleGAN2 EMA (modulation/demodulation)",
        style_mixing["stylegan2"],
    )
    save_truncation(truncation_samples)
    save_interpolations(interpolations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path)
    main(parser.parse_args().weights_root)
