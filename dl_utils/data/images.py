"""Shared image decoding and color-mode conversion helpers."""

from pathlib import Path

from PIL import Image


def flatten_to_rgb(
    image: Image.Image,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Return an RGB copy, compositing transparency onto ``background``."""
    has_alpha = "transparency" in image.info or any(
        band.upper() == "A" for band in image.getbands()
    )
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def load_rgb_image(
    path: str | Path,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Load an image as RGB, compositing any transparency onto a solid color."""
    with Image.open(path) as image:
        return flatten_to_rgb(image, background)
