"""Build the shared, lossless 256px Food-101 cache before training."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from dl_utils.diffusion.data import CACHE_SPEC, prepare_food101


def cache_image(source, destination):
    """Resize and center-crop to 256px RGB, then save lossless PNG pixels."""
    if destination.exists():
        with Image.open(destination) as image:
            if image.size != (256, 256) or image.mode != "RGB":
                raise ValueError(f"Incompatible cached image: {destination}")
            image.load()
        return False
    geometry = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.CenterCrop(256),
        ]
    )
    with Image.open(source) as image:
        prepared = geometry(image.convert("RGB"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        prepared.save(temporary, format="PNG", compress_level=1)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def prepare_food101_cache(source_root, destination_root, *, download=False, workers=8):
    """Download if requested, resume image preparation, then publish the manifest."""
    source_root, destination_root = Path(source_root), Path(destination_root)
    if workers < 1:
        raise ValueError("workers must be positive")
    if (
        destination_root.resolve() == source_root.resolve()
        or destination_root.resolve().is_relative_to(source_root.resolve())
        or source_root.resolve().is_relative_to(destination_root.resolve())
    ):
        raise ValueError("Use separate, non-nested source and prepared directories.")
    source_manifest = prepare_food101(source_root, download=download)
    manifest = json.loads(source_manifest.read_text())
    manifest["cache"] = CACHE_SPEC
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "diffusion_manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError(
            "Existing prepared manifest differs; use a separate directory."
        )
    records = [record for split in manifest["splits"].values() for record in split]

    def prepare_record(record):
        sample, _ = record
        return cache_image(
            source_root / "food-101/images" / sample,
            destination_root / "images" / Path(sample).with_suffix(".png"),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        created = sum(
            tqdm(
                pool.map(prepare_record, records, buffersize=workers * 2),
                total=len(records),
                desc="Preparing Food-101 256px",
                unit="image",
            )
        )
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    print(
        f"Food-101 cache ready: {destination_root} "
        f"({created} created, {len(records) - created} verified; RGB 256x256)."
    )
    return destination
