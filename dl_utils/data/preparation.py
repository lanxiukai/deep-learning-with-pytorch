"""Resumable image-folder caches independent of a particular dataset."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from dl_utils.data.images import flatten_to_rgb

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def resize_image(source: Path, target: Path, size: int) -> bool:
    """Resize one image atomically, flattening transparency onto white."""
    if target.exists():
        with Image.open(target) as cached:
            if cached.size != (size, size):
                raise RuntimeError(
                    f"Existing cache image has size {cached.size}, expected "
                    f"{(size, size)}: {target}"
                )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with Image.open(source) as image:
            image_format = image.format or source.suffix.removeprefix(".").upper()
            resized = flatten_to_rgb(image).resize(
                (size, size),
                Image.Resampling.BILINEAR,
            )
            save_options = {"compress_level": 1} if image_format == "PNG" else {}
            resized.save(temporary, format=image_format, **save_options)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def build_image_folder_cache(
    source_root: Path,
    target_root: Path,
    size: int,
    workers: int,
) -> None:
    """Build or resume a resized copy of an ImageFolder directory."""
    source_root = Path(source_root)
    target_root = Path(target_root)
    source_resolved = source_root.resolve()
    target_resolved = target_root.resolve()
    if source_resolved == target_resolved or target_resolved.is_relative_to(
        source_resolved
    ):
        raise ValueError("Target directory must be outside the source directory")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")

    sources = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not sources:
        raise RuntimeError(f"No supported images found under {source_root}")

    jobs = [
        (source, target_root / source.relative_to(source_root), size)
        for source in sources
    ]
    created = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(resize_image, *job) for job in jobs]
        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            created += int(future.result())
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"\rProcessed {completed}/{len(futures)} images",
                    end="",
                    flush=True,
                )
    print()

    missing = [
        source
        for source in sources
        if not (target_root / source.relative_to(source_root)).is_file()
    ]
    if missing:
        raise RuntimeError(f"Cache is incomplete: {len(missing)} images missing")
    print(
        f"Cache ready: {target_root} ({created} created, "
        f"{len(sources) - created} already present)"
    )


__all__ = [
    "build_image_folder_cache",
    "resize_image",
]
