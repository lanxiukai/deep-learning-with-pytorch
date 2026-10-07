"""Prepare the shared 256px Food-101 cache, optionally downloading its source."""

import argparse
from pathlib import Path

from dl_utils.diffusion.lesson_utils import DATA_DIR, PROJECT_ROOT
from dl_utils.diffusion.preparation import prepare_food101_cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--source-dir", type=Path, default=PROJECT_ROOT / "data/food101"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        prepare_food101_cache(
            args.source_dir, args.data_dir, download=args.download, workers=args.workers
        )
    )
