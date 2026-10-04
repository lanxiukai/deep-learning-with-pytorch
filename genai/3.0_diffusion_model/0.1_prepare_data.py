"""Download Food-101 once and prepare the shared, stratified training manifest."""

import argparse
from pathlib import Path

from dl_utils.diffusion.data import prepare_food101
from dl_utils.diffusion.lesson_utils import DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    print(prepare_food101(args.data_dir, download=args.download))
