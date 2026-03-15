from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .config import OUTPUT_DIR, PROJECT_ROOT, TARGET_RHOS
from .generate_data import generate_train_datasets
from .test_data import generate_test_datasets

disable_bytecode_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train or test data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Generate training datasets.")
    train_parser.add_argument(
        "--rho",
        type=int,
        nargs="+",
        default=TARGET_RHOS,
        help=f"Resolution(s) to generate (default: {TARGET_RHOS})",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for training HDF5 files (default: {OUTPUT_DIR})",
    )

    test_parser = subparsers.add_parser("test", help="Generate explicit test experiments.")
    test_parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for test experiments (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if args.command == "train":
        generate_train_datasets(target_rhos=args.rho, output_dir=output_dir)
        return

    if args.command == "test":
        generate_test_datasets(output_dir=output_dir)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
