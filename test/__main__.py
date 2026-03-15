from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .evaluate import evaluate_test_data, evaluate_train_data
from .result_save import format_results_table, save_unified_results

disable_bytecode_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate numerical and neural hk metrics on train/test data.")
    parser.add_argument(
        "methods",
        nargs="+",
        choices=["numerical", "neural"],
        help="One or both methods to evaluate.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    methods = list(dict.fromkeys(args.methods))

    train_rows = evaluate_train_data(project_root, methods)
    test_rows = evaluate_test_data(project_root, methods)
    all_rows = train_rows + test_rows

    print("Unified hk evaluation summary:")
    print(format_results_table(all_rows))

    output_path = save_unified_results(all_rows, project_root / "test" / "results")
    print(f"\nUnified hk evaluation saved to {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
