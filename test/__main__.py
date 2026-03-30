from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .evaluator import evaluate_test_data, evaluate_train_data
from .utils import format_results_table, save_unified_results


def _resolve_source_path(source: str | None) -> tuple[Path | None, str | None]:
    if not source:
        return None, None

    candidate = Path(source)
    if not candidate.is_absolute():
        data_root = PROJECT_ROOT / "data"
        if candidate.parts and candidate.parts[0] == "data":
            candidate = PROJECT_ROOT / candidate
        else:
            candidate = data_root / candidate
    return candidate, source


def _normalize_methods(raw_methods: list[str], parser: argparse.ArgumentParser) -> list[str]:
    alias_map = {
        "numerical": "numerical",
        "neural": "neural",
        "nerual": "neural",
        "origin": "origin",
        "paper": "origin",
    }
    normalized: list[str] = []
    invalid: list[str] = []
    for method in raw_methods:
        key = method.strip().lower()
        mapped = alias_map.get(key)
        if mapped is None:
            invalid.append(method)
            continue
        normalized.append(mapped)
    if invalid:
        parser.error(
            f"invalid method(s): {', '.join(invalid)}. "
            "Use one or more of: numerical, neural, origin "
            "(aliases: nerual -> neural, paper -> origin)."
        )
    return list(dict.fromkeys(normalized))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate numerical and neural hk metrics on train/test data.")
    parser.add_argument(
        "methods", nargs="+",
        help="One or more methods to evaluate (numerical, neural, origin/paper).",
    )
    parser.add_argument(
        "-data-split", "--data-split",
        choices=["train", "test"], default=None,
        help="Optionally evaluate only one split. Omit to run both.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Custom test-data source subdirectory",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit Run ID to evaluate.")
    args = parser.parse_args()

    methods = _normalize_methods(args.methods, parser)
    source_path, source_label = _resolve_source_path(args.source)
    all_rows = []

    if args.data_split in (None, "train"):
        all_rows.extend(evaluate_train_data(PROJECT_ROOT, methods, data_dir=source_path, run_id=args.run_id))
    if args.data_split in (None, "test"):
        all_rows.extend(evaluate_test_data(
            PROJECT_ROOT, methods,
            data_dir=source_path, data_source_opt=source_label, run_id=args.run_id,
        ))

    print("Unified hk evaluation summary:")
    print(format_results_table(all_rows))

    if args.run_id:
        output_dir = PROJECT_ROOT / "model" / args.run_id / "evals"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = PROJECT_ROOT / "test" / "results"

    output_path = save_unified_results(all_rows, output_dir)
    print(f"\nUnified hk evaluation saved to {output_path}")


if __name__ == "__main__":
    main()
