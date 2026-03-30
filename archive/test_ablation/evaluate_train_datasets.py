from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DEFAULT_PROJECT_CONFIG_PATH
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from test.evaluator import (
    CURVATURE_METHOD_NEURAL,
    DATA_SOURCE_H5_RAW,
    METHOD_LABELS,
    MetricAccumulator,
    NeuralPredictor,
)
from test.result_save import format_results_table

disable_bytecode_cache()


TRAIN_H5_METRIC_VIEW = "train_h5_rows"


@dataclass(frozen=True)
class DatasetModelPair:
    dataset_id: str
    dataset_dir: Path
    model_dir: Path


def _discover_training_dataset_dirs(data_root: Path) -> list[Path]:
    dataset_dirs: list[Path] = []
    if not data_root.exists():
        return dataset_dirs

    for path in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if any(path.glob("train_rho*.h5")):
            dataset_dirs.append(path)
    return dataset_dirs


def _choose_model_dir_for_dataset(dataset_id: str, model_root: Path) -> Path | None:
    if not model_root.exists():
        return None

    candidates = [path for path in model_root.iterdir() if path.is_dir()]
    if not candidates:
        return None

    dataset_aliases = [dataset_id]
    alt_id = re.sub(r"^data02_", "data_", dataset_id)
    if alt_id != dataset_id:
        dataset_aliases.append(alt_id)

    ranked: list[tuple[int, int, int, str, Path]] = []
    for path in candidates:
        name = path.name
        name_no_prefix = name.removeprefix("model_")

        match_rank = None
        for alias_index, alias in enumerate(dataset_aliases):
            if name_no_prefix == alias:
                match_rank = (alias_index, 0)
                break
            if name_no_prefix.startswith(f"{alias}_"):
                match_rank = (alias_index, 1)
                break
        if match_rank is None:
            continue

        alias_index, exactness = match_rank
        bs128_rank = 0 if "_bs128" in name_no_prefix else 1
        ranked.append((alias_index, exactness, bs128_rank, len(name_no_prefix), path))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[:-1])
    return ranked[0][-1]


def _build_dataset_model_pairs(project_root: Path) -> list[DatasetModelPair]:
    data_root = project_root / "data"
    model_root = project_root / "model"
    pairs: list[DatasetModelPair] = []

    for dataset_dir in _discover_training_dataset_dirs(data_root):
        model_dir = _choose_model_dir_for_dataset(dataset_dir.name, model_root)
        if model_dir is None:
            continue
        pairs.append(
            DatasetModelPair(
                dataset_id=dataset_dir.name,
                dataset_dir=dataset_dir,
                model_dir=model_dir,
            )
        )
    return pairs


def _evaluate_pair(
    project_root: Path,
    pair: DatasetModelPair,
    *,
    chunk_size: int,
    config_path: str | Path,
) -> list[dict[str, object]]:
    predictor = NeuralPredictor(project_root, model_dir=pair.model_dir, config_path=config_path)
    rows: list[dict[str, object]] = []

    rho_values = []
    for h5_path in sorted(pair.dataset_dir.glob("train_rho*.h5")):
        match = re.fullmatch(r"train_rho(\d+)\.h5", h5_path.name)
        if match:
            rho_values.append(int(match.group(1)))

    for rho in rho_values:
        if not predictor.has_bundle(rho):
            predictor.note_missing_bundle_once(rho)
            continue

        h5_path = pair.dataset_dir / f"train_rho{rho}.h5"
        grouped: dict[int, MetricAccumulator] = {}
        with h5py.File(h5_path, "r") as handle:
            total_samples = int(handle["X"].shape[0])
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                x = handle["X"][start:end]
                y_true = handle["Y"][start:end, 0]
                steps = handle["reinit_steps"][start:end, 0].astype(np.int32)
                pred = predictor.predict(rho, x)

                for step in np.unique(steps):
                    mask = steps == step
                    accumulator = grouped.setdefault(int(step), MetricAccumulator())
                    accumulator.update(y_true[mask], pred[mask])

        for step in sorted(grouped):
            metrics = grouped[step].as_metrics()
            rows.append(
                {
                    "data_split": "train",
                    "group_id": pair.dataset_id,
                    "rho_model": rho,
                    "step_or_iter": step,
                    "method": METHOD_LABELS["neural"],
                    "model_tag": predictor.model_tag,
                    "curvature_method": CURVATURE_METHOD_NEURAL,
                    "metric_view": TRAIN_H5_METRIC_VIEW,
                    "data_source": DATA_SOURCE_H5_RAW,
                    **metrics,
                }
            )
    return rows


def _build_report_text(pairs: list[DatasetModelPair], rows: list[dict[str, object]]) -> str:
    mapping_lines = ["Dataset-to-model mapping:"]
    for pair in pairs:
        mapping_lines.append(f"- {pair.dataset_id} -> {pair.model_dir.name}")
    if not pairs:
        mapping_lines.append("- <none>")

    body = format_results_table(rows) if rows else "<no evaluation rows>"
    return "\n".join(mapping_lines) + "\n\n" + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate neural model error on dataset-scoped training HDF5s by rho and reinit step."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of HDF5 rows to process per batch chunk.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path. Defaults to test/results/train_dataset_eval.txt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help=f"Shared project config path (default: {DEFAULT_PROJECT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    output_path = (
        Path(args.output)
        if args.output is not None
        else project_root / "test" / "results" / "train_dataset_eval.txt"
    )

    pairs = _build_dataset_model_pairs(project_root)
    rows: list[dict[str, object]] = []
    for pair in pairs:
        print(f"Evaluating dataset={pair.dataset_id} with model={pair.model_dir.name}")
        rows.extend(_evaluate_pair(project_root, pair, chunk_size=args.chunk_size, config_path=args.config))

    rows.sort(
        key=lambda row: (
            str(row["group_id"]),
            int(row["rho_model"]),
            int(row["step_or_iter"]),
            str(row["model_tag"]),
        )
    )
    report = _build_report_text(pairs, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved training-dataset evaluation report to {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
