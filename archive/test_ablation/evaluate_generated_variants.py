from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from project_config import DEFAULT_PROJECT_CONFIG_PATH

from .paper_alignment import OriginPredictorLite

CURVATURE_METHOD_STANDARD = "expanded_formula"
CURVATURE_METHOD_NEURAL = "model_prediction"
DATA_SOURCE_REBUILT = "rebuilt"
METHOD_LABELS = {
    "numerical": "Numerical",
    "origin": "Origin",
}
MODEL_TAG_BASELINE = "baseline_numerical"
MODEL_TAG_ORIGIN = "origin"
RESULT_COLUMNS = [
    "data_split",
    "group_id",
    "rho_model",
    "step_or_iter",
    "method",
    "model_tag",
    "curvature_method",
    "metric_view",
    "data_source",
    "N_samples",
    "MAE_hk",
    "MaxAE_hk",
    "MSE_hk",
]
GROUP_COLUMNS = [
    "data_split",
    "method",
    "model_tag",
    "curvature_method",
    "metric_view",
    "data_source",
]
TABLE_COLUMNS = [
    "group_id",
    "rho_model",
    "step_or_iter",
    "N_samples",
    "MAE_hk",
    "MaxAE_hk",
    "MSE_hk",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple generated explicit-test directories in one pass using origin and numerical methods."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="One or more generated explicit-test roots, e.g. data/so3_to2 data/so4_to2 ...",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output txt path. Defaults to test/results/generated_variants_eval.txt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help=f"Shared project config path (default: {DEFAULT_PROJECT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    rows = evaluate_datasets(project_root, args.datasets)
    table = format_results_table(rows)

    output_path = Path(args.output) if args.output else project_root / "test" / "results" / "generated_variants_eval.txt"
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table + "\n", encoding="utf-8")

    print(table)
    print(f"\nSaved to {output_path}")


def evaluate_datasets(project_root: Path, dataset_roots: list[str]) -> list[dict[str, object]]:
    origin_predictor = OriginPredictorLite(project_root)
    rows: list[dict[str, object]] = []

    for dataset_root_raw in dataset_roots:
        dataset_root = Path(dataset_root_raw)
        if not dataset_root.is_absolute():
            dataset_root = project_root / dataset_root
        dataset_root = dataset_root.resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

        try:
            dataset_tag = dataset_root.relative_to(project_root / "data").as_posix()
        except ValueError:
            dataset_tag = dataset_root.name
        experiment_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and (path / "meta.json").exists())

        for exp_dir in experiment_dirs:
            meta = json.loads((exp_dir / "meta.json").read_text(encoding="utf-8"))
            rho_model = int(meta["rho_model"])
            experiment_type = str(meta.get("experiment_type", "test"))
            blueprint_id = str(meta.get("blueprint_id", exp_dir.name))
            group_id = f"{dataset_tag}/{experiment_type}/{blueprint_id}"
            metric_view = str(meta.get("sampling_rule", "stored_h5"))
            data_source = f"{DATA_SOURCE_REBUILT}:{dataset_tag}"

            bundle_h5 = project_root / "model" / "origin" / f"nnet_{rho_model}.h5"
            bundle_csv = project_root / "model" / "origin" / f"trainStats_{rho_model}.csv"
            if not (bundle_h5.exists() and bundle_csv.exists()):
                continue

            for h5_path in sorted(exp_dir.glob("iter_*.h5")):
                step = int(h5_path.stem.split("_")[-1])
                payload = load_payload(h5_path)

                rows.append(
                    build_metric_row(
                        data_split="test",
                        group_id=group_id,
                        rho_model=rho_model,
                        step=step,
                        method=METHOD_LABELS["numerical"],
                        model_tag=MODEL_TAG_BASELINE,
                        curvature_method=CURVATURE_METHOD_STANDARD,
                        metric_view=metric_view,
                        data_source=data_source,
                        target=payload["target"],
                        pred=payload["numerical"],
                    )
                )

                origin_pred = origin_predictor.predict(rho_model, payload["stencils"])
                rows.append(
                    build_metric_row(
                        data_split="test",
                        group_id=group_id,
                        rho_model=rho_model,
                        step=step,
                        method=METHOD_LABELS["origin"],
                        model_tag=MODEL_TAG_ORIGIN,
                        curvature_method=CURVATURE_METHOD_NEURAL,
                        metric_view=metric_view,
                        data_source=data_source,
                        target=payload["target"],
                        pred=origin_pred,
                    )
                )

    rows.sort(
        key=lambda row: (
            str(row["data_source"]),
            str(row["method"]),
            int(row["rho_model"]),
            str(row["group_id"]),
            int(row["step_or_iter"]),
        )
    )
    return rows


def load_payload(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        return {
            "stencils": handle["stencils_raw"][:].astype(np.float64),
            "target": handle["hkappa_target"][:].astype(np.float64),
            "numerical": handle["hkappa_fd"][:].astype(np.float64),
        }


def build_metric_row(
    *,
    data_split: str,
    group_id: str,
    rho_model: int,
    step: int,
    method: str,
    model_tag: str,
    curvature_method: str,
    metric_view: str,
    data_source: str,
    target: np.ndarray,
    pred: np.ndarray,
) -> dict[str, object]:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    error = pred - target
    abs_error = np.abs(error)
    return {
        "data_split": data_split,
        "group_id": group_id,
        "rho_model": rho_model,
        "step_or_iter": step,
        "method": method,
        "model_tag": model_tag,
        "curvature_method": curvature_method,
        "metric_view": metric_view,
        "data_source": data_source,
        "N_samples": int(target.shape[0]),
        "MAE_hk": float(abs_error.mean()) if abs_error.size else 0.0,
        "MaxAE_hk": float(abs_error.max()) if abs_error.size else 0.0,
        "MSE_hk": float(np.mean(error**2)) if error.size else 0.0,
    }


def format_results_table(rows: list[dict[str, object]]) -> str:
    normalized = [{column: row.get(column, "") for column in RESULT_COLUMNS} for row in rows]
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = {}
    for row in normalized:
        key = tuple(str(row[column]) for column in GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    sections: list[str] = []
    for key in sorted(groups):
        data_split, method, model_tag, curvature_method, metric_view, data_source = key
        title = (
            f"[{data_split}] {method} | model={model_tag} | curvature={curvature_method} | "
            f"nodes={metric_view} | source={data_source}"
        )
        sections.append(title)
        sections.append(_format_single_table(groups[key]))
    return "\n\n".join(sections)


def _format_single_table(rows: list[dict[str, object]]) -> str:
    if pd is not None:
        frame = pd.DataFrame(rows, columns=TABLE_COLUMNS)
        formatters = {
            "MAE_hk": lambda value: f"{float(value):.6e}",
            "MaxAE_hk": lambda value: f"{float(value):.6e}",
            "MSE_hk": lambda value: f"{float(value):.6e}",
        }
        return frame.to_string(index=False, columns=TABLE_COLUMNS, formatters=formatters)

    prepared_rows = []
    for row in rows:
        prepared_rows.append(
            {
                "group_id": str(row["group_id"]),
                "rho_model": str(row["rho_model"]),
                "step_or_iter": str(row["step_or_iter"]),
                "N_samples": str(row["N_samples"]),
                "MAE_hk": f"{float(row['MAE_hk']):.6e}",
                "MaxAE_hk": f"{float(row['MaxAE_hk']):.6e}",
                "MSE_hk": f"{float(row['MSE_hk']):.6e}",
            }
        )

    widths = {
        column: max(len(column), *(len(prepared[column]) for prepared in prepared_rows))
        for column in TABLE_COLUMNS
    }
    header = " ".join(column.rjust(widths[column]) for column in TABLE_COLUMNS)
    lines = [header]
    for prepared in prepared_rows:
        lines.append(" ".join(prepared[column].rjust(widths[column]) for column in TABLE_COLUMNS))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
