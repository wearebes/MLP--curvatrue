from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .distribution_shift import summarize_shift
from .formula_sensitivity import (
    build_numerical_variants,
    build_origin_variants,
)
from .paper_alignment import (
    load_origin_stats,
    load_payload,
    normalized_distance,
)
from .paper_reference import PAPER_UNIFORM_REFERENCES, get_paper_reference
from .scale_sensitivity import read_model_h
from generate.dataset_test import LevelSetReinitializer, build_flower_phi0, build_grid

disable_bytecode_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused investigation for the 276 cases.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help=f"Shared project config path (default: {DEFAULT_PROJECT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit report path. Defaults to test/results/rho276_focus_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_rho276_focus_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n276 focus report saved to {output_path}")


def build_rho276_focus_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    data_root = project_config.project.data_dir
    origin_model_root = project_root / "model" / "origin"

    lines: list[str] = []
    lines.append("276 Focus Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: {project_config.source_path}")
    lines.append("")

    target_ids = {"smooth_276", "acute_276"}
    scenarios = [scenario for scenario in project_config.evaluation.scenarios if scenario.exp_id in target_ids]

    for scenario in scenarios:
        exp_id = scenario.exp_id
        model_h = read_model_h(origin_model_root / f"trainStats_{int(scenario.rho_model)}.csv")
        case_h = float(scenario.h)
        to_model_factor = model_h / case_h
        train_mu, train_sigma = load_origin_stats(origin_model_root / f"trainStats_{int(scenario.rho_model)}.csv")

        lines.append(f"[{exp_id}] {PAPER_UNIFORM_REFERENCES[exp_id].table_id}")
        lines.append(
            f"meta: case_h={case_h:.9e} | model_h={model_h:.9e} | "
            f"rho_eq_case={1.0 / case_h + 1.0:.6f} | rho_eq_model={1.0 / model_h + 1.0:.6f} | "
            f"to_model_factor={to_model_factor:.9e}"
        )

        x_grid, y_grid, h = build_grid(float(scenario.L), int(scenario.N))
        phi0 = build_flower_phi0(x_grid, y_grid, float(scenario.a), float(scenario.b), float(scenario.p))
        reinitializer = LevelSetReinitializer(cfl=project_config.generation.cfl)

        for step in project_config.evaluation.test_iters:
            payload = load_payload(data_root / exp_id / f"iter_{int(step)}.h5")
            paper_ref = get_paper_reference(exp_id, int(step))
            phi = reinitializer.reinitialize(phi0, h, int(step))

            raw_shift = summarize_shift(payload["stencils"], train_mu, train_sigma)
            scaled_shift = summarize_shift(payload["stencils"] * to_model_factor, train_mu, train_sigma)

            num_variants = build_numerical_variants(phi, payload["indices"], h, payload["numerical"])
            origin_variants = build_origin_variants(phi, payload["indices"], read_origin_predictor(project_root), int(scenario.rho_model))

            num_scores = {
                name: normalized_distance(
                    compute_metrics_local(payload["target"], pred),
                    paper_ref.numerical,
                )
                for name, pred in num_variants.items()
            }
            origin_scores = {
                name: normalized_distance(
                    compute_metrics_local(payload["target"], pred),
                    paper_ref.paper_model,
                )
                for name, pred in origin_variants.items()
            }

            lines.append(f"  step={int(step)}")
            lines.append(
                "    shift: "
                f"raw_z={raw_shift['mean_abs_shift_z']:.6e} | "
                f"scaled_z={scaled_shift['mean_abs_shift_z']:.6e} | "
                f"raw_std_dev={raw_shift['std_ratio_maxdev']:.6e} | "
                f"scaled_std_dev={scaled_shift['std_ratio_maxdev']:.6e}"
            )
            lines.append(
                "    numerical_best: "
                f"{min(num_scores, key=num_scores.get)} | "
                + " | ".join(f"{name}={score:.6f}" for name, score in num_scores.items())
            )
            lines.append(
                "    origin_best: "
                f"{min(origin_scores, key=origin_scores.get)} | "
                + " | ".join(f"{name}={score:.6f}" for name, score in origin_scores.items())
            )
        lines.append("")

    return "\n".join(lines).rstrip()


_ORIGIN_PREDICTOR_CACHE = None


def read_origin_predictor(project_root: Path):
    global _ORIGIN_PREDICTOR_CACHE
    if _ORIGIN_PREDICTOR_CACHE is None:
        from .paper_alignment import OriginPredictorLite

        _ORIGIN_PREDICTOR_CACHE = OriginPredictorLite(project_root)
    return _ORIGIN_PREDICTOR_CACHE


def compute_metrics_local(target, pred):
    import numpy as np

    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    abs_error = np.abs(error)

    class _Metric:
        mae = float(abs_error.mean()) if abs_error.size else 0.0
        max_ae = float(abs_error.max()) if abs_error.size else 0.0
        mse = float((error**2).mean()) if error.size else 0.0

    return _Metric()


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"rho276_focus_{stamp}.txt"


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
