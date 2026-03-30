from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate.dataset_test import build_grid, find_projection_theta, hkappa_analytic
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .paper_alignment import (
    MetricTriple,
    OriginPredictorLite,
    compute_metrics,
    format_metric_delta,
    format_metric_triple,
    load_payload,
    load_meta,
    normalized_distance,
)
from .paper_reference import PAPER_UNIFORM_REFERENCES, PaperMetricTriple, get_paper_reference

disable_bytecode_cache()


TARGET_VARIANT_PROJECTION = "projection"
TARGET_VARIANT_RADIAL = "radial"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare target definitions against paper metrics on rebuilt explicit tests.")
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
        help="Optional explicit report path. Defaults to test/results/target_sensitivity_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_target_sensitivity_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nTarget sensitivity report saved to {output_path}")


def build_target_sensitivity_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    origin_predictor = OriginPredictorLite(project_root)
    rebuilt_root = project_config.project.data_dir

    variant_case_wins = {TARGET_VARIANT_PROJECTION: 0, TARGET_VARIANT_RADIAL: 0}
    variant_step_wins = {TARGET_VARIANT_PROJECTION: 0, TARGET_VARIANT_RADIAL: 0}
    total_cases = 0
    total_steps = 0

    lines: list[str] = []
    lines.append("Target Sensitivity Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: {project_config.source_path}")
    lines.append(f"Data root: {rebuilt_root}")
    lines.append("")

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        if exp_id not in PAPER_UNIFORM_REFERENCES:
            continue

        total_cases += 1
        meta = load_meta(rebuilt_root / exp_id / "meta.json")
        case_scores = {TARGET_VARIANT_PROJECTION: 0.0, TARGET_VARIANT_RADIAL: 0.0}

        lines.append(f"[{exp_id}] {PAPER_UNIFORM_REFERENCES[exp_id].table_id}")
        lines.append(
            f"meta: h={float(meta['h']):.9e} | rho_eq={float(meta['rho_eq']):.4f} | "
            f"sampling={meta.get('sampling_rule', 'n/a')}"
        )

        for step in project_config.evaluation.test_iters:
            total_steps += 1
            payload = load_payload(rebuilt_root / exp_id / f"iter_{int(step)}.h5")
            xy = load_xy(rebuilt_root / exp_id / f"iter_{int(step)}.h5", payload["indices"], meta)
            targets = build_target_variants(
                xy=xy,
                stored_target=payload["target"],
                h=float(meta["h"]),
                a=float(meta["a"]),
                b=float(meta["b"]),
                p=float(meta["p"]),
            )
            fd_pred = payload["numerical"]
            origin_pred = origin_predictor.predict(scenario.rho_model, payload["stencils"])
            paper_ref = get_paper_reference(exp_id, int(step))

            step_scores: dict[str, float] = {}
            lines.append(f"  step={int(step)}")
            lines.append(
                "    target_delta: "
                f"projection_vs_stored={target_delta_summary(targets[TARGET_VARIANT_PROJECTION], payload['target'])} | "
                f"radial_vs_stored={target_delta_summary(targets[TARGET_VARIANT_RADIAL], payload['target'])}"
            )
            for variant_name, target in targets.items():
                numerical_metrics = compute_metrics(target, fd_pred)
                origin_metrics = compute_metrics(target, origin_pred)
                origin_score = normalized_distance(origin_metrics, paper_ref.paper_model)
                numerical_score = normalized_distance(numerical_metrics, paper_ref.numerical)
                combined_score = origin_score + numerical_score
                case_scores[variant_name] += combined_score
                step_scores[variant_name] = combined_score

                lines.append(
                    f"    {variant_name}: "
                    f"origin={format_metric_triple(origin_metrics)} | "
                    f"numerical={format_metric_triple(numerical_metrics)} | "
                    f"origin |d_paper|={format_metric_delta(origin_metrics, paper_ref.paper_model)} | "
                    f"numerical |d_paper|={format_metric_delta(numerical_metrics, paper_ref.numerical)} | "
                    f"score={combined_score:.6f}"
                )

            step_winner = min(step_scores, key=step_scores.get)
            variant_step_wins[step_winner] += 1
            lines.append(
                "    step_summary: "
                f"winner={step_winner} | "
                f"projection_score={step_scores[TARGET_VARIANT_PROJECTION]:.6f} | "
                f"radial_score={step_scores[TARGET_VARIANT_RADIAL]:.6f}"
            )

        case_winner = min(case_scores, key=case_scores.get)
        variant_case_wins[case_winner] += 1
        lines.append(
            "  case_summary: "
            f"winner={case_winner} | "
            f"projection_score={case_scores[TARGET_VARIANT_PROJECTION]:.6f} | "
            f"radial_score={case_scores[TARGET_VARIANT_RADIAL]:.6f}"
        )
        lines.append("")

    lines.insert(
        4,
        "Summary: "
        f"projection wins {variant_case_wins[TARGET_VARIANT_PROJECTION]}/{max(total_cases, 1)} cases "
        f"and {variant_step_wins[TARGET_VARIANT_PROJECTION]}/{max(total_steps, 1)} case-steps; "
        f"radial wins {variant_case_wins[TARGET_VARIANT_RADIAL]}/{max(total_cases, 1)} cases "
        f"and {variant_step_wins[TARGET_VARIANT_RADIAL]}/{max(total_steps, 1)} case-steps.",
    )
    return "\n".join(lines).rstrip()


def build_target_variants(
    *,
    xy: np.ndarray,
    stored_target: np.ndarray,
    h: float,
    a: float,
    b: float,
    p: float,
) -> dict[str, np.ndarray]:
    projection_theta = find_projection_theta(xy, a, b, p)
    radial_theta = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * np.pi)
    return {
        TARGET_VARIANT_PROJECTION: hkappa_analytic(projection_theta, h, a, b, p),
        TARGET_VARIANT_RADIAL: hkappa_analytic(radial_theta, h, a, b, p),
    }


def load_xy(h5_path: Path, indices: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    payload = load_payload_with_xy(h5_path)
    if payload is not None:
        return payload
    x_grid, y_grid, _ = build_grid(float(meta["L"]), int(meta["N"]))
    return np.column_stack((x_grid[indices[:, 0], indices[:, 1]], y_grid[indices[:, 0], indices[:, 1]])).astype(
        np.float64,
        copy=False,
    )


def load_payload_with_xy(h5_path: Path) -> np.ndarray | None:
    import h5py

    with h5py.File(h5_path, "r") as handle:
        if "xy" not in handle:
            return None
        return handle["xy"][:].astype(np.float64)


def target_delta_summary(candidate: np.ndarray, stored: np.ndarray) -> str:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(stored, dtype=np.float64)
    abs_diff = np.abs(diff)
    return f"mae={abs_diff.mean():.6e}, max={abs_diff.max():.6e}"


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"target_sensitivity_{stamp}.txt"


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
