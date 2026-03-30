from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from generate.pde_utils import ReinitQualityEvaluator
from generate.stencil_encoding import extract_3x3_stencils
from generate.dataset_test import (
    LevelSetReinitializer,
    build_flower_phi0,
    build_grid,
    find_projection_theta,
    hkappa_analytic,
)
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config

from .explicit_sdf_upper_bound import build_exact_signed_distance_field
from .paper_alignment import OriginPredictorLite, compute_metrics, normalized_distance
from .paper_reference import get_paper_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "current": np.array([], dtype=np.int64),
    "center_exact": np.array([4], dtype=np.int64),
    "cross_exact": np.array([1, 3, 5, 7], dtype=np.int64),
    "diag_exact": np.array([0, 2, 6, 8], dtype=np.int64),
    "center_cross_exact": np.array([1, 3, 4, 5, 7], dtype=np.int64),
    "full_exact": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
}


@dataclass(frozen=True)
class VariantMetrics:
    mae: float
    max_ae: float
    mse: float
    score: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation table: replace selected stencil groups by exact-SDF values and measure the paper model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help=f"Shared project config path (default: {DEFAULT_PROJECT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="",
        help="Optional output stem relative to project root. Defaults to test/results/stencil_repair_ablation_MMDD.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    rows, summary = build_ablation_results(project_config.project_root, config_path=args.config)
    csv_path, md_path = resolve_output_paths(project_config.project_root, args.output_stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, summary)
    print(f"[Saved CSV] {csv_path}")
    print(f"[Saved MD ] {md_path}")


def build_ablation_results(
    project_root: Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    predictor = OriginPredictorLite(project_root)
    rows: list[dict[str, object]] = []
    variant_win_counts = {name: 0 for name in VARIANTS}
    variant_score_sums = {name: 0.0 for name in VARIANTS}

    for scenario in project_config.evaluation.scenarios:
        X, Y, h = build_grid(float(scenario.L), int(scenario.N))
        phi0 = build_flower_phi0(X, Y, float(scenario.a), float(scenario.b), float(scenario.p))
        reinitializer = LevelSetReinitializer(cfl=float(project_config.generation.cfl))

        for step in project_config.evaluation.test_iters:
            phi = reinitializer.reinitialize(phi0, h, int(step))
            indices = current_indices(phi)
            xy = np.column_stack((X[indices[:, 0], indices[:, 1]], Y[indices[:, 0], indices[:, 1]]))
            theta = find_projection_theta(xy, float(scenario.a), float(scenario.b), float(scenario.p))
            target = hkappa_analytic(theta, h, float(scenario.a), float(scenario.b), float(scenario.p))
            exact_phi = build_exact_signed_distance_field(
                phi_shape=phi.shape,
                phi0=phi0,
                X=X,
                Y=Y,
                indices=indices,
                a=float(scenario.a),
                b=float(scenario.b),
                p=int(scenario.p),
            )

            current_stencils = extract_3x3_stencils(phi, indices)
            exact_stencils = extract_3x3_stencils(exact_phi, indices)
            paper_ref = get_paper_reference(scenario.exp_id, int(step))

            per_variant: dict[str, VariantMetrics] = {}
            for variant_name, replace_positions in VARIANTS.items():
                stencils = current_stencils.copy()
                if replace_positions.size:
                    stencils[:, replace_positions] = exact_stencils[:, replace_positions]
                pred = predictor.predict(int(scenario.rho_model), stencils)
                metrics = compute_metrics(target, pred)
                score = normalized_distance(metrics, paper_ref.paper_model)
                per_variant[variant_name] = VariantMetrics(
                    mae=float(metrics.mae),
                    max_ae=float(metrics.max_ae),
                    mse=float(metrics.mse),
                    score=float(score),
                )
                variant_score_sums[variant_name] += float(score)

            winner = min(per_variant, key=lambda name: per_variant[name].score)
            variant_win_counts[winner] += 1

            row: dict[str, object] = {
                "exp_id": scenario.exp_id,
                "step": int(step),
                "samples": int(indices.shape[0]),
                "winner": winner,
            }
            for variant_name, metrics in per_variant.items():
                row[f"{variant_name}_mae"] = metrics.mae
                row[f"{variant_name}_score"] = metrics.score
            rows.append(row)

    total_steps = max(len(rows), 1)
    summary = {
        name: {
            "wins": float(variant_win_counts[name]),
            "mean_score": float(variant_score_sums[name]) / total_steps,
        }
        for name in VARIANTS
    }
    return rows, summary


def current_indices(phi: np.ndarray) -> np.ndarray:
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["exp_id", "step", "samples", "winner"]
    for variant_name in VARIANTS:
        fieldnames.append(f"{variant_name}_mae")
        fieldnames.append(f"{variant_name}_score")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, object]], summary: dict[str, dict[str, float]]) -> None:
    lines: list[str] = []
    lines.append("# Stencil Repair Ablation")
    lines.append("")
    lines.append("Replace selected groups in the rebuilt explicit stencil by exact-SDF values, then run the paper model.")
    lines.append("")
    lines.append(
        "| case | step | winner | current_score | center_score | cross_score | diag_score | center_cross_score | full_exact_score |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['exp_id']} | {row['step']} | {row['winner']} | "
            f"{float(row['current_score']):.6f} | "
            f"{float(row['center_exact_score']):.6f} | "
            f"{float(row['cross_exact_score']):.6f} | "
            f"{float(row['diag_exact_score']):.6f} | "
            f"{float(row['center_cross_exact_score']):.6f} | "
            f"{float(row['full_exact_score']):.6f} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| variant | wins | mean_score |")
    lines.append("| --- | ---: | ---: |")
    for variant_name in VARIANTS:
        lines.append(
            f"| {variant_name} | {int(summary[variant_name]['wins'])} | {summary[variant_name]['mean_score']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output_paths(project_root: Path, raw_stem: str) -> tuple[Path, Path]:
    if raw_stem:
        stem_path = Path(raw_stem)
        if not stem_path.is_absolute():
            stem_path = project_root / stem_path
    else:
        stamp = datetime.now().strftime("%m%d")
        stem_path = project_root / "test" / "results" / f"stencil_repair_ablation_{stamp}"
    return stem_path.with_suffix(".csv"), stem_path.with_suffix(".md")


if __name__ == "__main__":
    main()
