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

from .explicit_sdf_upper_bound import build_exact_signed_distance_field, central_grad_norm
from .paper_alignment import OriginPredictorLite, compute_metrics, load_origin_stats, normalized_distance
from .paper_reference import get_paper_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "center": np.array([4], dtype=np.int64),
    "cross": np.array([1, 3, 5, 7], dtype=np.int64),
    "diag": np.array([0, 2, 6, 8], dtype=np.int64),
}


@dataclass(frozen=True)
class ResultRow:
    exp_id: str
    step: int
    samples: int
    current_eikonal_mae: float
    current_origin_mae: float
    exact_origin_mae: float
    paper_score_current: float
    paper_score_exact: float
    z_center_mae: float
    z_center_max: float
    z_cross_mae: float
    z_cross_max: float
    z_diag_mae: float
    z_diag_max: float
    dominant_group: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabulate normalized stencil gaps between rebuilt explicit stencils and exact-SDF stencils."
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
        help="Optional output stem relative to project root. Defaults to test/results/stencil_gap_table_MMDD.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    rows = build_rows(project_config.project_root, config_path=args.config)
    csv_path, md_path = resolve_output_paths(project_config.project_root, args.output_stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"[Saved CSV] {csv_path}")
    print(f"[Saved MD ] {md_path}")


def build_rows(project_root: Path, *, config_path: str | Path | None = None) -> list[ResultRow]:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    predictor = OriginPredictorLite(project_root)
    rows: list[ResultRow] = []

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        mu, sigma = load_origin_stats(project_root / "model" / "origin" / f"trainStats_{int(scenario.rho_model)}.csv")
        mu = mu.astype(np.float64)
        sigma = sigma.astype(np.float64)

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
            current_z = (current_stencils - mu) / sigma
            exact_z = (exact_stencils - mu) / sigma
            dz = current_z - exact_z
            group_stats = {
                name: summarize_group(dz[:, positions])
                for name, positions in GROUPS.items()
            }
            dominant_group = max(group_stats.items(), key=lambda item: item[1][0])[0]

            current_pred = predictor.predict(int(scenario.rho_model), current_stencils)
            exact_pred = predictor.predict(int(scenario.rho_model), exact_stencils)
            current_metrics = compute_metrics(target, current_pred)
            exact_metrics = compute_metrics(target, exact_pred)
            paper_ref = get_paper_reference(exp_id, int(step))
            current_score = normalized_distance(current_metrics, paper_ref.paper_model)
            exact_score = normalized_distance(exact_metrics, paper_ref.paper_model)
            grad_norm = central_grad_norm(phi, h)
            grad_err = np.abs(grad_norm[indices[:, 0], indices[:, 1]] - 1.0)

            rows.append(
                ResultRow(
                    exp_id=exp_id,
                    step=int(step),
                    samples=int(indices.shape[0]),
                    current_eikonal_mae=float(grad_err.mean()) if grad_err.size else 0.0,
                    current_origin_mae=float(current_metrics.mae),
                    exact_origin_mae=float(exact_metrics.mae),
                    paper_score_current=float(current_score),
                    paper_score_exact=float(exact_score),
                    z_center_mae=float(group_stats["center"][0]),
                    z_center_max=float(group_stats["center"][1]),
                    z_cross_mae=float(group_stats["cross"][0]),
                    z_cross_max=float(group_stats["cross"][1]),
                    z_diag_mae=float(group_stats["diag"][0]),
                    z_diag_max=float(group_stats["diag"][1]),
                    dominant_group=dominant_group,
                )
            )

    return rows


def current_indices(phi: np.ndarray) -> np.ndarray:
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)


def summarize_group(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    abs_values = np.abs(np.asarray(values, dtype=np.float64))
    return float(abs_values.mean()), float(abs_values.max())


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    fieldnames = [
        "exp_id",
        "step",
        "samples",
        "current_eikonal_mae",
        "current_origin_mae",
        "exact_origin_mae",
        "paper_score_current",
        "paper_score_exact",
        "z_center_mae",
        "z_center_max",
        "z_cross_mae",
        "z_cross_max",
        "z_diag_mae",
        "z_diag_max",
        "dominant_group",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(path: Path, rows: list[ResultRow]) -> None:
    lines: list[str] = []
    lines.append("# Stencil Gap Table")
    lines.append("")
    lines.append("Current rebuilt explicit stencils vs exact-SDF stencils on the same sample nodes.")
    lines.append("")
    lines.append(
        "| case | step | samples | eikonal_mae | origin_mae_current | origin_mae_exact | score_current | score_exact | z_center_mae | z_cross_mae | z_diag_mae | dominant |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for row in rows:
        lines.append(
            f"| {row.exp_id} | {row.step} | {row.samples} | "
            f"{row.current_eikonal_mae:.6e} | {row.current_origin_mae:.6e} | {row.exact_origin_mae:.6e} | "
            f"{row.paper_score_current:.6f} | {row.paper_score_exact:.6f} | "
            f"{row.z_center_mae:.6e} | {row.z_cross_mae:.6e} | {row.z_diag_mae:.6e} | {row.dominant_group} |"
        )

    lines.append("")
    lines.append("## Group Means")
    lines.append("")
    lines.append("| case | mean z_center_mae | mean z_cross_mae | mean z_diag_mae | dominant majority |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for exp_id in sorted({row.exp_id for row in rows}):
        subset = [row for row in rows if row.exp_id == exp_id]
        dominant = max(
            ("center", "cross", "diag"),
            key=lambda name: sum(1 for row in subset if row.dominant_group == name),
        )
        lines.append(
            f"| {exp_id} | "
            f"{np.mean([row.z_center_mae for row in subset]):.6e} | "
            f"{np.mean([row.z_cross_mae for row in subset]):.6e} | "
            f"{np.mean([row.z_diag_mae for row in subset]):.6e} | "
            f"{dominant} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output_paths(project_root: Path, raw_stem: str) -> tuple[Path, Path]:
    if raw_stem:
        stem_path = Path(raw_stem)
        if not stem_path.is_absolute():
            stem_path = project_root / stem_path
    else:
        stamp = datetime.now().strftime("%m%d")
        stem_path = project_root / "test" / "results" / f"stencil_gap_table_{stamp}"
    return stem_path.with_suffix(".csv"), stem_path.with_suffix(".md")


if __name__ == "__main__":
    main()
