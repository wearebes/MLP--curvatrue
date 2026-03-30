from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
    hkappa_div_normal_from_field,
)
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config

from .explicit_sdf_upper_bound import build_exact_signed_distance_field
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
    grad_err_mean: float
    grad_err_max: float
    grad_norm_std: float
    patch_phi_mae: float
    patch_phi_max: float
    stencil_raw_mae: float
    stencil_raw_rmse: float
    stencil_z_mae: float
    stencil_z_rmse: float
    z_center_mae: float
    z_cross_mae: float
    z_diag_mae: float
    origin_mae: float
    origin_maxae: float
    origin_mse: float
    numerical_mae: float
    numerical_maxae: float
    numerical_mse: float
    origin_paper_score: float
    numerical_paper_score: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabulate phi-field quality for rebuilt explicit tests against exact signed-distance stencils."
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
        default="test/results/phi_field_quality_table_0325_2",
        help="Output stem relative to project root; writes both .csv and .md",
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
        reinitializer = LevelSetReinitializer(
            cfl=float(project_config.generation.cfl),
            time_order=int(project_config.generation.time_order),
            space_order=int(project_config.generation.space_order),
        )

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
            dz = (current_stencils - mu) / sigma - (exact_stencils - mu) / sigma
            group_mae = {
                name: mean_abs(dz[:, positions])
                for name, positions in GROUPS.items()
            }

            patch_mask = patch_node_mask(phi.shape, indices)
            patch_phi_diff = phi[patch_mask] - exact_phi[patch_mask]
            stencil_diff = current_stencils - exact_stencils

            grad_metrics = ReinitQualityEvaluator.evaluate(phi, h)

            origin_metrics = compute_metrics(
                target,
                predictor.predict(int(scenario.rho_model), current_stencils),
            )
            numerical_metrics = compute_metrics(
                target,
                hkappa_div_normal_from_field(phi, indices, h),
            )
            paper_ref = get_paper_reference(exp_id, int(step))

            rows.append(
                ResultRow(
                    exp_id=exp_id,
                    step=int(step),
                    samples=int(indices.shape[0]),
                    grad_err_mean=float(grad_metrics["mean_abs_err_to_1"]),
                    grad_err_max=float(grad_metrics["max_abs_err_to_1"]),
                    grad_norm_std=float(grad_metrics["grad_norm_std"]),
                    patch_phi_mae=mean_abs(patch_phi_diff),
                    patch_phi_max=max_abs(patch_phi_diff),
                    stencil_raw_mae=mean_abs(stencil_diff),
                    stencil_raw_rmse=rmse(stencil_diff),
                    stencil_z_mae=mean_abs(dz),
                    stencil_z_rmse=rmse(dz),
                    z_center_mae=float(group_mae["center"]),
                    z_cross_mae=float(group_mae["cross"]),
                    z_diag_mae=float(group_mae["diag"]),
                    origin_mae=float(origin_metrics.mae),
                    origin_maxae=float(origin_metrics.max_ae),
                    origin_mse=float(origin_metrics.mse),
                    numerical_mae=float(numerical_metrics.mae),
                    numerical_maxae=float(numerical_metrics.max_ae),
                    numerical_mse=float(numerical_metrics.mse),
                    origin_paper_score=float(normalized_distance(origin_metrics, paper_ref.paper_model)),
                    numerical_paper_score=float(normalized_distance(numerical_metrics, paper_ref.numerical)),
                )
            )

    return rows


def current_indices(phi: np.ndarray) -> np.ndarray:
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)


def patch_node_mask(shape: tuple[int, int], indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if indices.size == 0:
        return mask
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            mask[indices[:, 0] + di, indices[:, 1] + dj] = True
    return mask


def mean_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def max_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr**2)))


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ResultRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(path: Path, rows: list[ResultRow]) -> None:
    lines: list[str] = []
    lines.append("# Phi Field Quality Table")
    lines.append("")
    lines.append("Current rebuilt explicit-test fields measured against exact signed-distance stencils on the same sample nodes.")
    lines.append("")
    lines.append(
        "| case | step | samples | grad_err_mean | grad_err_max | patch_phi_mae | stencil_z_rmse | z_center_mae | z_cross_mae | z_diag_mae | origin_mae | numerical_mae |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        lines.append(
            f"| {row.exp_id} | {row.step} | {row.samples} | "
            f"{row.grad_err_mean:.6e} | {row.grad_err_max:.6e} | {row.patch_phi_mae:.6e} | "
            f"{row.stencil_z_rmse:.6e} | {row.z_center_mae:.6e} | {row.z_cross_mae:.6e} | {row.z_diag_mae:.6e} | "
            f"{row.origin_mae:.6e} | {row.numerical_mae:.6e} |"
        )

    lines.append("")
    lines.append("## Case Means")
    lines.append("")
    lines.append("| case | mean grad_err_mean | mean patch_phi_mae | mean stencil_z_rmse | mean origin_mae | mean numerical_mae | dominant group |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for exp_id in sorted({row.exp_id for row in rows}):
        subset = [row for row in rows if row.exp_id == exp_id]
        dominant = max(
            ("center", "cross", "diag"),
            key=lambda name: np.mean([
                getattr(row, f"z_{name}_mae")
                for row in subset
            ]),
        )
        lines.append(
            f"| {exp_id} | "
            f"{np.mean([row.grad_err_mean for row in subset]):.6e} | "
            f"{np.mean([row.patch_phi_mae for row in subset]):.6e} | "
            f"{np.mean([row.stencil_z_rmse for row in subset]):.6e} | "
            f"{np.mean([row.origin_mae for row in subset]):.6e} | "
            f"{np.mean([row.numerical_mae for row in subset]):.6e} | "
            f"{dominant} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output_paths(project_root: Path, raw_stem: str) -> tuple[Path, Path]:
    stem_path = Path(raw_stem)
    if not stem_path.is_absolute():
        stem_path = project_root / stem_path
    return stem_path.with_suffix(".csv"), stem_path.with_suffix(".md")


if __name__ == "__main__":
    main()
