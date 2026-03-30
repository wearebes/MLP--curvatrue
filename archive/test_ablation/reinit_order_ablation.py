from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from generate.pde_utils import (
    LevelSetReinitializer as TrainReinitializer,
    ReinitQualityEvaluator,
)
from generate.stencil_encoding import extract_3x3_stencils
from generate.dataset_test import (
    LevelSetReinitializer as TestReinitializer,
    build_flower_phi0,
    build_grid,
    find_projection_theta,
    hkappa_analytic,
)
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config

from .explicit_sdf_upper_bound import build_exact_signed_distance_field, central_grad_norm
from .paper_alignment import OriginPredictorLite, compute_metrics, normalized_distance
from .paper_reference import get_paper_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FirstOrderReinitializer:
    def __init__(self, cfl: float = 0.5, eps_sign_factor: float = 1.0):
        self.cfl = float(cfl)
        self.eps_sign_factor = float(eps_sign_factor)

    def _smoothed_sign(self, phi0: np.ndarray, h: float) -> np.ndarray:
        eps = self.eps_sign_factor * h
        return phi0 / np.sqrt(phi0**2 + eps**2)

    def _get_derivatives_first_order(self, phi: np.ndarray, h: float):
        phi_pad = np.pad(phi, pad_width=1, mode="edge")
        dx_m = (phi_pad[1:-1, 1:-1] - phi_pad[1:-1, :-2]) / h
        dx_p = (phi_pad[1:-1, 2:] - phi_pad[1:-1, 1:-1]) / h
        dy_m = (phi_pad[1:-1, 1:-1] - phi_pad[:-2, 1:-1]) / h
        dy_p = (phi_pad[2:, 1:-1] - phi_pad[1:-1, 1:-1]) / h
        return dx_m, dx_p, dy_m, dy_p

    def _godunov_grad_norm(self, dx_m, dx_p, dy_m, dy_p, s0):
        grad_plus = np.sqrt(
            np.maximum(np.maximum(dx_m, -dx_p), 0.0) ** 2
            + np.maximum(np.maximum(dy_m, -dy_p), 0.0) ** 2
        )
        grad_minus = np.sqrt(
            np.maximum(np.maximum(-dx_m, dx_p), 0.0) ** 2
            + np.maximum(np.maximum(-dy_m, dy_p), 0.0) ** 2
        )
        return np.where(s0 >= 0, grad_plus, grad_minus)

    def _compute_rhs(self, phi: np.ndarray, s0: np.ndarray, h: float) -> np.ndarray:
        dx_m, dx_p, dy_m, dy_p = self._get_derivatives_first_order(phi, h)
        grad_g = self._godunov_grad_norm(dx_m, dx_p, dy_m, dy_p, s0)
        return -s0 * (grad_g - 1.0)

    def reinitialize(self, phi0: np.ndarray, h: float, n_steps: int) -> np.ndarray:
        if n_steps <= 0:
            return phi0.copy()
        phi = phi0.astype(np.float64, copy=True)
        s0 = self._smoothed_sign(phi, h)
        dt = self.cfl * h
        for _ in range(n_steps):
            l1 = self._compute_rhs(phi, s0, h)
            phi_1 = phi + dt * l1
            l2 = self._compute_rhs(phi_1, s0, h)
            phi_2 = 0.75 * phi + 0.25 * (phi_1 + dt * l2)
            l3 = self._compute_rhs(phi_2, s0, h)
            phi = (1.0 / 3.0) * phi + (2.0 / 3.0) * (phi_2 + dt * l3)
        return phi


@dataclass(frozen=True)
class VariantResult:
    origin_mae: float
    origin_score: float
    z_gap_mae: float
    eikonal_mae: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare explicit-test reinitialization variants against exact-SDF and the paper model."
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
        help="Optional output stem relative to project root. Defaults to test/results/reinit_order_ablation_MMDD.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    rows, summary = build_results(project_config.project_root, config_path=args.config)
    csv_path, md_path = resolve_output_paths(project_config.project_root, args.output_stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, summary)
    print(f"[Saved CSV] {csv_path}")
    print(f"[Saved MD ] {md_path}")


def build_results(
    project_root: Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    predictor = OriginPredictorLite(project_root)
    variants = {
        "current_test_weno5": TestReinitializer(cfl=float(project_config.generation.cfl)),
        "train_weno5": TrainReinitializer(cfl=float(project_config.generation.cfl)),
        "first_order": FirstOrderReinitializer(cfl=float(project_config.generation.cfl)),
    }
    rows: list[dict[str, object]] = []
    variant_win_counts = {name: 0 for name in variants}
    variant_score_sums = {name: 0.0 for name in variants}

    for scenario in project_config.evaluation.scenarios:
        mu, sigma = load_origin_stats_simple(project_root / "model" / "origin" / f"trainStats_{int(scenario.rho_model)}.csv")
        mu = mu.astype(np.float64)
        sigma = sigma.astype(np.float64)

        X, Y, h = build_grid(float(scenario.L), int(scenario.N))
        phi0 = build_flower_phi0(X, Y, float(scenario.a), float(scenario.b), float(scenario.p))

        for step in project_config.evaluation.test_iters:
            paper_ref = get_paper_reference(scenario.exp_id, int(step))

            base_phi = TestReinitializer(cfl=float(project_config.generation.cfl)).reinitialize(phi0, h, int(step))
            indices = current_indices(base_phi)
            xy = np.column_stack((X[indices[:, 0], indices[:, 1]], Y[indices[:, 0], indices[:, 1]]))
            theta = find_projection_theta(xy, float(scenario.a), float(scenario.b), float(scenario.p))
            target = hkappa_analytic(theta, h, float(scenario.a), float(scenario.b), float(scenario.p))
            exact_phi = build_exact_signed_distance_field(
                phi_shape=base_phi.shape,
                phi0=phi0,
                X=X,
                Y=Y,
                indices=indices,
                a=float(scenario.a),
                b=float(scenario.b),
                p=int(scenario.p),
            )
            exact_stencils = extract_3x3_stencils(exact_phi, indices)
            exact_z = (exact_stencils - mu) / sigma

            per_variant: dict[str, VariantResult] = {}
            for variant_name, reinitializer in variants.items():
                phi = reinitializer.reinitialize(phi0, h, int(step))
                stencils = extract_3x3_stencils(phi, indices)
                pred = predictor.predict(int(scenario.rho_model), stencils)
                metrics = compute_metrics(target, pred)
                score = normalized_distance(metrics, paper_ref.paper_model)
                z = (stencils - mu) / sigma
                z_gap = np.abs(z - exact_z)
                grad_norm = central_grad_norm(phi, h)
                grad_err = np.abs(grad_norm[indices[:, 0], indices[:, 1]] - 1.0)
                per_variant[variant_name] = VariantResult(
                    origin_mae=float(metrics.mae),
                    origin_score=float(score),
                    z_gap_mae=float(z_gap.mean()) if z_gap.size else 0.0,
                    eikonal_mae=float(grad_err.mean()) if grad_err.size else 0.0,
                )
                variant_score_sums[variant_name] += float(score)

            winner = min(per_variant, key=lambda name: per_variant[name].origin_score)
            variant_win_counts[winner] += 1
            row: dict[str, object] = {
                "exp_id": scenario.exp_id,
                "step": int(step),
                "samples": int(indices.shape[0]),
                "winner": winner,
            }
            for variant_name, result in per_variant.items():
                row[f"{variant_name}_origin_mae"] = result.origin_mae
                row[f"{variant_name}_origin_score"] = result.origin_score
                row[f"{variant_name}_z_gap_mae"] = result.z_gap_mae
                row[f"{variant_name}_eikonal_mae"] = result.eikonal_mae
            rows.append(row)

    total_steps = max(len(rows), 1)
    summary = {
        name: {
            "wins": float(variant_win_counts[name]),
            "mean_score": float(variant_score_sums[name]) / total_steps,
        }
        for name in variants
    }
    return rows, summary


def current_indices(phi: np.ndarray) -> np.ndarray:
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)


def load_origin_stats_simple(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mu_values: list[float] = []
    sigma_values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            feature_name = row[0].strip().strip('"')
            if feature_name == "h":
                continue
            mu_values.append(float(row[1]))
            sigma_values.append(float(row[2]))
    return np.asarray(mu_values, dtype=np.float64), np.asarray(sigma_values, dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["exp_id", "step", "samples", "winner"]
    for variant_name in ("current_test_weno5", "train_weno5", "first_order"):
        fieldnames.extend(
            [
                f"{variant_name}_origin_mae",
                f"{variant_name}_origin_score",
                f"{variant_name}_z_gap_mae",
                f"{variant_name}_eikonal_mae",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, object]], summary: dict[str, dict[str, float]]) -> None:
    lines: list[str] = []
    lines.append("# Reinit Order Ablation")
    lines.append("")
    lines.append("Same sample nodes and same targets; only swap the reinitialization implementation.")
    lines.append("")
    lines.append(
        "| case | step | winner | current_score | train_score | first_order_score | current_z_gap | train_z_gap | first_order_z_gap |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['exp_id']} | {row['step']} | {row['winner']} | "
            f"{float(row['current_test_weno5_origin_score']):.6f} | "
            f"{float(row['train_weno5_origin_score']):.6f} | "
            f"{float(row['first_order_origin_score']):.6f} | "
            f"{float(row['current_test_weno5_z_gap_mae']):.6e} | "
            f"{float(row['train_weno5_z_gap_mae']):.6e} | "
            f"{float(row['first_order_z_gap_mae']):.6e} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| variant | wins | mean_score |")
    lines.append("| --- | ---: | ---: |")
    for variant_name in ("current_test_weno5", "train_weno5", "first_order"):
        lines.append(
            f"| {variant_name} | {int(summary[variant_name]['wins'])} | {summary[variant_name]['mean_score']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_output_paths(project_root: Path, raw_stem: str) -> tuple[Path, Path]:
    if raw_stem:
        stem_path = Path(raw_stem)
        if not stem_path.is_absolute():
            stem_path = project_root / raw_stem
    else:
        stamp = datetime.now().strftime("%m%d")
        stem_path = project_root / "test" / "results" / f"reinit_order_ablation_{stamp}"
    return stem_path.with_suffix(".csv"), stem_path.with_suffix(".md")


if __name__ == "__main__":
    main()
