from __future__ import annotations

import argparse
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

from .paper_alignment import OriginPredictorLite, compute_metrics, load_origin_stats, normalized_distance
from .paper_reference import get_paper_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExactStencilPayload:
    target: np.ndarray
    current_stencils: np.ndarray
    exact_stencils: np.ndarray
    current_numerical: np.ndarray
    exact_numerical: np.ndarray
    sample_count: int
    patch_phi_mae: float
    patch_phi_max: float
    current_eikonal_mae: float
    stencil_raw_mae: float
    stencil_raw_max: float
    stencil_z_mae: float
    stencil_z_max: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure an analytic signed-distance upper bound for rebuilt explicit flower tests."
    )
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
        help="Optional explicit report path. Defaults to test/results/explicit_sdf_upper_bound_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[Saved] {output_path}")


def build_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    predictor = OriginPredictorLite(project_root)
    lines: list[str] = []
    total_steps = 0
    origin_step_wins = 0
    numerical_step_wins = 0

    lines.append("Explicit SDF Upper-Bound Report")
    lines.append("")
    lines.append(
        "Same sample nodes, same targets; only replace the rebuilt explicit-test phi values by analytic signed-distance stencils."
    )
    lines.append(
        "If the paper gap shrinks sharply under this swap, the main remaining issue is the irregular-test phi0 -> reinit chain."
    )
    lines.append("")

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        mu, sigma = load_origin_stats(project_root / "model" / "origin" / f"trainStats_{int(scenario.rho_model)}.csv")
        lines.append(
            f"{exp_id} | rho_model={int(scenario.rho_model)} | "
            f"case_h={float(scenario.h):.9e} | rho_eq_case={1.0 / float(scenario.h) + 1.0:.6f}"
        )

        X, Y, h = build_grid(float(scenario.L), int(scenario.N))
        phi0 = build_flower_phi0(X, Y, float(scenario.a), float(scenario.b), float(scenario.p))
        reinitializer = LevelSetReinitializer(cfl=float(project_config.generation.cfl))

        for step in project_config.evaluation.test_iters:
            phi = reinitializer.reinitialize(phi0, h, int(step))
            payload = build_exact_stencil_payload(
                phi=phi,
                phi0=phi0,
                X=X,
                Y=Y,
                h=h,
                train_mu=mu.astype(np.float64),
                train_sigma=sigma.astype(np.float64),
                a=float(scenario.a),
                b=float(scenario.b),
                p=int(scenario.p),
            )
            paper_ref = get_paper_reference(exp_id, int(step))

            current_origin = compute_metrics(
                payload.target,
                predictor.predict(int(scenario.rho_model), payload.current_stencils),
            )
            exact_origin = compute_metrics(
                payload.target,
                predictor.predict(int(scenario.rho_model), payload.exact_stencils),
            )
            current_numerical = compute_metrics(payload.target, payload.current_numerical)
            exact_numerical = compute_metrics(payload.target, payload.exact_numerical)

            current_origin_score = normalized_distance(current_origin, paper_ref.paper_model)
            exact_origin_score = normalized_distance(exact_origin, paper_ref.paper_model)
            current_num_score = normalized_distance(current_numerical, paper_ref.numerical)
            exact_num_score = normalized_distance(exact_numerical, paper_ref.numerical)

            total_steps += 1
            if exact_origin_score < current_origin_score:
                origin_step_wins += 1
            if exact_num_score < current_num_score:
                numerical_step_wins += 1

            lines.append(
                f"  step={int(step)} | samples={payload.sample_count} | "
                f"patch_phi_mae={payload.patch_phi_mae:.6e} | patch_phi_max={payload.patch_phi_max:.6e} | "
                f"current_eikonal_mae={payload.current_eikonal_mae:.6e} | "
                f"stencil_raw_mae={payload.stencil_raw_mae:.6e} | stencil_raw_max={payload.stencil_raw_max:.6e} | "
                f"stencil_z_mae={payload.stencil_z_mae:.6e} | stencil_z_max={payload.stencil_z_max:.6e}"
            )
            lines.append(
                "    origin/current: "
                f"MAE={current_origin.mae:.6e}, MaxAE={current_origin.max_ae:.6e}, MSE={current_origin.mse:.6e} | "
                f"|d_paper| score={current_origin_score:.6f}"
            )
            lines.append(
                "    origin/exact_sdf: "
                f"MAE={exact_origin.mae:.6e}, MaxAE={exact_origin.max_ae:.6e}, MSE={exact_origin.mse:.6e} | "
                f"|d_paper| score={exact_origin_score:.6f} | "
                f"winner={'exact_sdf' if exact_origin_score < current_origin_score else 'current'}"
            )
            lines.append(
                "    numerical/current: "
                f"MAE={current_numerical.mae:.6e}, MaxAE={current_numerical.max_ae:.6e}, MSE={current_numerical.mse:.6e} | "
                f"|d_paper| score={current_num_score:.6f}"
            )
            lines.append(
                "    numerical/exact_sdf: "
                f"MAE={exact_numerical.mae:.6e}, MaxAE={exact_numerical.max_ae:.6e}, MSE={exact_numerical.mse:.6e} | "
                f"|d_paper| score={exact_num_score:.6f} | "
                f"winner={'exact_sdf' if exact_num_score < current_num_score else 'current'}"
            )

        lines.append("")

    lines.append("Summary")
    lines.append(
        f"  - exact_sdf improves the origin row on {origin_step_wins}/{max(total_steps, 1)} case-steps."
    )
    lines.append(
        f"  - exact_sdf improves the numerical row on {numerical_step_wins}/{max(total_steps, 1)} case-steps."
    )
    return "\n".join(lines)


def build_exact_stencil_payload(
    *,
    phi: np.ndarray,
    phi0: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    h: float,
    train_mu: np.ndarray,
    train_sigma: np.ndarray,
    a: float,
    b: float,
    p: int,
) -> ExactStencilPayload:
    indices = current_indices(phi)
    xy = np.column_stack((X[indices[:, 0], indices[:, 1]], Y[indices[:, 0], indices[:, 1]]))
    theta = find_projection_theta(xy, a, b, p)
    target = hkappa_analytic(theta, h, a, b, p)

    exact_phi = build_exact_signed_distance_field(
        phi_shape=phi.shape,
        phi0=phi0,
        X=X,
        Y=Y,
        indices=indices,
        a=a,
        b=b,
        p=p,
    )

    current_stencils = extract_3x3_stencils(phi, indices)
    exact_stencils = extract_3x3_stencils(exact_phi, indices)
    current_numerical = compute_eq3_from_field(phi, indices, h)
    exact_numerical = compute_eq3_from_field(exact_phi, indices, h)

    patch_mask = patch_node_mask(phi.shape, indices)
    patch_diff = np.abs(phi[patch_mask] - exact_phi[patch_mask])
    stencil_diff = np.abs(current_stencils - exact_stencils)
    current_z = (current_stencils - train_mu) / train_sigma
    exact_z = (exact_stencils - train_mu) / train_sigma
    stencil_z_diff = np.abs(current_z - exact_z)
    current_grad_norm = central_grad_norm(phi, h)
    sample_grad_err = np.abs(current_grad_norm[indices[:, 0], indices[:, 1]] - 1.0)

    return ExactStencilPayload(
        target=target,
        current_stencils=current_stencils,
        exact_stencils=exact_stencils,
        current_numerical=current_numerical,
        exact_numerical=exact_numerical,
        sample_count=int(indices.shape[0]),
        patch_phi_mae=float(patch_diff.mean()) if patch_diff.size else 0.0,
        patch_phi_max=float(patch_diff.max()) if patch_diff.size else 0.0,
        current_eikonal_mae=float(sample_grad_err.mean()) if sample_grad_err.size else 0.0,
        stencil_raw_mae=float(stencil_diff.mean()) if stencil_diff.size else 0.0,
        stencil_raw_max=float(stencil_diff.max()) if stencil_diff.size else 0.0,
        stencil_z_mae=float(stencil_z_diff.mean()) if stencil_z_diff.size else 0.0,
        stencil_z_max=float(stencil_z_diff.max()) if stencil_z_diff.size else 0.0,
    )


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


def build_exact_signed_distance_field(
    *,
    phi_shape: tuple[int, int],
    phi0: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    indices: np.ndarray,
    a: float,
    b: float,
    p: int,
) -> np.ndarray:
    mask = patch_node_mask(phi_shape, indices)
    rows, cols = np.where(mask)
    xy = np.column_stack((X[rows, cols], Y[rows, cols]))
    theta = find_projection_theta(xy, a, b, p)
    radius = b + a * np.cos(p * theta)
    curve_x = radius * np.cos(theta)
    curve_y = radius * np.sin(theta)
    distance = np.sqrt((xy[:, 0] - curve_x) ** 2 + (xy[:, 1] - curve_y) ** 2)
    sign = np.sign(phi0[rows, cols])
    exact_phi = np.zeros(phi_shape, dtype=np.float64)
    exact_phi[rows, cols] = sign * distance
    return exact_phi


def compute_eq3_from_field(phi: np.ndarray, indices: np.ndarray, h: float) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float64)
    rows = indices[:, 0]
    cols = indices[:, 1]
    phi_x = (phi[rows, cols + 1] - phi[rows, cols - 1]) / (2.0 * h)
    phi_y = (phi[rows + 1, cols] - phi[rows - 1, cols]) / (2.0 * h)
    phi_xx = (phi[rows, cols + 1] - 2.0 * phi[rows, cols] + phi[rows, cols - 1]) / (h**2)
    phi_yy = (phi[rows + 1, cols] - 2.0 * phi[rows, cols] + phi[rows - 1, cols]) / (h**2)
    phi_xy = (
        phi[rows + 1, cols + 1]
        - phi[rows + 1, cols - 1]
        - phi[rows - 1, cols + 1]
        + phi[rows - 1, cols - 1]
    ) / (4.0 * h**2)
    numerator = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2) ** 1.5
    prediction = np.zeros_like(numerator)
    mask = denominator > 1e-12
    prediction[mask] = numerator[mask] / denominator[mask]
    return h * prediction


def central_grad_norm(phi: np.ndarray, h: float) -> np.ndarray:
    dy, dx = np.gradient(phi, h, h, edge_order=1)
    return np.sqrt(dx**2 + dy**2)


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"explicit_sdf_upper_bound_{stamp}.txt"


if __name__ == "__main__":
    main()
