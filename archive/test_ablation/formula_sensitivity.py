from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate.stencil_encoding import encode_patch_legacy_flat, encode_patch_training_order
from generate.dataset_test import LevelSetReinitializer, build_flower_phi0, build_grid
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config
from project_runtime import cleanup_bytecode_caches, disable_bytecode_cache

from .paper_alignment import (
    MetricTriple,
    OriginPredictorLite,
    compute_metrics,
    format_metric_delta,
    format_metric_triple,
    load_payload,
    normalized_distance,
)
from .paper_reference import PAPER_UNIFORM_REFERENCES, get_paper_reference

disable_bytecode_cache()

NUMERICAL_VARIANTS = ("stored_eq3", "field_eq3", "field_div_normal", "axis_flip_eq3")
ORIGIN_VARIANTS = ("training_order", "legacy_flat")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare numerical-formula and stencil-order variants against paper tables.")
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
        help="Optional explicit report path. Defaults to test/results/formula_sensitivity_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_formula_sensitivity_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nFormula sensitivity report saved to {output_path}")


def build_formula_sensitivity_report(project_root: Path, *, config_path: str | Path | None = None) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    origin_predictor = OriginPredictorLite(project_root)
    data_root = project_config.project.data_dir

    numerical_case_wins = {name: 0 for name in NUMERICAL_VARIANTS}
    numerical_step_wins = {name: 0 for name in NUMERICAL_VARIANTS}
    origin_case_wins = {name: 0 for name in ORIGIN_VARIANTS}
    origin_step_wins = {name: 0 for name in ORIGIN_VARIANTS}
    total_cases = 0
    total_steps = 0

    lines: list[str] = []
    lines.append("Formula Sensitivity Report")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Config: {project_config.source_path}")
    lines.append(f"Data root: {data_root}")
    lines.append("")

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        if exp_id not in PAPER_UNIFORM_REFERENCES:
            continue

        total_cases += 1
        case_numerical_scores = {name: 0.0 for name in NUMERICAL_VARIANTS}
        case_origin_scores = {name: 0.0 for name in ORIGIN_VARIANTS}

        lines.append(f"[{exp_id}] {PAPER_UNIFORM_REFERENCES[exp_id].table_id}")
        lines.append(
            f"meta: h={scenario.h:.9e} | rho_model={scenario.rho_model} | "
            f"paper_samples={PAPER_UNIFORM_REFERENCES[exp_id].sample_count}"
        )

        x_grid, y_grid, h = build_grid(float(scenario.L), int(scenario.N))
        phi0 = build_flower_phi0(x_grid, y_grid, float(scenario.a), float(scenario.b), float(scenario.p))
        reinitializer = LevelSetReinitializer(cfl=project_config.generation.cfl)

        for step in project_config.evaluation.test_iters:
            total_steps += 1
            payload = load_payload(data_root / exp_id / f"iter_{int(step)}.h5")
            phi = reinitializer.reinitialize(phi0, h, int(step))
            indices = payload["indices"]
            paper_ref = get_paper_reference(exp_id, int(step))

            numerical_predictions = build_numerical_variants(phi, indices, h, payload["numerical"])
            origin_predictions = build_origin_variants(phi, indices, origin_predictor, int(scenario.rho_model))

            lines.append(f"  step={int(step)}")
            num_scores: dict[str, float] = {}
            for variant_name, prediction in numerical_predictions.items():
                metrics = compute_metrics(payload["target"], prediction)
                score = normalized_distance(metrics, paper_ref.numerical)
                case_numerical_scores[variant_name] += score
                num_scores[variant_name] = score
                lines.append(
                    f"    numerical/{variant_name}: "
                    f"{format_metric_triple(metrics)} | "
                    f"|d_paper|={format_metric_delta(metrics, paper_ref.numerical)} | "
                    f"score={score:.6f}"
                )

            origin_scores: dict[str, float] = {}
            for variant_name, prediction in origin_predictions.items():
                metrics = compute_metrics(payload["target"], prediction)
                score = normalized_distance(metrics, paper_ref.paper_model)
                case_origin_scores[variant_name] += score
                origin_scores[variant_name] = score
                lines.append(
                    f"    origin/{variant_name}: "
                    f"{format_metric_triple(metrics)} | "
                    f"|d_paper|={format_metric_delta(metrics, paper_ref.paper_model)} | "
                    f"score={score:.6f}"
                )

            num_winner = min(num_scores, key=num_scores.get)
            origin_winner = min(origin_scores, key=origin_scores.get)
            numerical_step_wins[num_winner] += 1
            origin_step_wins[origin_winner] += 1
            lines.append(
                "    step_summary: "
                f"numerical_winner={num_winner} | origin_winner={origin_winner}"
            )

        num_case_winner = min(case_numerical_scores, key=case_numerical_scores.get)
        origin_case_winner = min(case_origin_scores, key=case_origin_scores.get)
        numerical_case_wins[num_case_winner] += 1
        origin_case_wins[origin_case_winner] += 1
        lines.append(
            "  case_summary: "
            f"numerical_winner={num_case_winner} | origin_winner={origin_case_winner}"
        )
        lines.append(
            "  case_scores: "
            + " | ".join(f"num/{name}={case_numerical_scores[name]:.6f}" for name in NUMERICAL_VARIANTS)
            + " | "
            + " | ".join(f"origin/{name}={case_origin_scores[name]:.6f}" for name in ORIGIN_VARIANTS)
        )
        lines.append("")

    lines.insert(
        4,
        "Summary: "
        + ", ".join(f"num {name} {numerical_case_wins[name]}/{max(total_cases,1)} cases {numerical_step_wins[name]}/{max(total_steps,1)} steps" for name in NUMERICAL_VARIANTS)
        + " || "
        + ", ".join(f"origin {name} {origin_case_wins[name]}/{max(total_cases,1)} cases {origin_step_wins[name]}/{max(total_steps,1)} steps" for name in ORIGIN_VARIANTS),
    )
    return "\n".join(lines).rstrip()


def build_numerical_variants(phi: np.ndarray, indices: np.ndarray, h: float, stored_prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "stored_eq3": np.asarray(stored_prediction, dtype=np.float64),
        "field_eq3": hkappa_from_full_field_standard(phi, indices, h),
        "field_div_normal": hkappa_from_full_field_div_normal(phi, indices, h),
        "axis_flip_eq3": hkappa_fd_variant_from_phi(phi, indices, h),
    }


def build_origin_variants(
    phi: np.ndarray,
    indices: np.ndarray,
    origin_predictor: OriginPredictorLite,
    rho_model: int,
) -> dict[str, np.ndarray]:
    training_stencils = []
    legacy_stencils = []
    for row, col in indices:
        patch = phi[row - 1:row + 2, col - 1:col + 2]
        training_stencils.append(encode_patch_training_order(patch))
        legacy_stencils.append(encode_patch_legacy_flat(patch))
    training_array = np.asarray(training_stencils, dtype=np.float64) if training_stencils else np.zeros((0, 9), dtype=np.float64)
    legacy_array = np.asarray(legacy_stencils, dtype=np.float64) if legacy_stencils else np.zeros((0, 9), dtype=np.float64)
    return {
        "training_order": origin_predictor.predict(rho_model, training_array),
        "legacy_flat": origin_predictor.predict(rho_model, legacy_array),
    }


def hkappa_from_full_field_standard(phi: np.ndarray, indices: np.ndarray, h: float) -> np.ndarray:
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


def hkappa_from_full_field_div_normal(phi: np.ndarray, indices: np.ndarray, h: float) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float64)
    phi_x = np.zeros_like(phi, dtype=np.float64)
    phi_y = np.zeros_like(phi, dtype=np.float64)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    grad_norm = np.sqrt(phi_x**2 + phi_y**2)
    nx = np.divide(phi_x, grad_norm, out=np.zeros_like(phi_x), where=grad_norm > 0.0)
    ny = np.divide(phi_y, grad_norm, out=np.zeros_like(phi_y), where=grad_norm > 0.0)
    dnx_dx = np.zeros_like(phi, dtype=np.float64)
    dny_dy = np.zeros_like(phi, dtype=np.float64)
    dnx_dx[:, 1:-1] = (nx[:, 2:] - nx[:, :-2]) / (2.0 * h)
    dny_dy[1:-1, :] = (ny[2:, :] - ny[:-2, :]) / (2.0 * h)
    return h * (dnx_dx + dny_dy)[indices[:, 0], indices[:, 1]]


def hkappa_fd_variant_from_phi(phi: np.ndarray, indices: np.ndarray, h: float) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float64)
    rows = indices[:, 0]
    cols = indices[:, 1]
    phi_x = (phi[rows, cols + 1] - phi[rows, cols - 1]) / (2.0 * h)
    phi_y = (phi[rows - 1, cols] - phi[rows + 1, cols]) / (2.0 * h)
    phi_xx = (phi[rows, cols + 1] - 2.0 * phi[rows, cols] + phi[rows, cols - 1]) / (h**2)
    phi_yy = (phi[rows - 1, cols] - 2.0 * phi[rows, cols] + phi[rows + 1, cols]) / (h**2)
    phi_xy = (
        phi[rows - 1, cols + 1]
        - phi[rows - 1, cols - 1]
        - phi[rows + 1, cols + 1]
        + phi[rows + 1, cols - 1]
    ) / (4.0 * h**2)
    numerator = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2) ** 1.5
    prediction = np.zeros_like(numerator)
    mask = denominator > 1e-12
    prediction[mask] = numerator[mask] / denominator[mask]
    return h * prediction


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"formula_sensitivity_{stamp}.txt"


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_bytecode_caches(PROJECT_ROOT)
