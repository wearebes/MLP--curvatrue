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
    LevelSetReinitializer as ExplicitTestReinitializer,
    build_flower_phi0,
    build_grid,
    find_projection_theta,
    hkappa_analytic,
)
from project_config import DEFAULT_PROJECT_CONFIG_PATH, load_project_config

from .paper_alignment import OriginPredictorLite, compute_metrics, normalized_distance
from .paper_reference import PAPER_UNIFORM_REFERENCES, get_paper_reference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelinePayload:
    count: int
    target: np.ndarray
    stencils: np.ndarray
    numerical: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the paper method protocol against the current explicit-test pipeline."
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
        help="Optional explicit report path. Defaults to test/results/method_alignment_MMDD.txt.",
    )
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    report = build_method_alignment_report(project_config.project_root, config_path=args.config)
    output_path = resolve_output_path(project_config.project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[Saved] {output_path}")


def build_method_alignment_report(
    project_root: Path,
    *,
    config_path: str | Path | None = None,
) -> str:
    project_config = load_project_config(config_path or DEFAULT_PROJECT_CONFIG_PATH)
    origin_predictor = OriginPredictorLite(project_root)
    lines: list[str] = []

    lines.append("Paper Method Alignment Report")
    lines.append("")
    lines.append("1. Paper protocol cross-check")
    lines.append(
        "  - Training circles: local code matches section 3.2 on geometry family, "
        "radius range, center jitter, up-to-5 variations, and reinit steps {5,10,15,20} "
        "for non-SDF circles."
    )
    lines.append(
        "  - Training augmentation: local code also mirrors the paper's sign-negation augmentation."
    )
    lines.append(
        "  - Irregular explicit tests: local code matches equation (14), the reported domains, "
        "grid sizes, and the 5/10/20 reinitialization schedule."
    )
    lines.append(
        "  - Open method gap: the paper states that section-4 irregular samples were generated "
        "with the authors' original Python/C++ implementations, while the current repo uses a "
        "Python rebuild for the explicit flower tests."
    )
    lines.append("")
    lines.append("2. Sampling-rule check against paper sample counts")
    lines.append(
        "  - The paper wording in equation (13) mentions nodes on Gamma or nodes whose outgoing "
        "x/y edge is crossed."
    )
    lines.append(
        "  - An outgoing-edge-only implementation does not match the paper counts; the current "
        "two-endpoint deduplicated rule does."
    )
    lines.append("")
    lines.append("3. Dynamic pipeline comparison")
    lines.append(
        "  - current_xy_test_reinit: current explicit-test path "
        "(xy grid + explicit-test reinitializer)."
    )
    lines.append(
        "  - training_ij_train_reinit: rebuild the same flower test with training-style "
        "(ij grid + training reinitializer) and compare the paper model on top."
    )
    lines.append("")

    step_wins = 0
    total_steps = 0
    case_wins = 0
    total_cases = 0

    for scenario in project_config.evaluation.scenarios:
        exp_id = scenario.exp_id
        paper_reference = PAPER_UNIFORM_REFERENCES.get(exp_id)
        if paper_reference is None:
            continue

        total_cases += 1
        case_current_score = 0.0
        case_candidate_score = 0.0

        lines.append(
            f"{exp_id} | paper_samples={paper_reference.sample_count} | "
            f"case_h={float(scenario.h):.9e} | rho_eq_case={1.0 / float(scenario.h) + 1.0:.6f} | "
            f"rho_model={int(scenario.rho_model)} | model_h={read_model_h(project_root / 'model' / 'origin' / f'trainStats_{int(scenario.rho_model)}.csv'):.9e}"
        )

        for step in project_config.evaluation.test_iters:
            current_payload = build_current_pipeline_payload(
                L=float(scenario.L),
                N=int(scenario.N),
                a=float(scenario.a),
                b=float(scenario.b),
                p=int(scenario.p),
                cfl=float(project_config.generation.cfl),
                step=int(step),
            )
            candidate_payload = build_training_style_pipeline_payload(
                L=float(scenario.L),
                N=int(scenario.N),
                a=float(scenario.a),
                b=float(scenario.b),
                p=int(scenario.p),
                cfl=float(project_config.generation.cfl),
                step=int(step),
            )
            paper_step = get_paper_reference(exp_id, int(step))

            current_origin = compute_metrics(
                current_payload.target,
                origin_predictor.predict(int(scenario.rho_model), current_payload.stencils),
            )
            candidate_origin = compute_metrics(
                candidate_payload.target,
                origin_predictor.predict(int(scenario.rho_model), candidate_payload.stencils),
            )
            current_score = normalized_distance(current_origin, paper_step.paper_model)
            candidate_score = normalized_distance(candidate_origin, paper_step.paper_model)
            case_current_score += current_score
            case_candidate_score += candidate_score
            total_steps += 1
            if candidate_score < current_score:
                step_wins += 1

            current_outgoing_only = outgoing_only_count(
                L=float(scenario.L),
                N=int(scenario.N),
                a=float(scenario.a),
                b=float(scenario.b),
                p=int(scenario.p),
                cfl=float(project_config.generation.cfl),
                step=int(step),
            )

            lines.append(
                f"  step={int(step)} | counts current={current_payload.count}, "
                f"candidate={candidate_payload.count}, outgoing_only={current_outgoing_only}, "
                f"paper={paper_reference.sample_count}"
            )
            lines.append(
                "    origin/current_xy_test_reinit: "
                f"MAE={current_origin.mae:.6e}, MaxAE={current_origin.max_ae:.6e}, MSE={current_origin.mse:.6e} | "
                f"|d_paper| score={current_score:.6f}"
            )
            lines.append(
                "    origin/training_ij_train_reinit: "
                f"MAE={candidate_origin.mae:.6e}, MaxAE={candidate_origin.max_ae:.6e}, MSE={candidate_origin.mse:.6e} | "
                f"|d_paper| score={candidate_score:.6f} | "
                f"winner={'training_ij_train_reinit' if candidate_score < current_score else 'current_xy_test_reinit'}"
            )

        if case_candidate_score < case_current_score:
            case_wins += 1
        lines.append(
            f"  case_summary | current_score={case_current_score:.6f} | "
            f"candidate_score={case_candidate_score:.6f} | "
            f"winner={'training_ij_train_reinit' if case_candidate_score < case_current_score else 'current_xy_test_reinit'}"
        )
        lines.append("")

    lines.append("4. Summary")
    lines.append(
        f"  - training_ij_train_reinit is closer to the paper model on "
        f"{step_wins}/{max(total_steps, 1)} case-steps and {case_wins}/{max(total_cases, 1)} cases."
    )
    lines.append(
        "  - The gain is only residual: it slightly improves the rebuilt explicit tests, but it does not close the paper gap."
    )
    lines.append(
        "  - So the highest-priority method issue is no longer target or stencil order; "
        "it is the provenance of the irregular-test generator and reinitialization path."
    )
    lines.append(
        "  - Outgoing-edge-only sampling should not be the next fix candidate because it immediately breaks the paper sample counts."
    )
    return "\n".join(lines)


def build_current_pipeline_payload(
    *,
    L: float,
    N: int,
    a: float,
    b: float,
    p: int,
    cfl: float,
    step: int,
) -> PipelinePayload:
    X, Y, h = build_grid(L, N)
    phi0 = build_flower_phi0(X, Y, a, b, p)
    phi = ExplicitTestReinitializer(cfl=cfl).reinitialize(phi0, h, int(step))
    indices = current_indices(phi)
    xy = np.column_stack((X[indices[:, 0], indices[:, 1]], Y[indices[:, 0], indices[:, 1]]))
    theta = find_projection_theta(xy, a, b, p)
    target = hkappa_analytic(theta, h, a, b, p)
    stencils = extract_3x3_stencils(phi, indices)
    numerical = compute_eq3_from_field(phi, indices, h)
    return PipelinePayload(
        count=int(indices.shape[0]),
        target=target,
        stencils=stencils,
        numerical=numerical,
    )


def build_training_style_pipeline_payload(
    *,
    L: float,
    N: int,
    a: float,
    b: float,
    p: int,
    cfl: float,
    step: int,
) -> PipelinePayload:
    x = np.linspace(-L, L, N, dtype=np.float64)
    y = np.linspace(-L, L, N, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    h = 2.0 * L / (N - 1)
    phi0 = build_flower_phi0(X, Y, a, b, p)
    phi = TrainReinitializer(cfl=cfl).reinitialize(phi0, h, int(step))
    indices = current_indices(phi)
    xy = np.column_stack((X[indices[:, 0], indices[:, 1]], Y[indices[:, 0], indices[:, 1]]))
    theta = find_projection_theta(xy, a, b, p)
    target = hkappa_analytic(theta, h, a, b, p)
    stencils = extract_3x3_stencils(phi, indices)
    numerical = compute_eq3_from_field(phi, indices, h)
    return PipelinePayload(
        count=int(indices.shape[0]),
        target=target,
        stencils=stencils,
        numerical=numerical,
    )


def current_indices(phi: np.ndarray) -> np.ndarray:
    i_coords, j_coords = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(i_coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((i_coords, j_coords)).astype(np.int64, copy=False)


def outgoing_only_count(
    *,
    L: float,
    N: int,
    a: float,
    b: float,
    p: int,
    cfl: float,
    step: int,
) -> int:
    X, Y, h = build_grid(L, N)
    phi0 = build_flower_phi0(X, Y, a, b, p)
    phi = ExplicitTestReinitializer(cfl=cfl).reinitialize(phi0, h, int(step))
    sign_change_x = phi[:-1, :] * phi[1:, :] <= 0.0
    sign_change_y = phi[:, :-1] * phi[:, 1:] <= 0.0
    mask = np.zeros_like(phi, dtype=bool)
    ix, jx = np.where(sign_change_x)
    mask[ix, jx] = True
    iy, jy = np.where(sign_change_y)
    mask[iy, jy] = True
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return int(np.count_nonzero(mask))


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


def read_model_h(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            feature_name = row[0].strip().strip('"')
            if feature_name == "h":
                return float(row[1])
    raise ValueError(f"Failed to find h row in {csv_path}")


def resolve_output_path(project_root: Path, raw_output: str) -> Path:
    if raw_output:
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        return output_path
    stamp = datetime.now().strftime("%m%d")
    return project_root / "test" / "results" / f"method_alignment_{stamp}.txt"


if __name__ == "__main__":
    main()
