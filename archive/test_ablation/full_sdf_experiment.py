import sys
import argparse
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from project_config import load_project_config, DEFAULT_PROJECT_CONFIG_PATH
from generate.dataset_test import (
    LevelSetReinitializer,
    build_flower_phi0,
    build_grid_ij,
)
from test.paper_alignment import OriginPredictorLite, compute_metrics, load_origin_stats
from test.explicit_sdf_upper_bound import patch_node_mask, current_indices
from generate.stencil_encoding import extract_3x3_stencils


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


def vectorized_exact_sdf(X: np.ndarray, Y: np.ndarray, a: float, b: float, p: int) -> np.ndarray:
    tol = 1e-11
    max_iter = 80
    two_pi = 2.0 * np.pi

    shape = X.shape
    x = X.reshape(-1)
    y = Y.reshape(-1)
    
    # 1. Newton projection
    theta = np.arctan2(y, x) % two_pi

    for _ in range(max_iter):
        ct = np.cos(theta)
        st = np.sin(theta)
        cpt = np.cos(p * theta)
        spt = np.sin(p * theta)

        r = b + a * cpt
        rp = -a * p * spt
        rpp = -a * p**2 * cpt

        cx = r * ct
        cy = r * st
        cpx = rp * ct - r * st
        cpy = rp * st + r * ct
        cppx = rpp * ct - 2.0 * rp * st - r * ct
        cppy = rpp * st + 2.0 * rp * ct - r * st

        dx = cx - x
        dy = cy - y

        f = dx * cpx + dy * cpy
        fp = cpx * cpx + cpy * cpy + dx * cppx + dy * cppy
        
        active = (np.abs(f) > tol) & (np.abs(fp) >= 1e-18)
        if not np.any(active):
            break

        step = np.zeros_like(theta)
        step[active] = f[active] / fp[active]

        theta_new = (theta - step) % two_pi
        theta[active] = theta_new[active]

    # 2. Distance to curve
    r_curve = b + a * np.cos(p * theta)
    cx = r_curve * np.cos(theta)
    cy = r_curve * np.sin(theta)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # 3. Sign assignment (from continuous function)
    theta_polar = np.arctan2(y, x)
    r_polar = np.sqrt(x**2 + y**2)
    phi0 = r_polar - a * np.cos(p * theta_polar) - b
    sign = np.sign(phi0)
    
    exact_phi = (sign * dist).reshape(shape)
    return exact_phi


def get_paper_ref(exp_id: str, it: int) -> tuple[float, float, float, float]:
    # Returns (origin_mse, origin_mae, num_mse, num_mae) from the paper table roughly 
    # as extracted from the reference JSONs
    # Hardcoded based on paper Tables for iterations=20
    if it == 20:
        if exp_id == "smooth_256": return (0.00000037, 0.000305, 0.000334, 0.003180) # Origin / Num
        if exp_id == "smooth_266": return (0.00000039, 0.000313, 0.000335, 0.003180)
        if exp_id == "smooth_276": return (0.00000057, 0.000346, 0.000336, 0.003190)
        
        if exp_id == "acute_256": return (0.00001090, 0.001920, 0.000333, 0.003260)
        if exp_id == "acute_266": return (0.00001070, 0.002010, 0.000335, 0.003290)
        if exp_id == "acute_276": return (0.00001220, 0.002160, 0.000338, 0.003320)
    return (float('nan'), float('nan'), float('nan'), float('nan'))


def hkappa_analytic_formula(theta, h, a, b, p):
    r = b + a * np.cos(p * theta)
    rp = -a * p * np.sin(p * theta)
    rpp = -a * p**2 * np.cos(p * theta)
    kappa = (r**2 + 2 * rp**2 - r * rpp) / ((r**2 + rp**2) ** 1.5)
    return h * kappa


def main() -> None:
    parser = argparse.ArgumentParser("Ablation of Full-Grid Exact SDF Reinitialization")
    parser.add_argument("--config", type=str, default=str(DEFAULT_PROJECT_CONFIG_PATH))
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    predictor = OriginPredictorLite(project_root)
    
    # We will pick 2 distinct representative cases
    scenarios_to_test = []
    for s in project_config.evaluation.scenarios:
        if s.exp_id in ["smooth_256", "acute_276"]:
            scenarios_to_test.append(s)

    out_dir = project_root / "test" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "full_grid_sdf_ablation_0325.txt"

    lines = []
    lines.append("=== Ablation of Full-Grid Exact SDF Reinitialization ===")
    lines.append("Comparing purely discrete phi0 vs full-grid explicit SDF passed through identical PDE flows.")
    lines.append("")

    for scenario in scenarios_to_test:
        exp_id = scenario.exp_id
        L = float(scenario.L)
        N = int(scenario.N)
        a = float(scenario.a)
        b = float(scenario.b)
        p = int(scenario.p)
        rho_model = int(scenario.rho_model)
        
        mu, sigma = load_origin_stats(project_root / "model" / "origin" / f"trainStats_{rho_model}.csv")

        lines.append(f"----- Scenario: {exp_id} -----")
        
        X, Y, h = build_grid_ij(L, N)
        phi0_formula = build_flower_phi0(X, Y, a, b, p)
        phi0_exact = vectorized_exact_sdf(X, Y, a, b, p)

        reinitializer = LevelSetReinitializer(cfl=float(project_config.generation.cfl))

        # We will only look at stat after 20 steps
        step = 20
        paper_mse, paper_mae, paper_num_mse, paper_num_mae = get_paper_ref(exp_id, step)

        # 1. Chain A: Formula
        phi_A = reinitializer.reinitialize(phi0_formula, h, step)
        indices_A = current_indices(phi_A)
        stencils_A = extract_3x3_stencils(phi_A, indices_A)
        xy_A = np.column_stack((X[indices_A[:, 0], indices_A[:, 1]], Y[indices_A[:, 0], indices_A[:, 1]]))
        theta_A = np.arctan2(xy_A[:, 1], xy_A[:, 0])  # Note: should ideally be projection, but simple theta for target is ok for quick benchmark or we just use analytic. Let's do exact projection for target robustness.
        # Wait, if we use find_projection_theta for target... Let's just use vectorized Newton on indices
        theta_proj_A = np.arctan2(xy_A[:, 1], xy_A[:, 0]) # approximate for now, will refine
        from generate.dataset_test import find_projection_theta
        theta_proj_A = find_projection_theta(xy_A, a, b, p)
        target_A = hkappa_analytic_formula(theta_proj_A, h, a, b, p)
        
        pred_num_A = compute_eq3_from_field(phi_A, indices_A, h)
        pred_orig_A = predictor.predict(rho_model, stencils_A)

        met_num_A = compute_metrics(target_A, pred_num_A)
        met_orig_A = compute_metrics(target_A, pred_orig_A)
        count_A = indices_A.shape[0]

        # 2. Chain B: Exact SDF
        phi_B = reinitializer.reinitialize(phi0_exact, h, step)
        indices_B = current_indices(phi_B)
        stencils_B = extract_3x3_stencils(phi_B, indices_B)
        xy_B = np.column_stack((X[indices_B[:, 0], indices_B[:, 1]], Y[indices_B[:, 0], indices_B[:, 1]]))
        theta_proj_B = find_projection_theta(xy_B, a, b, p)
        target_B = hkappa_analytic_formula(theta_proj_B, h, a, b, p)
        
        pred_num_B = compute_eq3_from_field(phi_B, indices_B, h)
        pred_orig_B = predictor.predict(rho_model, stencils_B)

        met_num_B = compute_metrics(target_B, pred_num_B)
        met_orig_B = compute_metrics(target_B, pred_orig_B)
        count_B = indices_B.shape[0]

        lines.append(f"  [Paper Source]:    Origin_MSE={paper_mse:.8f}, Num_MSE={paper_num_mse:.8f}")
        lines.append(f"  [Chain A Formula]: Origin_MSE={met_orig_A.mse:.8f}, Num_MSE={met_num_A.mse:.8f}  (N={count_A})")
        lines.append(f"  [Chain B Exact]:   Origin_MSE={met_orig_B.mse:.8f}, Num_MSE={met_num_B.mse:.8f}  (N={count_B})")
        lines.append("")

    report_text = "\n".join(lines)
    out_file.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[Saved to {out_file}]")

if __name__ == "__main__":
    main()
