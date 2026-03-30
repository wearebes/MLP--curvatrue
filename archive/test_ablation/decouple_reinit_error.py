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
from test.explicit_sdf_upper_bound import (
    build_exact_signed_distance_field,
    current_indices,
    patch_node_mask,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="3-way decoupling of Phi0 vs Reinit Errors")
    parser.add_argument("--config", type=str, default=str(DEFAULT_PROJECT_CONFIG_PATH))
    args = parser.parse_args()

    project_config = load_project_config(args.config)
    
    # We will pick 2 distinct representative cases
    scenarios_to_test = []
    for s in project_config.evaluation.scenarios:
        if s.exp_id in ["smooth_256", "acute_276"]:
            scenarios_to_test.append(s)
            
    if not scenarios_to_test:
        scenarios_to_test = project_config.evaluation.scenarios[:2]

    # Setup saving directory
    out_dir = project_root / "test" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "decouple_reinit_error_0325.txt"

    lines = []
    lines.append("=== PDE Source Error Isolation Report ===")
    lines.append("Testing Pre-reinit, Post-reinit, and Exact-reinit Deviations from True Signed-Distance Manifold.")
    lines.append("")

    for scenario in scenarios_to_test:
        exp_id = scenario.exp_id
        L = float(scenario.L)
        N = int(scenario.N)
        a = float(scenario.a)
        b = float(scenario.b)
        p = int(scenario.p)
        
        # Test steps
        steps = [20]
        if hasattr(project_config.evaluation, 'test_iters'):
            steps = project_config.evaluation.test_iters

        lines.append(f"----- Scenario: {exp_id} (a={a}, b={b}, p={p}) -----")
        
        # We MUST use our fixed build_grid_ij to make indexing proper
        X, Y, h = build_grid_ij(L, N)
        phi0 = build_flower_phi0(X, Y, a, b, p)
        reinitializer = LevelSetReinitializer(cfl=float(project_config.generation.cfl))

        for step in steps:
            # 1. Standard pipeline: phi0 -> reinit
            phi_standard = reinitializer.reinitialize(phi0, h, int(step))
            
            # Identify valid patch target nodes where zero crossing happens
            indices = current_indices(phi_standard)
            mask = patch_node_mask(phi_standard.shape, indices)
            patch_count = np.sum(mask)

            if patch_count == 0:
                lines.append(f"  Step {step}: No zero crossings found!")
                continue
                
            # Compute Exact SDF
            exact_phi = build_exact_signed_distance_field(
                phi_shape=phi_standard.shape,
                phi0=phi0,
                X=X,
                Y=Y,
                indices=indices,
                a=a,
                b=b,
                p=p,
            )

            # 2. Reinit exactly from the EXACT SDF
            phi_exact_reinit = reinitializer.reinitialize(exact_phi, h, int(step))

            # --- Evaluate 3-way Gaps on the 3x3 patches ---
            # 1. Gap_phi0: How far is raw analytic phi0 from exact SDF?
            gap_phi0 = np.abs(phi0[mask] - exact_phi[mask])
            phi0_mae = gap_phi0.mean()
            phi0_max = gap_phi0.max()

            # 2. Gap_standard: How far is standard reinit from exact SDF?
            gap_standard = np.abs(phi_standard[mask] - exact_phi[mask])
            std_mae = gap_standard.mean()
            std_max = gap_standard.max()

            # 3. Gap_exact_reinit: How far does reinit distort an already PERFECT SDF?
            gap_exact_reinit = np.abs(phi_exact_reinit[mask] - exact_phi[mask])
            exact_mae = gap_exact_reinit.mean()
            exact_max = gap_exact_reinit.max()

            lines.append(f"[{exp_id} | iter {int(step)} | {int(patch_count)} patch nodes]")
            lines.append(f"  1. Gap(phi0)          -> MAE: {phi0_mae:.8f}, MaxAE: {phi0_max:.8f}")
            lines.append(f"  2. Gap(standard)      -> MAE: {std_mae:.8f}, MaxAE: {std_max:.8f}")
            lines.append(f"  3. Gap(exact_reinit)  -> MAE: {exact_mae:.8f}, MaxAE: {exact_max:.8f}")
            
            # Logical diagnosis based on evidence
            if std_mae < phi0_mae:
                msg = "Diagnosis: Reinit is actually FIXING the phi0 error by pulling it closer."
            else:
                msg = "Diagnosis: Reinit makes the raw phi0 WORSE (or fails to fix severe distortions)."
                
            if exact_mae > 1e-4:
                msg += f" + CRITICAL WARNING: Reinit actively distorts EXACT SDFs (Numerical viscosity/scheme error!)"
                
            lines.append(f"  > {msg}")
            lines.append("")

    report_text = "\n".join(lines)
    out_file.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[Saved to {out_file}]")

if __name__ == "__main__":
    main()
