import os
import h5py
import numpy as np


def validate_curvature_dataset(h5_filepath: str, rho: int = 256):
    print(f"\nStarting comprehensive dataset validation: {h5_filepath}  (rho={rho})")

    h = 1.0 / (rho - 1)
    passed = 0

    try:
        with h5py.File(h5_filepath, 'r') as f:
            X = f['X'][:]
            Y = f['Y'][:]
            steps = f['reinit_steps'][:]
            blueprint_idx = f['blueprint_idx'][:]
            radius_idx = f['radius_idx'][:]

            total_samples = X.shape[0]
            print(f" [1/6] Basic dimension check: {total_samples:,} samples loaded.")
            assert X.shape[1] == 9, "Feature dimension must be 9 (3x3 stencil)"
            assert Y.shape[1] == 1, "Label dimension must be 1"
            assert steps.shape[1] == 1
            assert blueprint_idx.shape[1] == 1
            assert radius_idx.shape[1] == 1
            passed += 1

            # --- Check 2: NaN and Inf ---
            print(" [2/6] Data purity check (NaN / Inf)...")
            assert not np.isnan(X).any(), "X contains NaN"
            assert not np.isnan(Y).any(), "Y contains NaN"
            assert not np.isinf(X).any(), "X contains Inf"
            assert not np.isinf(Y).any(), "Y contains Inf"
            passed += 1

            # --- Check 3: Metadata integrity ---
            print(" [3/6] Metadata integrity check...")
            unique_steps   = np.unique(steps[:, 0])
            unique_radius  = np.unique(radius_idx[:, 0])
            unique_bp      = np.unique(blueprint_idx[:, 0])
            print(f"    reinit_steps unique   : {unique_steps.tolist()}")
            print(f"    radius_idx range      : [{int(unique_radius.min())}, {int(unique_radius.max())}]  (unique={unique_radius.size})")
            print(f"    blueprint_idx range   : [{int(unique_bp.min())}, {int(unique_bp.max())}]  (unique={unique_bp.size})")
            assert np.all(blueprint_idx[:, 0] >= 0), (
                "blueprint_idx contains negative values.  "
                "Old HDF5 files generated before the int32-fix will show -1 here; "
                "please regenerate with generate_full_dataset_for_resolution()."
            )
            assert np.all(radius_idx[:, 0] >= 0), "radius_idx contains negative values."
            passed += 1

            # --- Check 4: Augmentation symmetry ---
            print(" [4/6] Data augmentation symmetry check...")
            x_mean = np.abs(np.mean(X))
            y_mean = np.abs(np.mean(Y))
            assert x_mean < 1e-6, f"Symmetry failed: |mean(X)| = {x_mean:.2e} (should be ~0)"
            assert y_mean < 1e-6, f"Symmetry failed: |mean(Y)| = {y_mean:.2e} (should be ~0)"
            passed += 1

            # --- Check 5: Label theoretical bounds ---
            # Augmentation adds (-X, -Y) pairs, so Y spans [-hk_max, -hk_min] U [hk_min, hk_max].
            # Negative Y is EXPECTED and CORRECT; use |Y| for range comparison.
            print(" [5/6] Curvature label theoretical bound check...")
            r_min  = 1.6 * h
            r_max  = 0.5 - 2.0 * h
            hk_max = float(h / r_min)   # = h / r_min  (largest |h*kappa|, smallest circle)
            hk_min = float(h / r_max)   # = h / r_max  (smallest |h*kappa|, largest circle)

            y_signed_min = float(np.min(Y))
            y_signed_max = float(np.max(Y))
            y_abs        = np.abs(Y)
            y_abs_min    = float(np.min(y_abs))
            y_abs_max    = float(np.max(y_abs))

            print(f"    Theoretical |h*kappa| range  : [{hk_min:.6f}, {hk_max:.6f}]")
            print(f"    Expected signed Y (post-aug)  : [{-hk_max:.6f}, {-hk_min:.6f}] \u222a [{hk_min:.6f}, {hk_max:.6f}]")
            print(f"    Actual   signed Y             : [{y_signed_min:.6f}, {y_signed_max:.6f}]")
            print(f"    Actual   |Y|                  : [{y_abs_min:.6f}, {y_abs_max:.6f}]")
            if y_signed_min < 0:
                print(f"    Note: min(Y) = {y_signed_min:.6f} < 0 is EXPECTED \u2014 augmentation mirror, not a defect.")

            assert y_abs_max <= hk_max * 1.01, (
                f"Physical impossibility: max|h*kappa| = {y_abs_max:.6f} exceeds "
                f"theoretical ceiling {hk_max:.6f} by >1%.  Check CircleGeometryGenerator."
            )
            assert y_abs_min >= hk_min * 0.99, (
                f"Physical impossibility: min|h*kappa| = {y_abs_min:.6f} undershoots "
                f"theoretical floor {hk_min:.6f} by >1%.  Check CircleGeometryGenerator."
            )
            passed += 1

            # --- Check 6: Eikonal |grad phi| approx 1 on SDF samples ---
            print(" [6/6] Eikonal physical-law check (|grad phi| \u2248 1 on SDF samples)...")
            sdf_mask = (steps[:, 0] == 0)
            X_sdf    = X[sdf_mask]

            if len(X_sdf) > 0:
                rng_idx = np.random.choice(len(X_sdf), min(1000, len(X_sdf)), replace=False)
                sX = X_sdf[rng_idx]
                phi_x = (sX[:, 5] - sX[:, 3]) / (2.0 * h)
                phi_y = (sX[:, 1] - sX[:, 7]) / (2.0 * h)
                mean_grad = float(np.mean(np.sqrt(phi_x**2 + phi_y**2)))
                print(f"    SDF sample avg |grad phi| : {mean_grad:.6f}  (theoretical: 1.000000)")
                assert 0.90 < mean_grad < 1.10, (
                    f"Physical-law violation: mean |grad phi| = {mean_grad:.6f}"
                )
            passed += 1

    except Exception as e:
        print(f"\n  [FAIL] Check {passed+1}/6 failed: {e}")
        return

    print(f"\n{'='*55}")
    print(f"  All 6/6 checks PASSED  \u2713  {h5_filepath}  (rho={rho})")
    print(f"{'='*55}")


if __name__ == "__main__":
    for _rho in [256, 266, 276]:
        _path = f"data/train_rho{_rho}.h5"
        if os.path.exists(_path):
            validate_curvature_dataset(_path, rho=_rho)
        else:
            print(f"[Skip] {_path} not found \u2014 run generate_data.py first.")
