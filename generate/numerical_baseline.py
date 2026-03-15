import os
import h5py
import numpy as np


def compute_numerical_hkappa(X: np.ndarray, h: float) -> np.ndarray:
    """
    Computes numerical h*kappa for each 3x3 stencil sample using a
    div(normal) discretization:

        n = grad(phi) / |grad(phi)|
        kappa = div(n)

    Stencil layout (from HDF5DatasetCompiler.extract_stencils):
        idx 0: phi[i-1, j+1]   idx 1: phi[i,   j+1]   idx 2: phi[i+1, j+1]
        idx 3: phi[i-1, j  ]   idx 4: phi[i,   j  ]   idx 5: phi[i+1, j  ]
        idx 6: phi[i-1, j-1]   idx 7: phi[i,   j-1]   idx 8: phi[i+1, j-1]

    Because only a 3x3 stencil is available, we evaluate n_x and n_y on the
    four direct neighbors of the center using one-sided differences in the
    outward direction and central differences in the transverse direction, then
    approximate div(n) at the center with centered differences.
    """
    X = X.astype(np.float64)
    center = X[:, 4]

    # Left / right neighbors: one-sided x-derivative, central y-derivative
    dx_left = (center - X[:, 3]) / h
    dy_left = (X[:, 0] - X[:, 6]) / (2.0 * h)
    nx_left = _normalized_component(dx_left, dy_left)

    dx_right = (X[:, 5] - center) / h
    dy_right = (X[:, 2] - X[:, 8]) / (2.0 * h)
    nx_right = _normalized_component(dx_right, dy_right)

    # Up / down neighbors: central x-derivative, one-sided y-derivative
    dx_up = (X[:, 2] - X[:, 0]) / (2.0 * h)
    dy_up = (X[:, 1] - center) / h
    ny_up = _normalized_component(dy_up, dx_up)

    dx_down = (X[:, 8] - X[:, 6]) / (2.0 * h)
    dy_down = (center - X[:, 7]) / h
    ny_down = _normalized_component(dy_down, dx_down)

    dnx_dx = (nx_right - nx_left) / (2.0 * h)
    dny_dy = (ny_up - ny_down) / (2.0 * h)
    return (dnx_dx + dny_dy) * h


def _normalized_component(primary: np.ndarray, transverse: np.ndarray) -> np.ndarray:
    grad_norm = np.sqrt(primary ** 2 + transverse ** 2)
    component = np.zeros_like(primary)
    safe = grad_norm > 1e-12
    component[safe] = primary[safe] / grad_norm[safe]
    return component


def evaluate_numerical_baseline(h5_filepath: str, rho: int) -> dict:
    """
    Loads a generated HDF5 dataset, computes the numerical-FD curvature estimate
    for every sample, and reports MAE / MaxAE / MSE versus the analytical ground
    truth stored in Y (= h*kappa_true).
    """
    h = 1.0 / (rho - 1)

    with h5py.File(h5_filepath, "r") as f:
        X     = f["X"][:]                  
        Y     = f["Y"][:, 0]               
        steps = f["reinit_steps"][:, 0]     

    Y_pred  = compute_numerical_hkappa(X, h)   
    Y_true  = Y.astype(np.float64)
    err     = Y_pred - Y_true
    abs_err = np.abs(err)

    mae_g   = float(np.mean(abs_err))
    maxae_g = float(np.max(abs_err))
    mse_g   = float(np.mean(err ** 2))

    print(f"\n{'='*65}")
    print(f"  Numerical Baseline Metrics  |  rho={rho},  h={h:.8f}")
    print(f"{'='*65}")
    print(f"  Total samples : {len(Y_true):,}")
    print(f"  Global MAE    : {mae_g:.6e}")
    print(f"  Global MaxAE  : {maxae_g:.6e}")
    print(f"  Global MSE    : {mse_g:.6e}")

    print()
    print(f"  {'Steps':>10}  {'N':>10}  {'MAE':>12}  {'MaxAE':>12}  {'MSE':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
    step_rows = []
    for s in sorted(np.unique(steps)):
        mask    = steps == s
        n       = int(mask.sum())
        mae_s   = float(np.mean(abs_err[mask]))
        maxae_s = float(np.max(abs_err[mask]))
        mse_s   = float(np.mean(err[mask] ** 2))
        label   = "SDF (0)" if s == 0 else f"reinit ({int(s):2d})"
        print(f"  {label:>10}  {n:>10,}  {mae_s:>12.6e}  {maxae_s:>12.6e}  {mse_s:>12.6e}")
        step_rows.append((s, n, mae_s, maxae_s, mse_s))

    return {"rho": rho, "h": h, "N": len(Y_true),
            "mae": mae_g, "max_ae": maxae_g, "mse": mse_g,
            "step_rows": step_rows}


if __name__ == "__main__":
    all_results = []
    for rho in [256, 266, 276]:
        path = f"data/train_rho{rho}.h5"
        if os.path.exists(path):
            res = evaluate_numerical_baseline(path, rho)
            all_results.append(res)
        else:
            print(f"[Skip] {path} not found — please generate data first.")

    if all_results:
        print(f"\n{'='*65}")
        print("  Cross-Resolution Summary  (Numerical FD Baseline vs h*kappa_true)")
        print(f"{'='*65}")
        print(f"  {'rho':>6}  {'N':>10}  {'MAE':>12}  {'MaxAE':>12}  {'MSE':>12}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
        for r in all_results:
            print(f"  {r['rho']:>6}  {r['N']:>10,}  {r['mae']:>12.6e}  {r['max_ae']:>12.6e}  {r['mse']:>12.6e}")
