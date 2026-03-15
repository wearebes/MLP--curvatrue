from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from generate.config import CFL
from generate.test_data import (
    TEST_ITERS,
    LevelSetReinitializer,
    build_flower_phi0,
    build_grid,
    hkappa_div_normal_from_field,
    interface_band_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose one explicit test case with sample-wise MSE, |grad phi|-1, and curvature variants."
    )
    parser.add_argument("--exp-id", default="smooth_276", help="Experiment folder under data/, e.g. smooth_276")
    parser.add_argument("--step", type=int, default=20, help="Reinitialization step / iter to inspect")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data"),
        help="Project data directory",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    exp_dir = data_dir / args.exp_id
    meta_path = exp_dir / "meta.json"
    h5_path = exp_dir / f"iter_{args.step}.h5"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with h5py.File(h5_path, "r") as handle:
        stencils_raw = handle["stencils_raw"][:].astype(np.float64)
        target = handle["hkappa_target"][:].astype(np.float64)
        stored_fd = handle["hkappa_fd"][:].astype(np.float64)
        indices = handle["indices"][:].astype(np.int64)

    x_grid, y_grid, h_from_grid = build_grid(float(meta["L"]), int(meta["N"]))
    h_meta = float(meta["h"])
    if not np.isclose(h_meta, h_from_grid):
        raise ValueError(f"Inconsistent h in metadata ({h_meta}) and grid ({h_from_grid})")

    phi0 = build_flower_phi0(x_grid, y_grid, float(meta["a"]), float(meta["b"]), float(meta["p"]))
    phi = LevelSetReinitializer(cfl=CFL).reinitialize(phi0, h_meta, args.step)

    central_grad = central_grad_norm(phi, h_meta)
    grad_on_samples = central_grad[indices[:, 0], indices[:, 1]]
    grad_abs_err = np.abs(grad_on_samples - 1.0)

    mask = interface_band_mask(phi)
    recomputed_indices = np.argwhere(mask)

    recomputed_fd = hkappa_div_normal_from_field(phi, indices, h_meta)
    alt_fd = hkappa_fd_variant(stencils_raw, h_meta)
    phi_stats = center_derivative_stats(stencils_raw, h_meta)
    full_formula_standard = hkappa_from_full_field_standard(phi, indices, h_meta)
    full_formula_standard_delta = hkappa_from_full_field_standard(phi, indices, h_meta, delta=1e-12)
    full_formula_div_normal = hkappa_from_full_field_div_normal(phi, indices, h_meta, delta=0.0)
    full_formula_div_normal_delta = hkappa_from_full_field_div_normal(phi, indices, h_meta, delta=1e-12)
    step_rows = load_step_rows(exp_dir)

    print("=" * 88)
    print(f"Single-Case Diagnosis | exp_id={args.exp_id} | step={args.step}")
    print("=" * 88)
    print(f"Data dir               : {data_dir}")
    print(f"HDF5 path              : {h5_path}")
    print(f"Experiment type        : {meta.get('experiment_type', 'unknown')}")
    print(f"rho_model              : {meta['rho_model']}")
    print(f"Grid N                 : {meta['N']}")
    print(f"h                      : {h_meta:.9e}")
    print()

    print("h-definition checks")
    print(f"h from metadata        : {h_meta:.9e}")
    print(f"h = 2L/(N-1)           : {(2.0 * float(meta['L']) / (int(meta['N']) - 1)):.9e}")
    print(f"h = 1/(rho_model-1)    : {(1.0 / (int(meta['rho_model']) - 1)):.9e}")
    print(f"ratio meta / 1/(rho-1) : {h_meta / (1.0 / (int(meta['rho_model']) - 1)):.9e}")
    print()

    print("Sample-set checks")
    print(f"Stored sample count    : {len(indices)}")
    print(f"Recomputed mask count  : {len(recomputed_indices)}")
    print(f"Index set exact match  : {np.array_equal(indices, recomputed_indices)}")
    print()

    print("Step-to-step MSE improvement on this exact case")
    for row in step_rows:
        print(
            f"step {row['step']:>2}               "
            f"MSE={row['mse']:.9e} | MAE={row['mae']:.9e} | MaxAE={row['max_ae']:.9e}"
        )
    for prev, curr in zip(step_rows, step_rows[1:]):
        improvement = 1.0 - (curr["mse"] / prev["mse"])
        print(
            f"step {prev['step']:>2} -> {curr['step']:>2}      "
            f"relative MSE drop={improvement:.2%}"
        )
    print()

    print("Derivative / denominator sanity checks on sampled stencils")
    print(f"phi_x mean/std         : {phi_stats['phi_x_mean']:.9e} / {phi_stats['phi_x_std']:.9e}")
    print(f"phi_y mean/std         : {phi_stats['phi_y_mean']:.9e} / {phi_stats['phi_y_std']:.9e}")
    print(f"phi_xx mean/std        : {phi_stats['phi_xx_mean']:.9e} / {phi_stats['phi_xx_std']:.9e}")
    print(f"phi_yy mean/std        : {phi_stats['phi_yy_mean']:.9e} / {phi_stats['phi_yy_std']:.9e}")
    print(f"phi_xy mean/std        : {phi_stats['phi_xy_mean']:.9e} / {phi_stats['phi_xy_std']:.9e}")
    print(f"min |grad phi|         : {phi_stats['grad_min']:.9e}")
    print(f"min denominator        : {phi_stats['denominator_min']:.9e}")
    print(f"count denom <= 1e-12   : {phi_stats['denominator_small_count']}")
    print()

    print("Direct sample-wise metrics against analytic hk")
    print_metrics("Stored hkappa_fd", target, stored_fd)
    print_metrics("Recomputed current", target, recomputed_fd)
    print_metrics("Alternative variant", target, alt_fd)
    print_metrics("Full-field standard", target, full_formula_standard)
    print_metrics("Standard + delta", target, full_formula_standard_delta)
    print_metrics("Div(normal)", target, full_formula_div_normal)
    print_metrics("Div(normal)+delta", target, full_formula_div_normal_delta)
    print()

    print("Consistency checks")
    print(f"Stored vs recomputed max abs diff : {np.max(np.abs(stored_fd - recomputed_fd)):.9e}")
    print(f"Stored vs recomputed MSE diff     : {sample_mse(target, stored_fd) - sample_mse(target, recomputed_fd):.9e}")
    print(f"Current vs alt variant MSE diff   : {sample_mse(target, recomputed_fd) - sample_mse(target, alt_fd):.9e}")
    print(f"Standard vs div-normal MSE diff   : {sample_mse(target, full_formula_standard) - sample_mse(target, full_formula_div_normal):.9e}")
    print()

    print("Why weighted sample-wise MSE matters")
    squared_error = (stored_fd - target) ** 2
    print(f"Sample-wise MSE        : {squared_error.mean():.9e}")
    print(f"Mean over 8 equal bins : {equal_bin_average(squared_error, bin_count=8):.9e}")
    print("Note                   : the paper-style metric should use the sample-wise MSE above.")
    print()

    print("Target / numerical hk distribution on sampled nodes")
    print_distribution("Target hk", target)
    print_distribution("Stored hkappa_fd", stored_fd)
    print_distribution("Div(normal)", full_formula_div_normal)
    print()

    print("|grad phi| diagnostics on sampled interface-adjacent nodes")
    print(f"Mean |grad phi|        : {grad_on_samples.mean():.9e}")
    print(f"Std  |grad phi|        : {grad_on_samples.std():.9e}")
    print(f"Mean ||grad|-1|        : {grad_abs_err.mean():.9e}")
    print(f"Max  ||grad|-1|        : {grad_abs_err.max():.9e}")
    for quantile in [0.5, 0.9, 0.99, 1.0]:
        print(f"Quantile {quantile:>4.2f}         : {np.quantile(grad_abs_err, quantile):.9e}")


def central_grad_norm(phi: np.ndarray, h: float) -> np.ndarray:
    phi_x = np.zeros_like(phi, dtype=np.float64)
    phi_y = np.zeros_like(phi, dtype=np.float64)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)
    return np.sqrt(phi_x**2 + phi_y**2)


def load_step_rows(exp_dir: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for step in TEST_ITERS:
        h5_path = exp_dir / f"iter_{step}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as handle:
            target = handle["hkappa_target"][:].astype(np.float64)
            pred = handle["hkappa_fd"][:].astype(np.float64)
        rows.append(
            {
                "step": step,
                "mse": sample_mse(target, pred),
                "mae": sample_mae(target, pred),
                "max_ae": sample_max_ae(target, pred),
            }
        )
    return rows


def center_derivative_stats(stencils_raw: np.ndarray, h: float) -> dict[str, float | int]:
    phi_x = (stencils_raw[:, 5] - stencils_raw[:, 3]) / (2.0 * h)
    phi_y = (stencils_raw[:, 7] - stencils_raw[:, 1]) / (2.0 * h)
    phi_xx = (stencils_raw[:, 5] - 2.0 * stencils_raw[:, 4] + stencils_raw[:, 3]) / (h**2)
    phi_yy = (stencils_raw[:, 7] - 2.0 * stencils_raw[:, 4] + stencils_raw[:, 1]) / (h**2)
    phi_xy = (stencils_raw[:, 8] - stencils_raw[:, 6] - stencils_raw[:, 2] + stencils_raw[:, 0]) / (4.0 * h**2)
    grad = np.sqrt(phi_x**2 + phi_y**2)
    denominator = (phi_x**2 + phi_y**2) ** 1.5
    return {
        "phi_x_mean": float(phi_x.mean()),
        "phi_x_std": float(phi_x.std()),
        "phi_y_mean": float(phi_y.mean()),
        "phi_y_std": float(phi_y.std()),
        "phi_xx_mean": float(phi_xx.mean()),
        "phi_xx_std": float(phi_xx.std()),
        "phi_yy_mean": float(phi_yy.mean()),
        "phi_yy_std": float(phi_yy.std()),
        "phi_xy_mean": float(phi_xy.mean()),
        "phi_xy_std": float(phi_xy.std()),
        "grad_min": float(grad.min()),
        "denominator_min": float(denominator.min()),
        "denominator_small_count": int((denominator <= 1e-12).sum()),
    }


def hkappa_from_full_field_standard(
    phi: np.ndarray,
    indices: np.ndarray,
    h: float,
    *,
    delta: float = 0.0,
) -> np.ndarray:
    rows = indices[:, 0]
    cols = indices[:, 1]
    phi_x = (phi[rows, cols + 1] - phi[rows, cols - 1]) / (2.0 * h)
    phi_y = (phi[rows + 1, cols] - phi[rows - 1, cols]) / (2.0 * h)
    phi_xx = (phi[rows, cols + 1] - 2.0 * phi[rows, cols] + phi[rows, cols - 1]) / (h**2)
    phi_yy = (phi[rows + 1, cols] - 2.0 * phi[rows, cols] + phi[rows - 1, cols]) / (h**2)
    phi_xy = (phi[rows + 1, cols + 1] - phi[rows + 1, cols - 1] - phi[rows - 1, cols + 1] + phi[rows - 1, cols - 1]) / (4.0 * h**2)

    numerator = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2 + delta) ** 1.5
    prediction = np.zeros_like(numerator)
    if delta > 0.0:
        prediction = numerator / denominator
    else:
        mask = denominator > 1e-12
        prediction[mask] = numerator[mask] / denominator[mask]
    return h * prediction


def hkappa_from_full_field_div_normal(
    phi: np.ndarray,
    indices: np.ndarray,
    h: float,
    *,
    delta: float = 0.0,
) -> np.ndarray:
    phi_x = np.zeros_like(phi, dtype=np.float64)
    phi_y = np.zeros_like(phi, dtype=np.float64)
    phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * h)
    phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * h)

    grad_norm = np.sqrt(phi_x**2 + phi_y**2 + delta)
    nx = np.divide(phi_x, grad_norm, out=np.zeros_like(phi_x), where=grad_norm > 0.0)
    ny = np.divide(phi_y, grad_norm, out=np.zeros_like(phi_y), where=grad_norm > 0.0)

    dnx_dx = np.zeros_like(phi, dtype=np.float64)
    dny_dy = np.zeros_like(phi, dtype=np.float64)
    dnx_dx[:, 1:-1] = (nx[:, 2:] - nx[:, :-2]) / (2.0 * h)
    dny_dy[1:-1, :] = (ny[2:, :] - ny[:-2, :]) / (2.0 * h)
    kappa = dnx_dx + dny_dy

    return h * kappa[indices[:, 0], indices[:, 1]]


def hkappa_fd_variant(stencils_raw: np.ndarray, h: float) -> np.ndarray:
    # Same stencil layout as explicit test generation, but with the other sign convention
    # often used when matrix-row direction is accidentally interpreted as +y.
    phi_x = (stencils_raw[:, 5] - stencils_raw[:, 3]) / (2.0 * h)
    phi_y = (stencils_raw[:, 1] - stencils_raw[:, 7]) / (2.0 * h)
    phi_xx = (stencils_raw[:, 5] - 2.0 * stencils_raw[:, 4] + stencils_raw[:, 3]) / (h**2)
    phi_yy = (stencils_raw[:, 1] - 2.0 * stencils_raw[:, 4] + stencils_raw[:, 7]) / (h**2)
    phi_xy = (stencils_raw[:, 2] - stencils_raw[:, 0] - stencils_raw[:, 8] + stencils_raw[:, 6]) / (4.0 * h**2)

    numerator = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2) ** 1.5
    prediction = np.zeros_like(numerator)
    mask = denominator > 1e-12
    prediction[mask] = numerator[mask] / denominator[mask]
    return h * prediction


def sample_mse(target: np.ndarray, pred: np.ndarray) -> float:
    error = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(error**2))


def sample_mae(target: np.ndarray, pred: np.ndarray) -> float:
    error = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(np.abs(error)))


def sample_max_ae(target: np.ndarray, pred: np.ndarray) -> float:
    error = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.max(np.abs(error)))


def equal_bin_average(values: np.ndarray, *, bin_count: int) -> float:
    bins = np.array_split(values, bin_count)
    return float(np.mean([float(bin.mean()) for bin in bins if bin.size > 0]))


def print_distribution(label: str, values: np.ndarray) -> None:
    print(
        f"{label:<22} min={values.min():.9e} | max={values.max():.9e} | "
        f"mean={values.mean():.9e} | std={values.std():.9e}"
    )


def print_metrics(label: str, target: np.ndarray, pred: np.ndarray) -> None:
    print(
        f"{label:<22} MSE={sample_mse(target, pred):.9e} | "
        f"MAE={sample_mae(target, pred):.9e} | "
        f"MaxAE={sample_max_ae(target, pred):.9e}"
    )


if __name__ == "__main__":
    main()
