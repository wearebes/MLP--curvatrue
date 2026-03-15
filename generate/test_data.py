from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

from .config import CFL
from .test_blueprints import TEST_BLUEPRINTS


TEST_ITERS = [5, 10, 20]


class LevelSetReinitializer:
    """Exact reinitializer ported from the original test notebook."""

    def __init__(self, cfl: float = 0.5, eps_weno: float = 1e-6, eps_sign_factor: float = 1.0):
        self.cfl = cfl
        self.eps_weno = eps_weno
        self.eps_sign_factor = eps_sign_factor

    def _smoothed_sign(self, phi0: np.ndarray, h: float) -> np.ndarray:
        eps = self.eps_sign_factor * h
        return phi0 / np.sqrt(phi0**2 + eps**2)

    def _hj_weno5_1d(self, v1, v2, v3, v4, v5) -> np.ndarray:
        beta0 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + (1.0 / 4.0) * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
        beta1 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + (1.0 / 4.0) * (v2 - v4) ** 2
        beta2 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + (1.0 / 4.0) * (3.0 * v3 - 4.0 * v4 + v5) ** 2

        alpha0 = 0.1 / (beta0 + self.eps_weno) ** 2
        alpha1 = 0.6 / (beta1 + self.eps_weno) ** 2
        alpha2 = 0.3 / (beta2 + self.eps_weno) ** 2
        sum_alpha = alpha0 + alpha1 + alpha2

        w0, w1, w2 = alpha0 / sum_alpha, alpha1 / sum_alpha, alpha2 / sum_alpha

        p0 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p1 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p2 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5

        return w0 * p0 + w1 * p1 + w2 * p2

    def _get_derivatives_weno5(self, phi: np.ndarray, h: float):
        nx, ny = phi.shape
        phi_pad = np.pad(phi, pad_width=3, mode="edge")

        d_x = (phi_pad[:, 1:] - phi_pad[:, :-1]) / h
        d_y = (phi_pad[1:, :] - phi_pad[:-1, :]) / h

        dx_m = self._hj_weno5_1d(
            d_x[3:-3, 0:ny],
            d_x[3:-3, 1:ny + 1],
            d_x[3:-3, 2:ny + 2],
            d_x[3:-3, 3:ny + 3],
            d_x[3:-3, 4:ny + 4],
        )
        dx_p = self._hj_weno5_1d(
            d_x[3:-3, 5:ny + 5],
            d_x[3:-3, 4:ny + 4],
            d_x[3:-3, 3:ny + 3],
            d_x[3:-3, 2:ny + 2],
            d_x[3:-3, 1:ny + 1],
        )

        dy_m = self._hj_weno5_1d(
            d_y[0:nx, 3:-3],
            d_y[1:nx + 1, 3:-3],
            d_y[2:nx + 2, 3:-3],
            d_y[3:nx + 3, 3:-3],
            d_y[4:nx + 4, 3:-3],
        )
        dy_p = self._hj_weno5_1d(
            d_y[5:nx + 5, 3:-3],
            d_y[4:nx + 4, 3:-3],
            d_y[3:nx + 3, 3:-3],
            d_y[2:nx + 2, 3:-3],
            d_y[1:nx + 1, 3:-3],
        )
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

    def _compute_rhs(self, phi, s0, h):
        dx_m, dx_p, dy_m, dy_p = self._get_derivatives_weno5(phi, h)
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


def build_grid(L: float, N: int):
    x = np.linspace(-L, L, N, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    h = 2.0 * L / (N - 1)
    return X, Y, h


def build_flower_phi0(X, Y, a, b, p):
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    return r - a * np.cos(p * theta) - b


def interface_band_mask(phi: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(phi, dtype=bool)
    mask[:, :-1] |= phi[:, :-1] * phi[:, 1:] <= 0.0
    mask[:-1, :] |= phi[:-1, :] * phi[1:, :] <= 0.0
    mask[0, :] = mask[-1, :] = False
    mask[:, 0] = mask[:, -1] = False
    return mask


def get_valid_indices(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def indices_to_xy(indices: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.column_stack((X[indices[:, 0], indices[:, 1]], Y[indices[:, 0], indices[:, 1]]))


def extract_3x3_stencils(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    stencils = []
    for row, col in indices:
        patch = phi[row - 1:row + 2, col - 1:col + 2]
        stencils.append(patch.flatten())
    return np.array(stencils, dtype=np.float64)


def find_projection_theta(xy: np.ndarray, a: float, b: float, p: float) -> np.ndarray:
    def dist_sq(theta, x, y):
        r = b + a * np.cos(p * theta)
        cx, cy = r * np.cos(theta), r * np.sin(theta)
        return (x - cx) ** 2 + (y - cy) ** 2

    theta_proj = []
    for x, y in xy:
        theta0 = np.arctan2(y, x)
        result = minimize(dist_sq, theta0, args=(x, y))
        theta_proj.append(result.x[0] % (2 * np.pi))
    return np.array(theta_proj)


def hkappa_analytic(theta_proj: np.ndarray, h: float, a: float, b: float, p: float) -> np.ndarray:
    r = b + a * np.cos(p * theta_proj)
    rp = -a * p * np.sin(p * theta_proj)
    rpp = -a * p**2 * np.cos(p * theta_proj)
    kappa = (r**2 + 2 * rp**2 - r * rpp) / (r**2 + rp**2) ** 1.5
    return h * kappa


def hkappa_div_normal_from_field(
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
def save_meta(exp_dir: Path, cfg: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(cfg)
    payload["rho_eq"] = 1.0 / cfg["h"] + 1.0
    payload["Omega"] = f"[-{cfg['L']}, {cfg['L']}]^2"
    with (exp_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def save_iter_h5(exp_dir: Path, n_iter: int, payload: dict) -> None:
    filepath = exp_dir / f"iter_{n_iter}.h5"
    with h5py.File(filepath, "w") as handle:
        for key, value in payload.items():
            handle.create_dataset(key, data=value, compression="gzip")


def generate_test_datasets(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reinitializer = LevelSetReinitializer(cfl=CFL)

    print("Launching Test Data Generation Pipeline")
    for cfg in TEST_BLUEPRINTS:
        exp_dir = output_dir / cfg["exp_id"]
        print(f"\nStarting experiment: {cfg['exp_id']} ...")

        X, Y, h = build_grid(cfg["L"], cfg["N"])
        phi0 = build_flower_phi0(X, Y, cfg["a"], cfg["b"], cfg["p"])

        mask_fixed = interface_band_mask(phi0)
        fixed_indices = get_valid_indices(mask_fixed)
        xy = indices_to_xy(fixed_indices, X, Y)

        theta_proj = find_projection_theta(xy, cfg["a"], cfg["b"], cfg["p"])
        hkappa_target = hkappa_analytic(theta_proj, h, cfg["a"], cfg["b"], cfg["p"])

        save_meta(exp_dir, cfg)

        for n_iter in TEST_ITERS:
            phi = reinitializer.reinitialize(phi0, h, n_iter)
            stencils_raw = extract_3x3_stencils(phi, fixed_indices)
            hkappa_fd = hkappa_div_normal_from_field(phi, fixed_indices, h)

            payload = {
                "indices": fixed_indices,
                "xy": xy,
                "theta_proj": theta_proj,
                "stencils_raw": stencils_raw,
                "hkappa_target": hkappa_target,
                "hkappa_fd": hkappa_fd,
            }
            save_iter_h5(exp_dir, n_iter, payload)

            sample_count = stencils_raw.shape[0]
            print(f"  [{cfg['exp_id']} | Iter {n_iter}] M={sample_count} points.")
            print(f"    Target hk min/max : ({hkappa_target.min():.4f}, {hkappa_target.max():.4f})")
            print(f"    FD     hk min/max : ({hkappa_fd.min():.4f}, {hkappa_fd.max():.4f})")

            if np.isnan(hkappa_target).any():
                raise ValueError(f"Target contains NaN for {cfg['exp_id']} iter={n_iter}")
            if np.isnan(hkappa_fd).any():
                raise ValueError(f"Numerical hk contains NaN for {cfg['exp_id']} iter={n_iter}")

    print("\nExplicit test data generation finished.")
    print(f"Saved test experiments under: {output_dir}")
