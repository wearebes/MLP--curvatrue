"""
core/numerics.py — Unified numerical helpers for the level-set curvature pipeline.

This module centralises:

* ``build_grid``          – rectangular grid for both "ij" and "xy" conventions
* ``compute_hkappa``      – curvature × h from phi field, axis-aware
* Analytical flower tools – ``build_flower_phi0``, ``vectorized_exact_sdf``,
                            ``find_projection_theta``, ``hkappa_analytic``
* ``ReinitQualityEvaluator`` – interface-band quality metric

No torch imports. No side-effects. Pure numpy.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    import mpmath as mp
except ImportError:  # pragma: no cover
    mp = None


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(
    L: float,
    N: int,
    indexing: Literal["ij", "xy"] = "xy",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build a 2-D Cartesian grid on [-L, L]^2.

    Parameters
    ----------
    L : float
        Half-width of the domain.
    N : int
        Number of grid points per side.
    indexing : {"ij", "xy"}
        Passed directly to ``np.meshgrid``.

    Returns
    -------
    X, Y : ndarray of shape (N, N)
    h    : float  grid spacing (= 2L / (N-1))
    """
    x = np.linspace(-L, L, N, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing=indexing)
    h = 2.0 * L / (N - 1)
    return X, Y, float(h)


# ---------------------------------------------------------------------------
# Curvature × h  (axis-aware)
# ---------------------------------------------------------------------------

def compute_hkappa(
    phi: np.ndarray,
    indices: np.ndarray,
    h: float,
    indexing: Literal["ij", "xy"] = "xy",
    *,
    delta: float = 0.0,
) -> np.ndarray:
    """
    Compute h·κ at the given grid indices using central finite differences.

    ``indexing`` tells which axis carries x:
    * "ij" → axis-0 = x (row index steps x), axis-1 = y
    * "xy" → axis-0 = y (row index steps y), axis-1 = x

    Parameters
    ----------
    phi     : 2-D ndarray
    indices : (M, 2) integer array of (row, col) positions
    h       : grid spacing
    indexing: "ij" or "xy"
    delta   : small number added to denominator for numerical safety

    Returns
    -------
    hkappa : (M,) ndarray
    """
    if indices.size == 0:
        return np.zeros((0,), dtype=np.float64)

    rows = indices[:, 0]
    cols = indices[:, 1]

    if indexing == "ij":
        # axis-0 = x, axis-1 = y
        phi_x  = (phi[rows+1, cols  ] - phi[rows-1, cols  ]) / (2.0 * h)
        phi_y  = (phi[rows,   cols+1] - phi[rows,   cols-1]) / (2.0 * h)
        phi_xx = (phi[rows+1, cols  ] - 2.0*phi[rows, cols] + phi[rows-1, cols  ]) / h**2
        phi_yy = (phi[rows,   cols+1] - 2.0*phi[rows, cols] + phi[rows,   cols-1]) / h**2
        phi_xy = (phi[rows+1, cols+1] - phi[rows+1, cols-1]
                  - phi[rows-1, cols+1] + phi[rows-1, cols-1]) / (4.0 * h**2)
    else:
        # axis-0 = y (row), axis-1 = x (col)
        phi_x  = (phi[rows,   cols+1] - phi[rows,   cols-1]) / (2.0 * h)
        phi_y  = (phi[rows+1, cols  ] - phi[rows-1, cols  ]) / (2.0 * h)
        phi_xx = (phi[rows,   cols+1] - 2.0*phi[rows, cols] + phi[rows,   cols-1]) / h**2
        phi_yy = (phi[rows+1, cols  ] - 2.0*phi[rows, cols] + phi[rows-1, cols  ]) / h**2
        phi_xy = (phi[rows+1, cols+1] - phi[rows+1, cols-1]
                  - phi[rows-1, cols+1] + phi[rows-1, cols-1]) / (4.0 * h**2)

    numerator   = phi_x**2 * phi_yy - 2.0 * phi_x * phi_y * phi_xy + phi_y**2 * phi_xx
    denominator = (phi_x**2 + phi_y**2 + delta) ** 1.5
    out = np.zeros_like(numerator)
    mask = denominator > 1e-12
    out[mask] = numerator[mask] / denominator[mask]
    return h * out


# ---------------------------------------------------------------------------
# Flower / exact-SDF analytical utilities
# ---------------------------------------------------------------------------

def build_flower_phi0(X: np.ndarray, Y: np.ndarray, a: float, b: float, p: int) -> np.ndarray:
    """Unsigned level-set of the flower curve r = b + a·cos(p·θ)."""
    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)
    return r - a * np.cos(p * theta) - b


def vectorized_exact_sdf(
    X: np.ndarray,
    Y: np.ndarray,
    a: float,
    b: float,
    p: int,
    *,
    tol: float = 1e-11,
    max_iter: int = 80,
) -> np.ndarray:
    """
    Compute the signed distance function to the flower curve on a full grid
    using a vectorised Newton projection.
    """
    two_pi = 2.0 * np.pi
    shape  = X.shape
    x = X.reshape(-1)
    y = Y.reshape(-1)
    theta = np.arctan2(y, x) % two_pi

    for _ in range(max_iter):
        ct, st   = np.cos(theta), np.sin(theta)
        cpt, spt = np.cos(p * theta), np.sin(p * theta)
        r   =  b + a * cpt
        rp  = -a * p * spt
        rpp = -a * p**2 * cpt
        cx = r * ct;   cy = r * st
        cpx = rp * ct - r * st;   cpy = rp * st + r * ct
        cppx = rpp * ct - 2.0*rp*st - r*ct;   cppy = rpp*st + 2.0*rp*ct - r*st
        dx, dy = cx - x, cy - y
        f  = dx * cpx + dy * cpy
        fp = cpx*cpx + cpy*cpy + dx*cppx + dy*cppy
        active = (np.abs(f) > tol) & (np.abs(fp) >= 1e-18)
        if not np.any(active):
            break
        step = np.zeros_like(theta)
        step[active] = f[active] / fp[active]
        theta = ((theta - step) % two_pi)

    r_c  = b + a * np.cos(p * theta)
    cx_f = r_c * np.cos(theta)
    cy_f = r_c * np.sin(theta)
    dist = np.sqrt((x - cx_f)**2 + (y - cy_f)**2)
    phi0 = np.sqrt(x**2 + y**2) - a * np.cos(p * np.arctan2(y, x)) - b
    return (np.sign(phi0) * dist).reshape(shape)


def _require_mpmath() -> None:
    if mp is None:
        raise RuntimeError(
            "mpmath is required for high_precision_exact_sdf() but is not installed."
        )


def _to_mpf(value: float) -> "mp.mpf":
    return mp.mpf(repr(float(value)))


def _wrap_angle(theta: "mp.mpf", two_pi: "mp.mpf") -> "mp.mpf":
    wrapped = theta % two_pi
    return wrapped + two_pi if wrapped < 0 else wrapped


def _high_precision_projection_theta_single(
    x: float,
    y: float,
    a: float,
    b: float,
    p: int,
    *,
    dps: int,
    tol: float | None,
    max_iter: int,
) -> "mp.mpf":
    _require_mpmath()

    with mp.workdps(int(dps)):
        x_mp = _to_mpf(x)
        y_mp = _to_mpf(y)
        a_mp = _to_mpf(a)
        b_mp = _to_mpf(b)
        p_mp = mp.mpf(int(p))
        two_pi = 2 * mp.pi
        theta = _wrap_angle(mp.atan2(y_mp, x_mp), two_pi)
        tol_mp = mp.mpf(tol) if tol is not None else mp.power(10, -(int(dps) - 10))

        for _ in range(max_iter):
            ct = mp.cos(theta)
            st = mp.sin(theta)
            ptheta = p_mp * theta
            cpt = mp.cos(ptheta)
            spt = mp.sin(ptheta)

            r = b_mp + a_mp * cpt
            rp = -a_mp * p_mp * spt
            rpp = -a_mp * (p_mp**2) * cpt

            cx = r * ct
            cy = r * st
            cpx = rp * ct - r * st
            cpy = rp * st + r * ct
            cppx = rpp * ct - 2 * rp * st - r * ct
            cppy = rpp * st + 2 * rp * ct - r * st

            f = (cx - x_mp) * cpx + (cy - y_mp) * cpy
            if mp.fabs(f) <= tol_mp:
                break

            fp = cpx * cpx + cpy * cpy + (cx - x_mp) * cppx + (cy - y_mp) * cppy
            if mp.fabs(fp) <= mp.eps:
                break

            step = f / fp
            theta = _wrap_angle(theta - step, two_pi)
            if mp.fabs(step) <= tol_mp:
                break

        return theta


def high_precision_exact_sdf(
    X: np.ndarray,
    Y: np.ndarray,
    a: float,
    b: float,
    p: int,
    *,
    dps: int = 80,
    tol: float | None = None,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Compute a high-precision signed distance field with mpmath-backed Newton
    orthogonal projection.

    This is much slower than ``vectorized_exact_sdf`` and is intended for
    benchmark / reference-data generation, not routine dataset builds.
    """
    _require_mpmath()

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} and {Y.shape}")

    result = np.empty_like(X, dtype=np.float64)
    flat_x = np.asarray(X, dtype=np.float64).ravel()
    flat_y = np.asarray(Y, dtype=np.float64).ravel()
    flat_out = result.ravel()

    with mp.workdps(int(dps)):
        a_mp = _to_mpf(a)
        b_mp = _to_mpf(b)
        p_mp = mp.mpf(int(p))

        for idx, (x, y) in enumerate(zip(flat_x.tolist(), flat_y.tolist(), strict=True)):
            theta = _high_precision_projection_theta_single(
                x,
                y,
                a,
                b,
                p,
                dps=dps,
                tol=tol,
                max_iter=max_iter,
            )
            x_mp = _to_mpf(x)
            y_mp = _to_mpf(y)
            r_theta = b_mp + a_mp * mp.cos(p_mp * theta)
            cx = r_theta * mp.cos(theta)
            cy = r_theta * mp.sin(theta)
            dist = mp.sqrt((x_mp - cx) ** 2 + (y_mp - cy) ** 2)

            theta0 = mp.atan2(y_mp, x_mp)
            radial_phi = mp.sqrt(x_mp**2 + y_mp**2) - a_mp * mp.cos(p_mp * theta0) - b_mp
            flat_out[idx] = float(mp.sign(radial_phi) * dist)

    return result


def find_projection_theta(
    xy: np.ndarray,
    a: float,
    b: float,
    p: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> np.ndarray:
    """
    For each sample point (x, y) find the angular parameter θ on the flower
    curve closest to (x, y) via Newton iteration + scipy fallback.
    """
    two_pi = 2.0 * np.pi

    def _terms(t):
        ct, st = np.cos(t), np.sin(t)
        cpt, spt = np.cos(p*t), np.sin(p*t)
        r   =  b + a * cpt
        rp  = -a * p * spt
        rpp = -a * p**2 * cpt
        cx = r*ct;   cy = r*st
        cpx = rp*ct - r*st;   cpy = rp*st + r*ct
        cppx = rpp*ct - 2*rp*st - r*ct;   cppy = rpp*st + 2*rp*ct - r*st
        return cx, cy, cpx, cpy, cppx, cppy

    result = []
    for x, y in xy:
        theta = float(np.arctan2(y, x) % two_pi)
        converged = False
        for _ in range(max_iter):
            cx, cy, cpx, cpy, cppx, cppy = _terms(theta)
            f  = (cx - x)*cpx + (cy - y)*cpy
            fp = cpx**2 + cpy**2 + (cx-x)*cppx + (cy-y)*cppy
            if abs(f) <= tol:
                converged = True
                break
            if abs(fp) < 1e-18:
                break
            step = f / fp
            theta = (theta - step) % two_pi
            if abs(step) <= tol:
                converged = True
                break
        if not converged:
            def _dsq(t_arr, px, py):
                t = float(t_arr[0]) % two_pi
                r = b + a * np.cos(p*t)
                return (px - r*np.cos(t))**2 + (py - r*np.sin(t))**2
            res = minimize(_dsq, np.array([theta]), args=(x, y), tol=tol)
            theta = float(res.x[0]) % two_pi
        result.append(theta)
    return np.array(result, dtype=np.float64)


def hkappa_analytic(
    theta_proj: np.ndarray,
    h: float,
    a: float,
    b: float,
    p: float,
) -> np.ndarray:
    """Analytical h·κ for the flower curve at the projected angles."""
    r   =  b + a * np.cos(p * theta_proj)
    rp  = -a * p * np.sin(p * theta_proj)
    rpp = -a * p**2 * np.cos(p * theta_proj)
    kappa = (r**2 + 2*rp**2 - r*rpp) / (r**2 + rp**2)**1.5
    return h * kappa


# ---------------------------------------------------------------------------
# Reinit-quality evaluator (interface-band ∇φ ≈ 1)
# ---------------------------------------------------------------------------

class ReinitQualityEvaluator:
    """Compute |∇φ| deviation from 1 strictly on interface-adjacent nodes."""

    @staticmethod
    def get_sampling_coordinates(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rows, cols) of interface-adjacent nodes (sign-change edges)."""
        scx = phi[:-1, :] * phi[1:, :] <= 0.0
        scy = phi[:, :-1] * phi[:, 1:] <= 0.0
        mask = np.zeros_like(phi, dtype=bool)
        ix, jx = np.where(scx);  mask[ix, jx] = True;  mask[ix+1, jx] = True
        iy, jy = np.where(scy);  mask[iy, jy] = True;  mask[iy, jy+1] = True
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
        return np.where(mask)

    @classmethod
    def evaluate(cls, phi: np.ndarray, h: float) -> dict:
        dy, dx = np.gradient(phi, h, h, edge_order=1)
        grad_norm = np.sqrt(dx**2 + dy**2)
        I, J = cls.get_sampling_coordinates(phi)
        if len(I) == 0:
            nan = float("nan")
            return {"mean_abs_err_to_1": nan, "max_abs_err_to_1": nan, "grad_norm_std": nan}
        gband = grad_norm[I, J]
        err = np.abs(gband - 1.0)
        return {
            "mean_abs_err_to_1": float(np.mean(err)),
            "max_abs_err_to_1":  float(np.max(err)),
            "grad_norm_std":     float(np.std(gband)),
        }
