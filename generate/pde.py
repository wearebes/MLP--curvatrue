"""
core/pde.py — Unified PDE level-set reinitialization kernel.

One class ``LevelSetReinitializer`` handles both axis-indexing conventions:

* ``indexing="ij"``  – used by the training data pipeline (generate/dataset_train.py)
                       phi[i, j]  <-> (x_i, y_j)  with meshgrid(x, y, indexing='ij')
* ``indexing="xy"``  – used by the test-data / evaluation pipeline (exact_sdf chain)
                       phi[row, col]  <->  (x_col, y_row)  with meshgrid(x, x, indexing='xy')

The numerical scheme is identical in both cases; only the axis semantics of
the finite-difference stencils are swapped.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LevelSetReinitializer:
    """
    Level-set reinitialization PDE solver.

    Numerical scheme
    ----------------
    * Spatial: 3rd-order one-sided / 4th-order one-sided / 5th-order WENO
    * Temporal: 2nd-order / 3rd-order TVD Runge-Kutta
    * Hamiltonian: classic Rouy-Tourin Godunov upwind

    Parameters
    ----------
    indexing : {"ij", "xy"}
        Axis convention of the input phi array.
        "ij" → phi[i, j] corresponds to spatial point (x_i, y_j).
        "xy" → phi[row, col] corresponds to spatial point (x_col, y_row).
    cfl, eps_weno, eps_sign_factor : float
        Solver hyper-parameters.
    time_order : int
        TVD-RK order (2 or 3).
    space_order : int
        One-sided derivative order (3, 4, or 5).
    """

    def __init__(
        self,
        indexing: Literal["ij", "xy"] = "ij",
        cfl: float = 0.5,
        eps_weno: float = 1e-6,
        eps_sign_factor: float = 1.0,
        sign_mode: Literal["frozen_phi0", "dynamic_phi"] = "frozen_phi0",
        *,
        time_order: int = 3,
        space_order: int = 5,
    ) -> None:
        if indexing not in ("ij", "xy"):
            raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing!r}")
        if time_order not in (2, 3):
            raise ValueError(f"time_order must be 2 or 3, got {time_order}")
        if space_order not in (3, 4, 5):
            raise ValueError(f"space_order must be 3, 4, or 5, got {space_order}")
        if sign_mode not in ("frozen_phi0", "dynamic_phi"):
            raise ValueError(f"sign_mode must be 'frozen_phi0' or 'dynamic_phi', got {sign_mode!r}")
        self.indexing = indexing
        self.cfl = float(cfl)
        self.eps_weno = float(eps_weno)
        self.eps_sign_factor = float(eps_sign_factor)
        self.sign_mode = sign_mode
        self.time_order = int(time_order)
        self.space_order = int(space_order)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smoothed_sign(self, phi0: np.ndarray, h: float) -> np.ndarray:
        eps = self.eps_sign_factor * h
        return phi0 / np.sqrt(phi0 ** 2 + eps ** 2)

    @staticmethod
    def _hj_weno5_1d(v1, v2, v3, v4, v5) -> np.ndarray:
        eps = 1e-6  # fixed; caller may set eps_weno as attribute but WENO uses local
        b0 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + (1.0 / 4.0) * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
        b1 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + (1.0 / 4.0) * (v2 - v4) ** 2
        b2 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + (1.0 / 4.0) * (3.0 * v3 - 4.0 * v4 + v5) ** 2
        a0 = 0.1 / (b0 + eps) ** 2
        a1 = 0.6 / (b1 + eps) ** 2
        a2 = 0.3 / (b2 + eps) ** 2
        sa = a0 + a1 + a2
        w0, w1, w2 = a0 / sa, a1 / sa, a2 / sa
        p0 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p1 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p2 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5
        return w0 * p0 + w1 * p1 + w2 * p2

    # The _hj_weno5_1d above uses a hard-coded eps for WENO smoothness weights.
    # Use eps_weno from self when calling the WENO path.
    def _hj_weno5_1d_eps(self, v1, v2, v3, v4, v5) -> np.ndarray:
        b0 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + (1.0 / 4.0) * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
        b1 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + (1.0 / 4.0) * (v2 - v4) ** 2
        b2 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + (1.0 / 4.0) * (3.0 * v3 - 4.0 * v4 + v5) ** 2
        a0 = 0.1 / (b0 + self.eps_weno) ** 2
        a1 = 0.6 / (b1 + self.eps_weno) ** 2
        a2 = 0.3 / (b2 + self.eps_weno) ** 2
        sa = a0 + a1 + a2
        w0, w1, w2 = a0 / sa, a1 / sa, a2 / sa
        p0 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3
        p1 = -(1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4
        p2 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5
        return w0 * p0 + w1 * p1 + w2 * p2

    # ------------------------------------------------------------------
    # Derivative stencils — axis-aware
    # ------------------------------------------------------------------
    # In "ij" mode:  axis-0 = x direction, axis-1 = y direction
    # In "xy" mode:  axis-0 = y direction, axis-1 = x direction
    #
    # We unify by always calling the stencil along "axis-0" for the first
    # spatial coordinate and "axis-1" for the second, then name them
    # appropriately based on indexing.

    def _deriv_space3(self, phi: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
        """Return (D0_m, D0_p, D1_m, D1_p) one-sided 3rd-order differences."""
        n0, n1 = phi.shape
        pp = np.pad(phi, 3, mode="edge")

        if self.indexing == "ij":
            # axis-0 = x
            d0_m = (11*pp[3:n0+3,3:n1+3] - 18*pp[2:n0+2,3:n1+3] + 9*pp[1:n0+1,3:n1+3] - 2*pp[0:n0,3:n1+3]) / (6*h)
            d0_p = (-11*pp[3:n0+3,3:n1+3] + 18*pp[4:n0+4,3:n1+3] - 9*pp[5:n0+5,3:n1+3] + 2*pp[6:n0+6,3:n1+3]) / (6*h)
            d1_m = (11*pp[3:n0+3,3:n1+3] - 18*pp[3:n0+3,2:n1+2] + 9*pp[3:n0+3,1:n1+1] - 2*pp[3:n0+3,0:n1]) / (6*h)
            d1_p = (-11*pp[3:n0+3,3:n1+3] + 18*pp[3:n0+3,4:n1+4] - 9*pp[3:n0+3,5:n1+5] + 2*pp[3:n0+3,6:n1+6]) / (6*h)
        else:
            # axis-0 = y (row), axis-1 = x (col) → x derivatives live on axis-1
            d0_m = (11*pp[3:n0+3,3:n1+3] - 18*pp[3:n0+3,2:n1+2] + 9*pp[3:n0+3,1:n1+1] - 2*pp[3:n0+3,0:n1]) / (6*h)
            d0_p = (-11*pp[3:n0+3,3:n1+3] + 18*pp[3:n0+3,4:n1+4] - 9*pp[3:n0+3,5:n1+5] + 2*pp[3:n0+3,6:n1+6]) / (6*h)
            d1_m = (11*pp[3:n0+3,3:n1+3] - 18*pp[2:n0+2,3:n1+3] + 9*pp[1:n0+1,3:n1+3] - 2*pp[0:n0,3:n1+3]) / (6*h)
            d1_p = (-11*pp[3:n0+3,3:n1+3] + 18*pp[4:n0+4,3:n1+3] - 9*pp[5:n0+5,3:n1+3] + 2*pp[6:n0+6,3:n1+3]) / (6*h)
        return d0_m, d0_p, d1_m, d1_p

    def _deriv_space4(self, phi: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
        n0, n1 = phi.shape
        pp = np.pad(phi, 4, mode="edge")

        if self.indexing == "ij":
            d0_m = (25*pp[4:n0+4,4:n1+4] - 48*pp[3:n0+3,4:n1+4] + 36*pp[2:n0+2,4:n1+4] - 16*pp[1:n0+1,4:n1+4] + 3*pp[0:n0,4:n1+4]) / (12*h)
            d0_p = (-25*pp[4:n0+4,4:n1+4] + 48*pp[5:n0+5,4:n1+4] - 36*pp[6:n0+6,4:n1+4] + 16*pp[7:n0+7,4:n1+4] - 3*pp[8:n0+8,4:n1+4]) / (12*h)
            d1_m = (25*pp[4:n0+4,4:n1+4] - 48*pp[4:n0+4,3:n1+3] + 36*pp[4:n0+4,2:n1+2] - 16*pp[4:n0+4,1:n1+1] + 3*pp[4:n0+4,0:n1]) / (12*h)
            d1_p = (-25*pp[4:n0+4,4:n1+4] + 48*pp[4:n0+4,5:n1+5] - 36*pp[4:n0+4,6:n1+6] + 16*pp[4:n0+4,7:n1+7] - 3*pp[4:n0+4,8:n1+8]) / (12*h)
        else:
            # xy: x on axis-1, y on axis-0
            d0_m = (25*pp[4:n0+4,4:n1+4] - 48*pp[4:n0+4,3:n1+3] + 36*pp[4:n0+4,2:n1+2] - 16*pp[4:n0+4,1:n1+1] + 3*pp[4:n0+4,0:n1]) / (12*h)
            d0_p = (-25*pp[4:n0+4,4:n1+4] + 48*pp[4:n0+4,5:n1+5] - 36*pp[4:n0+4,6:n1+6] + 16*pp[4:n0+4,7:n1+7] - 3*pp[4:n0+4,8:n1+8]) / (12*h)
            d1_m = (25*pp[4:n0+4,4:n1+4] - 48*pp[3:n0+3,4:n1+4] + 36*pp[2:n0+2,4:n1+4] - 16*pp[1:n0+1,4:n1+4] + 3*pp[0:n0,4:n1+4]) / (12*h)
            d1_p = (-25*pp[4:n0+4,4:n1+4] + 48*pp[5:n0+5,4:n1+4] - 36*pp[6:n0+6,4:n1+4] + 16*pp[7:n0+7,4:n1+4] - 3*pp[8:n0+8,4:n1+4]) / (12*h)
        return d0_m, d0_p, d1_m, d1_p

    def _deriv_weno5(self, phi: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
        n0, n1 = phi.shape
        pp = np.pad(phi, 3, mode="edge")

        if self.indexing == "ij":
            # axis-0 diffs for x, axis-1 diffs for y
            dx = (pp[1:, :] - pp[:-1, :]) / h
            dy = (pp[:, 1:] - pp[:, :-1]) / h
            d0_m = self._hj_weno5_1d_eps(dx[0:n0,3:-3], dx[1:n0+1,3:-3], dx[2:n0+2,3:-3], dx[3:n0+3,3:-3], dx[4:n0+4,3:-3])
            d0_p = self._hj_weno5_1d_eps(dx[5:n0+5,3:-3], dx[4:n0+4,3:-3], dx[3:n0+3,3:-3], dx[2:n0+2,3:-3], dx[1:n0+1,3:-3])
            d1_m = self._hj_weno5_1d_eps(dy[3:-3,0:n1], dy[3:-3,1:n1+1], dy[3:-3,2:n1+2], dy[3:-3,3:n1+3], dy[3:-3,4:n1+4])
            d1_p = self._hj_weno5_1d_eps(dy[3:-3,5:n1+5], dy[3:-3,4:n1+4], dy[3:-3,3:n1+3], dy[3:-3,2:n1+2], dy[3:-3,1:n1+1])
        else:
            # axis-1 diffs for x, axis-0 diffs for y
            dx = (pp[:, 1:] - pp[:, :-1]) / h
            dy = (pp[1:, :] - pp[:-1, :]) / h
            d0_m = self._hj_weno5_1d_eps(dx[3:-3,0:n1], dx[3:-3,1:n1+1], dx[3:-3,2:n1+2], dx[3:-3,3:n1+3], dx[3:-3,4:n1+4])
            d0_p = self._hj_weno5_1d_eps(dx[3:-3,5:n1+5], dx[3:-3,4:n1+4], dx[3:-3,3:n1+3], dx[3:-3,2:n1+2], dx[3:-3,1:n1+1])
            d1_m = self._hj_weno5_1d_eps(dy[0:n0,3:-3], dy[1:n0+1,3:-3], dy[2:n0+2,3:-3], dy[3:n0+3,3:-3], dy[4:n0+4,3:-3])
            d1_p = self._hj_weno5_1d_eps(dy[5:n0+5,3:-3], dy[4:n0+4,3:-3], dy[3:n0+3,3:-3], dy[2:n0+2,3:-3], dy[1:n0+1,3:-3])
        return d0_m, d0_p, d1_m, d1_p

    def _get_derivatives(self, phi: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
        if self.space_order == 3:
            return self._deriv_space3(phi, h)
        if self.space_order == 4:
            return self._deriv_space4(phi, h)
        return self._deriv_weno5(phi, h)

    @staticmethod
    def _godunov_grad_norm(dm0, dp0, dm1, dp1, S0: np.ndarray) -> np.ndarray:
        gp = np.sqrt(
            np.maximum(np.maximum(dm0, -dp0), 0.0) ** 2
            + np.maximum(np.maximum(dm1, -dp1), 0.0) ** 2
        )
        gm = np.sqrt(
            np.maximum(np.maximum(-dm0, dp0), 0.0) ** 2
            + np.maximum(np.maximum(-dm1, dp1), 0.0) ** 2
        )
        return np.where(S0 >= 0, gp, gm)

    def _rhs(self, phi: np.ndarray, S0: np.ndarray, h: float) -> np.ndarray:
        dm0, dp0, dm1, dp1 = self._get_derivatives(phi, h)
        G = self._godunov_grad_norm(dm0, dp0, dm1, dp1, S0)
        return -S0 * (G - 1.0)

    def _sign_field(self, phi_stage: np.ndarray, phi0: np.ndarray, h: float) -> np.ndarray:
        if self.sign_mode == "dynamic_phi":
            return self._smoothed_sign(phi_stage, h)
        return self._smoothed_sign(phi0, h)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reinitialize(self, phi0: np.ndarray, h: float, n_steps: int) -> np.ndarray:
        """Run n_steps of TVD-RK reinitialization on phi0."""
        if n_steps <= 0:
            return phi0.copy()
        phi = phi0.astype(np.float64, copy=True)
        dt = self.cfl * h

        for _ in range(n_steps):
            if self.time_order == 2:
                S1 = self._sign_field(phi, phi0, h)
                L1 = self._rhs(phi, S1, h)
                phi_1 = phi + dt * L1
                S2 = self._sign_field(phi_1, phi0, h)
                L2 = self._rhs(phi_1, S2, h)
                phi = 0.5 * phi + 0.5 * (phi_1 + dt * L2)
            else:
                S1 = self._sign_field(phi, phi0, h)
                L1 = self._rhs(phi, S1, h)
                phi_1 = phi + dt * L1
                S2 = self._sign_field(phi_1, phi0, h)
                L2 = self._rhs(phi_1, S2, h)
                phi_2 = 0.75 * phi + 0.25 * (phi_1 + dt * L2)
                S3 = self._sign_field(phi_2, phi0, h)
                L3 = self._rhs(phi_2, S3, h)
                phi = (1.0 / 3.0) * phi + (2.0 / 3.0) * (phi_2 + dt * L3)
        return phi
