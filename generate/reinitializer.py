import copy
import numpy as np
from typing import Dict, Any, List, Tuple

from .config import CFL, EPS_WENO, EPS_SIGN_FACTOR, REINIT_STEPS


class ReinitQualityEvaluator:
    """
    Evaluates the quality of a level-set field (|∇φ| ≈ 1)
    strictly on the grid nodes to be sampled for training.
    """

    @staticmethod
    def _central_grad_norm(phi: np.ndarray, h: float) -> np.ndarray:
        """Computes gradient norm using central differences for diagnostics."""
        dy, dx = np.gradient(phi, h, h, edge_order=1)
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def get_sampling_coordinates(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [PUBLIC API]
        Finds the unique interior grid nodes adjacent to interface-crossing
        horizontal or vertical edges.

        For every crossed edge, both endpoints are added to the candidate
        sample-node set. The final coordinates are deduplicated so each
        interface-adjacent node contributes a single 3x3 stencil.
        """
        # Sign changes across vertical and horizontal edges.
        sign_change_x = phi[:-1, :] * phi[1:, :] <= 0.0
        sign_change_y = phi[:, :-1] * phi[:, 1:] <= 0.0

        mask = np.zeros_like(phi, dtype=bool)

        # Vertical edges connect (i, j) <-> (i + 1, j).
        ix, jx = np.where(sign_change_x)
        mask[ix, jx] = True
        mask[ix + 1, jx] = True

        # Horizontal edges connect (i, j) <-> (i, j + 1).
        iy, jy = np.where(sign_change_y)
        mask[iy, jy] = True
        mask[iy, jy + 1] = True

        # Keep only interior nodes so every sample has a valid 3x3 stencil.
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False

        I, J = np.where(mask)

        return I, J

    @classmethod
    def evaluate(cls, phi: np.ndarray, h: float) -> Dict[str, float]:
        """Compute SDF quality metrics exactly on the nodes to be sampled."""
        grad_norm = cls._central_grad_norm(phi, h)

        I, J = cls.get_sampling_coordinates(phi)

        if len(I) == 0:
            return {
                "mean_abs_err_to_1": float('nan'),
                "max_abs_err_to_1": float('nan'),
                "grad_norm_std": float('nan'),
            }

        # Evaluate only at the selected coordinates (duplicates don't skew the max/mean)
        grad_in_band = grad_norm[I, J]
        abs_err = np.abs(grad_in_band - 1.0)

        return {
            "mean_abs_err_to_1": float(np.mean(abs_err)),
            "max_abs_err_to_1": float(np.max(abs_err)),
            "grad_norm_std": float(np.std(grad_in_band)),
        }


class LevelSetReinitializer:
    """
    Level-set reinitialization PDE solver implementing:
    - 5th-order Hamilton-Jacobi WENO (Jiang & Peng, 2000)
    - Classic Rouy-Tourin Godunov upwind Hamiltonian
    - 3rd-order TVD Runge-Kutta (SSP-RK3)
    """
    def __init__(self, cfl: float = CFL, eps_weno: float = EPS_WENO, eps_sign_factor: float = EPS_SIGN_FACTOR):
        self.cfl = cfl
        self.eps_weno = eps_weno
        self.eps_sign_factor = eps_sign_factor

    def _smoothed_sign(self, phi0: np.ndarray, h: float) -> np.ndarray:
        """Smoothed sign function S(phi0) = phi0 / sqrt(phi0^2 + eps^2)."""
        eps = self.eps_sign_factor * h
        return phi0 / np.sqrt(phi0**2 + eps**2)

    def _hj_weno5_1d(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray, v5: np.ndarray) -> np.ndarray:
        """
        Inputs v1, v2, v3, v4, v5 represent the 1st-order difference quotients
        (delta_{i-5/2}, ..., delta_{i+3/2}) for D^-
        or their flipped counterparts for D^+.
        """
        # 1. Smoothness Indicators (beta_k)
        beta0 = (13.0/12.0) * (v1 - 2.0*v2 + v3)**2 + (1.0/4.0) * (v1 - 4.0*v2 + 3.0*v3)**2
        beta1 = (13.0/12.0) * (v2 - 2.0*v3 + v4)**2 + (1.0/4.0) * (v2 - v4)**2
        beta2 = (13.0/12.0) * (v3 - 2.0*v4 + v5)**2 + (1.0/4.0) * (3.0*v3 - 4.0*v4 + v5)**2

        # 2. Linear and Nonlinear Weights (alpha_k and omega_k)
        alpha0 = 0.1 / (beta0 + self.eps_weno)**2
        alpha1 = 0.6 / (beta1 + self.eps_weno)**2
        alpha2 = 0.3 / (beta2 + self.eps_weno)**2
        sum_alpha = alpha0 + alpha1 + alpha2

        omega0 = alpha0 / sum_alpha
        omega1 = alpha1 / sum_alpha
        omega2 = alpha2 / sum_alpha

        # 3. Polynomial Approximations (p_k)
        p0 = (1.0/3.0)*v1 - (7.0/6.0)*v2 + (11.0/6.0)*v3
        p1 = -(1.0/6.0)*v2 + (5.0/6.0)*v3 + (1.0/3.0)*v4
        p2 = (1.0/3.0)*v3 + (5.0/6.0)*v4 - (1.0/6.0)*v5

        # 4. Final WENO5 Approximation
        return omega0*p0 + omega1*p1 + omega2*p2

    def _get_derivatives_weno5(self, phi: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the four one-sided spatial derivatives (Dx_m, Dx_p, Dy_m, Dy_p)
        using the standard HJ-WENO5 difference quotient mapping.
        """
        nx, ny = phi.shape

        # Pad domain with constant extrapolation (3 layers needed for 5-point stencil)
        phi_pad = np.pad(phi, pad_width=3, mode='edge')

        # Forward difference quotients: d_x[i] corresponds to delta_{i+1/2}
        d_x = (phi_pad[1:, :] - phi_pad[:-1, :]) / h
        d_y = (phi_pad[:, 1:] - phi_pad[:, :-1]) / h

        # X-Direction Derivatives
        Dx_m = self._hj_weno5_1d(
            v1=d_x[0:nx, 3:-3],    # delta_{i-5/2}
            v2=d_x[1:nx+1, 3:-3],  # delta_{i-3/2}
            v3=d_x[2:nx+2, 3:-3],  # delta_{i-1/2}
            v4=d_x[3:nx+3, 3:-3],  # delta_{i+1/2}
            v5=d_x[4:nx+4, 3:-3]   # delta_{i+3/2}
        )

        Dx_p = self._hj_weno5_1d(
            v1=d_x[5:nx+5, 3:-3],  # delta_{i+5/2}
            v2=d_x[4:nx+4, 3:-3],  # delta_{i+3/2}
            v3=d_x[3:nx+3, 3:-3],  # delta_{i+1/2}
            v4=d_x[2:nx+2, 3:-3],  # delta_{i-1/2}
            v5=d_x[1:nx+1, 3:-3]   # delta_{i-3/2}
        )

        Dy_m = self._hj_weno5_1d(
            v1=d_y[3:-3, 0:ny],
            v2=d_y[3:-3, 1:ny+1],
            v3=d_y[3:-3, 2:ny+2],
            v4=d_y[3:-3, 3:ny+3],
            v5=d_y[3:-3, 4:ny+4]
        )

        Dy_p = self._hj_weno5_1d(
            v1=d_y[3:-3, 5:ny+5],  # delta_{j+5/2}
            v2=d_y[3:-3, 4:ny+4],  # delta_{j+3/2}
            v3=d_y[3:-3, 3:ny+3],  # delta_{j+1/2}
            v4=d_y[3:-3, 2:ny+2],
            v5=d_y[3:-3, 1:ny+1]
        )

        return Dx_m, Dx_p, Dy_m, Dy_p

    def _godunov_grad_norm(self, Dx_m: np.ndarray, Dx_p: np.ndarray, Dy_m: np.ndarray, Dy_p: np.ndarray, S0: np.ndarray) -> np.ndarray:
        """
        Computes the gradient norm using the classic Rouy-Tourin Godunov formulation.
        """
        # For S0 >= 0: max(D^-, -D^+, 0)^2 for each direction
        grad_plus = np.sqrt(
            np.maximum(np.maximum(Dx_m, -Dx_p), 0.0)**2 +
            np.maximum(np.maximum(Dy_m, -Dy_p), 0.0)**2
        )

        # For S0 < 0: max(-D^-, D^+, 0)^2 for each direction
        grad_minus = np.sqrt(
            np.maximum(np.maximum(-Dx_m, Dx_p), 0.0)**2 +
            np.maximum(np.maximum(-Dy_m, Dy_p), 0.0)**2
        )

        return np.where(S0 >= 0, grad_plus, grad_minus)

    def _compute_rhs(self, phi: np.ndarray, S0: np.ndarray, h: float) -> np.ndarray:
        """Computes the semi-discrete right-hand side: -S0 * (|grad_phi|_G - 1)."""
        Dx_m, Dx_p, Dy_m, Dy_p = self._get_derivatives_weno5(phi, h)
        grad_G = self._godunov_grad_norm(Dx_m, Dx_p, Dy_m, Dy_p, S0)
        return -S0 * (grad_G - 1.0)

    def reinitialize(self, phi0: np.ndarray, h: float, n_steps: int) -> np.ndarray:
        """Executes the SSP-RK3 pseudo-time integration."""
        if n_steps <= 0:
            return phi0.copy()

        # Enforce float64 for PDE iterations to avoid truncation errors
        phi = phi0.astype(np.float64, copy=True)
        S0 = self._smoothed_sign(phi, h)
        dt = self.cfl * h

        for _ in range(n_steps):
            L1 = self._compute_rhs(phi, S0, h)
            phi_1 = phi + dt * L1

            L2 = self._compute_rhs(phi_1, S0, h)
            phi_2 = 0.75 * phi + 0.25 * (phi_1 + dt * L2)

            L3 = self._compute_rhs(phi_2, S0, h)
            phi = (1.0/3.0) * phi + (2.0/3.0) * (phi_2 + dt * L3)

        return phi


class ReinitFieldPackBuilder:
    """
    Pipeline integrator: handles SDF pass-through and non-SDF batch reinitialization.
    """
    def __init__(self, cfl: float = CFL):
        self.reinitializer = LevelSetReinitializer(cfl=cfl)

    def build(self, field_pack: Dict[str, Any], steps_list: List[int] = REINIT_STEPS) -> Dict[str, Dict]:
        """
        Routes the field_pack based on whether it is an exact SDF or non-SDF.
        """
        # Defensive programming to prevent processing fields that were already reinitialized
        if field_pack["meta"].get("reinit") is not None:
            print(f"[Warning] Blueprint {field_pack['meta'].get('blueprint_id', 'Unknown')} already has reinit records. Skipping.")
            return {}

        phi0 = field_pack["field"]["phi"]
        h = field_pack["params"]["h"]

        # Determine if this is an SDF based on phi_type field
        phi_type = field_pack["field"].get("phi_type", "")
        is_sdf = "sdf" in phi_type and "nonsdf" not in phi_type

        # Route 1: SDF Pass-Through
        if is_sdf:
            pack_out = copy.deepcopy(field_pack)
            pack_out["field"]["phi"] = pack_out["field"]["phi"].astype(np.float32)
            return {"0": pack_out}

        # Route 2: Non-SDF Reinitialization Generation
        results = {}
        for steps in steps_list:
            phi_re = self.reinitializer.reinitialize(phi0, h, steps)
            metrics = ReinitQualityEvaluator.evaluate(phi_re, h)

            new_pack = copy.deepcopy(field_pack)
            new_pack["field"]["phi"] = phi_re.astype(np.float32)
            new_pack["field"]["phi_type"] += f"_reinit{steps}"

            new_pack["meta"]["reinit"] = {
                "steps": steps,
                "scheme": "HJ-WENO5 + SSP-RK3 + Godunov (Rouy-Tourin)",
                "metrics_near_interface": metrics
            }
            results[str(steps)] = new_pack

        return results
