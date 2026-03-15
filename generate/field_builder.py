import numpy as np
from typing import Dict, Any

from .geometry import CircleGeometryGenerator


class LevelSetFieldBuilder:
    """
    Build level-set fields from geometry blueprints.
    Indexing convention (IMPORTANT):
        phi[i, j] <-> point (x_i, y_j)
    achieved by np.meshgrid(..., indexing='ij')
    """
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.spacing_atol = 1e-9

    def _build_grid(self, rho: int):
        """
        Build uniform grid on [0,1]x[0,1] with indexing='ij'.
        Returns
        -------
        x : (rho,) ndarray
            x_i = i*h
        y : (rho,) ndarray
            y_j = j*h
        X : (rho, rho) ndarray
            X[i, j] = x_i
        Y : (rho, rho) ndarray
            Y[i, j] = y_j
        h : float
            grid spacing
        """
        rho = int(rho)
        if rho < 2:
            raise ValueError("rho must be >= 2")

        h = 1.0 / (rho - 1)
        x = np.linspace(0.0, 1.0, rho, dtype=self.dtype)
        y = np.linspace(0.0, 1.0, rho, dtype=self.dtype)

        X, Y = np.meshgrid(x, y, indexing='ij')
        return x, y, X, Y, float(h)

    def _parse_blueprint(self, blueprint: Dict[str, Any]):
        """
        Parse and validate minimal required fields from blueprint.
        """
        if "label" not in blueprint:
            raise KeyError("Blueprint missing 'label'; downstream dataset builder may fail.")

        try:
            rho = int(blueprint["meta"]["resolution"])
            h_bp = float(blueprint["params"]["h"])
            r = float(blueprint["params"]["radius"])
            cx, cy = blueprint["params"]["center"]
            cx = float(cx)
            cy = float(cy)
        except KeyError as e:
            raise KeyError(f"Blueprint missing required key: {e}")

        return rho, h_bp, r, cx, cy

    def _pack_output(
        self,
        blueprint: Dict[str, Any],
        phi: np.ndarray,
        phi_type: str,
        return_grid: bool,
        x: np.ndarray,
        y: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        h: float,
    ) -> Dict[str, Any]:
        """
        Standardize output structure and pass through label (IMPORTANT).
        """
        out = {
            "meta": {
                **blueprint["meta"],
                "stage": "level_set_field",
            },
            "params": {
                **blueprint["params"],
            },
            "label": {
                **blueprint["label"],
            },
            "field": {
                "phi_type": phi_type,
                "indexing": "ij",
                "phi": phi.astype(self.dtype, copy=False),
            },
        }

        if return_grid:
            out["grid"] = {
                "x": x,
                "y": y,
                "X": X,
                "Y": Y,
                "h": float(h),
            }

        return out

    def build_circle_sdf(self, blueprint: Dict[str, Any], return_grid: bool = True) -> Dict[str, Any]:
        """
        Build signed-distance level-set field for a circle:

            phi_cs(x,y) = sqrt((x-cx)^2 + (y-cy)^2) - r
        """
        rho, h_bp, r, cx, cy = self._parse_blueprint(blueprint)
        x, y, X, Y, h = self._build_grid(rho)

        if not np.isclose(h, h_bp, atol=self.spacing_atol, rtol=0.0):
            raise ValueError(f"Grid spacing mismatch: blueprint h={h_bp}, rebuilt h={h}")

        phi = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r

        return self._pack_output(
            blueprint=blueprint,
            phi=phi,
            phi_type="circle_sdf",
            return_grid=return_grid,
            x=x, y=y, X=X, Y=Y, h=h
        )

    def build_circle_nonsdf(self, blueprint: Dict[str, Any], return_grid: bool = True) -> Dict[str, Any]:
        """
        Build non-signed-distance circle level-set field:

            phi_cn(x,y) = (x-cx)^2 + (y-cy)^2 - r^2

        Note:
            This is NOT a signed-distance function and is intended as input
            to the reinitialization stage.
        """
        rho, h_bp, r, cx, cy = self._parse_blueprint(blueprint)
        x, y, X, Y, h = self._build_grid(rho)

        if not np.isclose(h, h_bp, atol=self.spacing_atol, rtol=0.0):
            raise ValueError(f"Grid spacing mismatch: blueprint h={h_bp}, rebuilt h={h}")

        phi = (X - cx) ** 2 + (Y - cy) ** 2 - r ** 2

        return self._pack_output(
            blueprint=blueprint,
            phi=phi,
            phi_type="circle_nonsdf",
            return_grid=return_grid,
            x=x, y=y, X=X, Y=Y, h=h
        )

    @staticmethod
    def quick_sanity_checks(field_pack: Dict[str, Any], verbose: bool = True) -> None:
        """
        Quick anti-bug checks for circle fields (sdf or nonsdf).

        Checks:
        1) A grid point nearest to the center should be inside the circle (phi < 0).
        2) A corner point should be outside (typically phi > 0).
        3) There should exist sign changes on grid edges (interface crosses some edges).
        4) label is present and includes h_kappa (for downstream use).
        """
        if "label" not in field_pack:
            raise KeyError("field_pack missing 'label' (dataflow bug).")
        if "h_kappa" not in field_pack["label"]:
            raise KeyError("field_pack['label'] missing 'h_kappa'.")

        phi = field_pack["field"]["phi"]
        phi_type = field_pack["field"]["phi_type"]
        x = field_pack["grid"]["x"]
        y = field_pack["grid"]["y"]
        h = field_pack["grid"]["h"]

        cx, cy = field_pack["params"]["center"]
        r = field_pack["params"]["radius"]
        h_kappa = field_pack["label"]["h_kappa"]

        rho = phi.shape[0]
        assert phi.shape == (rho, rho), "phi must be square (rho, rho)"

        i0 = int(np.argmin(np.abs(x - cx)))
        j0 = int(np.argmin(np.abs(y - cy)))
        phi_center_near = float(phi[i0, j0])
        assert phi_center_near < 0.0, (
            f"Nearest grid point to center is not inside circle: phi[{i0},{j0}]={phi_center_near}"
        )

        phi_corner = float(phi[0, 0])
        assert phi_corner > 0.0, f"Corner point unexpectedly not outside: phi[0,0]={phi_corner}"

        sign_change_i = np.any(phi[:-1, :] * phi[1:, :] <= 0.0)
        sign_change_j = np.any(phi[:, :-1] * phi[:, 1:] <= 0.0)
        assert (sign_change_i or sign_change_j), "No sign-changing edges found; interface may be missing."

        phi_min = float(np.min(phi))
        phi_max = float(np.max(phi))
        min_abs_phi = float(np.min(np.abs(phi)))

        if verbose:
            print("[Sanity Checks Passed]")
            print(f"  phi_type             : {phi_type}")
            print(f"  indexing convention  : phi[i,j] <-> (x_i, y_j)")
            print(f"  rho                  : {rho}")
            print(f"  h                    : {h:.12f}")
            print(f"  center               : ({cx:.12f}, {cy:.12f})")
            print(f"  radius               : {r:.12f}")
            print(f"  label h_kappa        : {h_kappa:.12f}")
            print(f"  nearest-center idx   : (i0,j0)=({i0},{j0})")
            print(f"  phi[i0,j0]           : {phi_center_near:.12f}  (should be < 0)")
            print(f"  phi[0,0]             : {phi_corner:.12f}  (should be > 0)")
            print(f"  sign-change edges    : i-dir={bool(sign_change_i)}, j-dir={bool(sign_change_j)}")
            print(f"  min(phi), max(phi)   : ({phi_min:.12f}, {phi_max:.12f})")
            print(f"  min |phi| on grid    : {min_abs_phi:.12e} (not necessarily 0 due to grid alignment)")


if __name__ == "__main__":
    gen = CircleGeometryGenerator(resolution_rho=256, seed=42, variations=5)
    blueprints = gen.generate_blueprints()
    bp = blueprints[0]

    builder = LevelSetFieldBuilder(dtype=np.float64)

    pack_sdf = builder.build_circle_sdf(bp, return_grid=True)
    print(f"[SDF] blueprint_id = {pack_sdf['meta']['blueprint_id']}")
    print(f"[SDF] phi shape    = {pack_sdf['field']['phi'].shape}")
    builder.quick_sanity_checks(pack_sdf, verbose=True)
    print()

    pack_nonsdf = builder.build_circle_nonsdf(bp, return_grid=True)
    print(f"[NonSDF] blueprint_id = {pack_nonsdf['meta']['blueprint_id']}")
    print(f"[NonSDF] phi shape     = {pack_nonsdf['field']['phi'].shape}")
    builder.quick_sanity_checks(pack_nonsdf, verbose=True)
