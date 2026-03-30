"""
generate/train_data.py — 训练数据生成完整流水线。

合并自：geometry.py + pde_utils.py + stencil_encoding.py + dataset_compiler.py
         + validation.py + dataset_train.py

公共 API
--------
generate_train_datasets(target_rhos, output_dir, *, config)
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from .pde import LevelSetReinitializer
from .numerics import ReinitQualityEvaluator
from .config import GenerateConfig, load_generate_config


# ═══════════════════════════════════════════════════════════════════════════
# § 1  Stencil encoding
# ═══════════════════════════════════════════════════════════════════════════

def encode_patch_legacy_flat(patch_2d: np.ndarray) -> np.ndarray:
    return np.asarray(patch_2d, dtype=np.float64).reshape(-1)


def encode_patch_training_order(patch_2d: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch_2d, dtype=np.float64)
    return patch[:, ::-1].T.reshape(-1)


def extract_3x3_stencils(
    phi: np.ndarray,
    indices: np.ndarray,
    *,
    dtype=np.float64,
) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, 9), dtype=dtype)
    stencils = []
    for row, col in indices:
        patch = phi[row - 1:row + 2, col - 1:col + 2]
        stencils.append(encode_patch_training_order(patch))
    return np.asarray(stencils, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Geometry blueprint generator
# ═══════════════════════════════════════════════════════════════════════════

class CircleGeometryGenerator:
    def __init__(self, resolution_rho: int, seed: int = 42, variations: int = 5):
        self.rho = int(resolution_rho)
        self.global_seed = int(seed)
        self.variations = int(variations)

        if self.variations < 1:
            raise ValueError("variations must be >= 1")

        self.h = 1.0 / (self.rho - 1)
        self.r_min = 1.6 * self.h
        self.r_max = 0.5 - 2.0 * self.h
        self.num_radii = int(np.floor((self.rho - 8.2) / 2.0)) + 1

        if self.num_radii < 1 or self.r_min >= self.r_max:
            raise ValueError(f"Resolution rho={self.rho} too small for valid circular interfaces.")

        self.radii_set = np.linspace(self.r_min, self.r_max, self.num_radii, dtype=float)
        self.center_min = 0.5 - self.h / 2.0
        self.center_max = 0.5 + self.h / 2.0

    def _subseed(self, r_idx: int, v_idx: int) -> int:
        mask64 = (1 << 64) - 1
        x = int(self.global_seed) & mask64
        x ^= 1469598103934665603;  x &= mask64
        x ^= ((int(r_idx) + 1) * 1099511628211) & mask64;  x &= mask64
        x ^= ((int(v_idx) + 1) * 14029467366897019727) & mask64;  x &= mask64
        return int(x % (1 << 32))

    def generate_blueprints(self) -> List[Dict]:
        blueprints = []
        for r_idx, r in enumerate(self.radii_set):
            analytic_kappa = 1.0 / r
            analytic_h_kappa = self.h / r
            for v_idx in range(self.variations):
                sub_seed = self._subseed(r_idx, v_idx)
                rng = np.random.default_rng(sub_seed)
                cx = rng.uniform(self.center_min, self.center_max)
                cy = rng.uniform(self.center_min, self.center_max)
                blueprints.append({
                    "meta": {
                        "blueprint_id": f"rho{self.rho}_r{r_idx:03d}_v{v_idx:02d}_s{sub_seed}",
                        "blueprint_idx": int(self.rho * 100000 + r_idx * self.variations + v_idx),
                        "geometry_type": "circle",
                        "resolution": self.rho,
                        "radius_idx": int(r_idx),
                        "variation_idx": int(v_idx),
                        "global_seed": self.global_seed,
                        "sub_seed": sub_seed,
                    },
                    "params": {"h": float(self.h), "radius": float(r), "center": (float(cx), float(cy))},
                    "label": {"source": "analytic_circle", "kappa": float(analytic_kappa), "h_kappa": float(analytic_h_kappa)},
                })
        return blueprints


# ═══════════════════════════════════════════════════════════════════════════
# § 3  Level-set field builders
# ═══════════════════════════════════════════════════════════════════════════

class LevelSetFieldBuilder:
    """Build level-set fields from blueprints. phi[i, j] <-> (x_i, y_j) (indexing='ij')."""

    def __init__(self, dtype=np.float64) -> None:
        self.dtype = dtype

    def _build_grid(self, rho: int):
        rho = int(rho)
        h = 1.0 / (rho - 1)
        x = np.linspace(0.0, 1.0, rho, dtype=self.dtype)
        X, Y = np.meshgrid(x, x, indexing="ij")
        return x, X, Y, float(h)

    def _parse_blueprint(self, bp: Dict[str, Any]):
        rho = int(bp["meta"]["resolution"])
        h_bp = float(bp["params"]["h"])
        r = float(bp["params"]["radius"])
        cx, cy = float(bp["params"]["center"][0]), float(bp["params"]["center"][1])
        return rho, h_bp, r, cx, cy

    def _pack(self, bp, phi, phi_type, x, X, Y, h, return_grid):
        out = {
            "meta":   {**bp["meta"], "stage": "level_set_field"},
            "params": {**bp["params"]},
            "label":  {**bp["label"]},
            "field":  {"phi_type": phi_type, "indexing": "ij", "phi": phi.astype(self.dtype, copy=False)},
        }
        if return_grid:
            out["grid"] = {"x": x, "X": X, "Y": Y, "h": float(h)}
        return out

    def build_circle_sdf(self, bp: Dict[str, Any], *, return_grid: bool = True):
        rho, h_bp, r, cx, cy = self._parse_blueprint(bp)
        x, X, Y, h = self._build_grid(rho)
        phi = np.sqrt((X - cx)**2 + (Y - cy)**2) - r
        return self._pack(bp, phi, "circle_sdf", x, X, Y, h, return_grid)

    def build_circle_nonsdf(self, bp: Dict[str, Any], *, return_grid: bool = True):
        rho, h_bp, r, cx, cy = self._parse_blueprint(bp)
        x, X, Y, h = self._build_grid(rho)
        phi = (X - cx)**2 + (Y - cy)**2 - r**2
        return self._pack(bp, phi, "circle_nonsdf", x, X, Y, h, return_grid)


class ReinitFieldPackBuilder:
    """Integrate reinitialization into field packs (indexing='ij')."""

    def __init__(
        self,
        cfl: float = 0.5,
        *,
        sign_mode: str = "frozen_phi0",
        time_order: int = 3,
        space_order: int = 5,
    ) -> None:
        self.reinitializer = LevelSetReinitializer(
            indexing="ij",
            cfl=cfl,
            sign_mode=sign_mode,
            time_order=time_order,
            space_order=space_order,
        )
        self.sign_mode = str(sign_mode)
        self.time_order = int(time_order)
        self.space_order = int(space_order)

    def build(self, field_pack: Dict[str, Any], steps_list: List[int] | None = None) -> Dict[str, Dict]:
        if steps_list is None:
            steps_list = [5, 10, 15, 20]
        if field_pack["meta"].get("reinit") is not None:
            return {}

        phi0 = field_pack["field"]["phi"]
        h = field_pack["params"]["h"]
        phi_type = field_pack["field"].get("phi_type", "")
        is_sdf = "sdf" in phi_type and "nonsdf" not in phi_type

        if is_sdf:
            pack = copy.deepcopy(field_pack)
            pack["field"]["phi"] = pack["field"]["phi"].astype(np.float32)
            return {"0": pack}

        results: Dict[str, Dict] = {}
        for steps in steps_list:
            phi_re = self.reinitializer.reinitialize(phi0, h, steps)
            metrics = ReinitQualityEvaluator.evaluate(phi_re, h)
            new_pack = copy.deepcopy(field_pack)
            new_pack["field"]["phi"] = phi_re.astype(np.float32)
            new_pack["field"]["phi_type"] += f"_reinit{steps}"
            new_pack["meta"]["reinit"] = {
                "steps": steps,
                "scheme": (
                    f"space_order={self.space_order} + time_order={self.time_order} + "
                    f"sign_mode={self.sign_mode} + Godunov (Rouy-Tourin)"
                ),
                "metrics_near_interface": metrics,
            }
            results[str(steps)] = new_pack
        return results


# ═══════════════════════════════════════════════════════════════════════════
# § 4  HDF5 dataset compiler
# ═══════════════════════════════════════════════════════════════════════════

class HDF5DatasetCompiler:
    """Extract 3×3 stencils from field_packs and write into an HDF5 db."""

    def __init__(self, h5_filepath: str, mode: str = "w"):
        self.filepath = h5_filepath
        self.file = h5py.File(self.filepath, mode)
        if mode == "w":
            self.X_dset = self.file.create_dataset("X", shape=(0, 9), maxshape=(None, 9), dtype=np.float32, compression="gzip")
            self.Y_dset = self.file.create_dataset("Y", shape=(0, 1), maxshape=(None, 1), dtype=np.float32, compression="gzip")
            self.Steps_dset = self.file.create_dataset("reinit_steps", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            self.Blueprint_dset = self.file.create_dataset("blueprint_idx", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")
            self.Radius_dset = self.file.create_dataset("radius_idx", shape=(0, 1), maxshape=(None, 1), dtype=np.int32, compression="gzip")

    @staticmethod
    def _extract_stencils(field_pack: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        phi = field_pack["field"]["phi"]
        h_kappa = float(field_pack["label"]["h_kappa"])
        I, J = ReinitQualityEvaluator.get_sampling_coordinates(phi)
        if len(I) == 0:
            return np.zeros((0, 9), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
        indices = np.column_stack((I, J)).astype(np.int64, copy=False)
        patches = extract_3x3_stencils(phi, indices, dtype=np.float32)
        X_b = patches.astype(np.float32)
        Y_b = np.full((X_b.shape[0], 1), h_kappa, dtype=np.float32)
        return np.vstack([X_b, -X_b]), np.vstack([Y_b, -Y_b])

    def append_data(self, field_packs: List[Dict[str, Any]]):
        X_list, Y_list, steps_list, bp_list, rad_list = [], [], [], [], []
        for pack in field_packs:
            X_b, Y_b = self._extract_stencils(pack)
            if X_b.shape[0] == 0:
                continue
            meta = pack["meta"]
            phi_type = pack["field"].get("phi_type", "")
            is_sdf = "sdf" in phi_type and "nonsdf" not in phi_type
            if is_sdf:
                steps = 0
            else:
                reinfo = meta.get("reinit")
                if reinfo is None or "steps" not in reinfo:
                    raise KeyError(f"Non-SDF pack (phi_type: {phi_type}) missing reinit steps.")
                steps = int(reinfo["steps"])
            bp_idx = int(meta["blueprint_idx"])
            rad_idx = int(meta["radius_idx"])
            n_b = X_b.shape[0]
            X_list.append(X_b)
            Y_list.append(Y_b)
            steps_list.append(np.full((n_b, 1), steps, dtype=np.int32))
            bp_list.append(np.full((n_b, 1), bp_idx, dtype=np.int32))
            rad_list.append(np.full((n_b, 1), rad_idx, dtype=np.int32))

        if not X_list:
            return
        X_all = np.vstack(X_list)
        Y_all = np.vstack(Y_list)
        curr = self.X_dset.shape[0]
        new = curr + X_all.shape[0]
        for dset in (self.X_dset, self.Y_dset, self.Steps_dset, self.Blueprint_dset, self.Radius_dset):
            dset.resize((new, dset.shape[1]))
        self.X_dset[curr:new] = X_all
        self.Y_dset[curr:new] = Y_all
        self.Steps_dset[curr:new] = np.vstack(steps_list)
        self.Blueprint_dset[curr:new] = np.vstack(bp_list)
        self.Radius_dset[curr:new] = np.vstack(rad_list)

    def verify_final(self) -> None:
        self.file.flush()
        n = self.X_dset.shape[0]
        if n == 0:
            print("[verify_final] WARNING: file is empty (0 samples).")
            return
        check_n = min(10_000, n)
        rng = np.random.default_rng(seed=0)
        idx = np.sort(rng.choice(n, check_n, replace=False))
        bp = np.array(self.Blueprint_dset[idx, 0], dtype=np.int32)
        rad = np.array(self.Radius_dset[idx, 0], dtype=np.int32)
        stp = np.array(self.Steps_dset[idx, 0], dtype=np.int32)
        errors = []
        if np.any(bp < 0):
            errors.append(f"blueprint_idx: {int((bp < 0).sum())} negative values")
        if np.any(rad < 0):
            errors.append(f"radius_idx: {int((rad < 0).sum())} negative values")
        if errors:
            raise RuntimeError(f"[verify_final] FAILED: {'; '.join(errors)}")
        print(f"[verify_final] OK — {n:,} samples, {check_n} checked.")

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


# ═══════════════════════════════════════════════════════════════════════════
# § 5  Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_curvature_dataset(h5_filepath: str, rho: int = 256):
    h = 1.0 / (rho - 1)
    passed = 0
    try:
        with h5py.File(h5_filepath, 'r') as f:
            X = f['X'][:]
            Y = f['Y'][:]
            steps = f['reinit_steps'][:]
            blueprint_idx = f['blueprint_idx'][:]
            radius_idx = f['radius_idx'][:]

            total = X.shape[0]
            print(f" [1/6] Dimensions: {total:,} samples. ", end="")
            assert X.shape[1] == 9 and Y.shape[1] == 1
            passed += 1

            print("[2/6] NaN/Inf. ", end="")
            assert not np.isnan(X).any() and not np.isnan(Y).any()
            assert not np.isinf(X).any() and not np.isinf(Y).any()
            passed += 1

            print("[3/6] Meta. ", end="")
            assert np.all(blueprint_idx[:, 0] >= 0) and np.all(radius_idx[:, 0] >= 0)
            passed += 1

            print("[4/6] Symmetry. ", end="")
            assert np.abs(np.mean(X)) < 1e-6 and np.abs(np.mean(Y)) < 1e-6
            passed += 1

            print("[5/6] Bounds. ", end="")
            r_min = 1.6 * h
            r_max = 0.5 - 2.0 * h
            hk_max = h / r_min
            y_abs_max = float(np.max(np.abs(Y)))
            assert y_abs_max <= hk_max * 1.01
            passed += 1

            print("[6/6] Eikonal. ", end="")
            sdf_mask = steps[:, 0] == 0
            X_sdf = X[sdf_mask]
            if len(X_sdf) > 0:
                idx = np.random.choice(len(X_sdf), min(1000, len(X_sdf)), replace=False)
                sX = X_sdf[idx]
                grad = np.sqrt(((sX[:, 5] - sX[:, 3]) / (2*h))**2 + ((sX[:, 1] - sX[:, 7]) / (2*h))**2)
                assert 0.90 < float(np.mean(grad)) < 1.10
            passed += 1

        print(f"\n  All 6/6 checks PASSED ✓  {h5_filepath}  (rho={rho})")
    except Exception as e:
        print(f"\n  [FAIL] Check {passed+1}/6: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# § 6  Pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def _generate_for_resolution(
    rho: int,
    h5_filename: str,
    *,
    config: GenerateConfig,
) -> None:
    td = config.train_data
    if os.path.exists(h5_filename):
        os.remove(h5_filename)

    generator = CircleGeometryGenerator(
        resolution_rho=rho, seed=td.geometry_seed, variations=td.variations,
    )
    field_builder = LevelSetFieldBuilder(dtype=np.float64)
    reinit_builder = ReinitFieldPackBuilder(
        cfl=td.cfl,
        sign_mode=td.sign_mode,
        time_order=td.time_order,
        space_order=td.space_order,
    )
    reinit_steps = list(td.reinit_steps)

    print(f"\n>>> [rho={rho}] space_order={td.space_order} time_order={td.time_order}")
    compiler = HDF5DatasetCompiler(h5_filename, mode="w")
    try:
        blueprints = generator.generate_blueprints()
        print(f"    {len(blueprints)} blueprints generated.")
        for bp in tqdm(blueprints, desc=f"rho={rho}"):
            pack_sdf = field_builder.build_circle_sdf(bp, return_grid=False)
            pack_nonsdf = field_builder.build_circle_nonsdf(bp, return_grid=False)
            out_sdf = reinit_builder.build(pack_sdf)
            out_nonsdf = reinit_builder.build(pack_nonsdf, steps_list=reinit_steps)
            batch = [out_sdf["0"]]
            for s in reinit_steps:
                batch.append(out_nonsdf[str(s)])
            compiler.append_data(batch)
        compiler.verify_final()
    finally:
        compiler.close()


def generate_train_datasets(
    target_rhos: list[int] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    *,
    config: GenerateConfig | None = None,
) -> None:
    cfg = config or load_generate_config()
    rhos = list(target_rhos or cfg.train_data.resolutions)
    out = Path(output_dir or cfg.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Launching Train Data Generation Pipeline")
    for rho in rhos:
        db = out / f"train_rho{rho}.h5"
        _generate_for_resolution(rho, str(db), config=cfg)
        validate_curvature_dataset(str(db), rho=rho)

    print("\nFinal summary:")
    for rho in rhos:
        db = out / f"train_rho{rho}.h5"
        if db.exists():
            with h5py.File(db, "r") as f:
                n = f["X"].shape[0]
                steps = np.unique(f["reinit_steps"][:, 0].astype(np.int32)).tolist()
            print(f"  rho={rho}: {n:>9,} samples | steps={steps}")
