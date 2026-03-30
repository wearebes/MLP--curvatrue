"""
generate/test_data.py — 可切换初值模式的测试数据生成流水线。

支持两类初值：
- formula_phi0: 论文 irregular flower 口径的代数花形公式初值
- formula_phi0_projection_band: 仅在界面窄带内用 projection 修正公式初值
- exact_sdf:    牛顿投影构造的 exact SDF 初值

公共 API
--------
generate_test_datasets(output_dir, *, config)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from .pde import LevelSetReinitializer
from .numerics import (
    ReinitQualityEvaluator,
    build_grid,
    build_flower_phi0,
    compute_hkappa,
    find_projection_theta,
    high_precision_exact_sdf,
    hkappa_analytic,
    vectorized_exact_sdf,
)
from .train_data import extract_3x3_stencils
from .config import (
    GenerateConfig,
    TEST_DATA_MODE_EXACT_SDF,
    TEST_DATA_MODE_FORMULA,
    TEST_DATA_MODE_FORMULA_PROJECTION_BAND,
    load_generate_config,
)


# ── 采样坐标 ────────────────────────────────────────────────

def _get_interface_indices(phi: np.ndarray) -> np.ndarray:
    I, J = ReinitQualityEvaluator.get_sampling_coordinates(phi)
    if len(I) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack((I, J)).astype(np.int64)


def _indices_to_xy(indices: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.column_stack((X[indices[:, 0], indices[:, 1]],
                            Y[indices[:, 0], indices[:, 1]]))


# ── I/O helpers ─────────────────────────────────────────────

def _save_meta(exp_dir: Path, cfg: dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(dict(cfg), fh, indent=4)


def _save_iter_h5(exp_dir: Path, n_iter: int, payload: dict) -> None:
    with h5py.File(exp_dir / f"iter_{n_iter}.h5", "w") as fh:
        for key, val in payload.items():
            fh.create_dataset(key, data=val, compression="gzip")


def _build_meta(
    cfg: dict,
    *,
    mode: str,
    cfl: float,
    eps_sign_factor: float,
    sign_mode: str,
    time_order: int,
    space_order: int,
    formula_projection_band_cells: float,
    exact_sdf_method: str,
    exact_sdf_mp_dps: int,
    exact_sdf_newton_tol: float | None,
    exact_sdf_newton_max_iter: int,
) -> dict:
    payload = dict(cfg)
    payload["rho_eq"] = 1.0 / float(cfg["h"]) + 1.0
    payload["Omega"] = f"[-{cfg['L']}, {cfg['L']}]^2"
    payload["data_source"] = mode
    payload["phi0_mode"] = mode
    if mode == TEST_DATA_MODE_FORMULA:
        payload["phi0_constructor"] = "build_flower_phi0"
    elif mode == TEST_DATA_MODE_FORMULA_PROJECTION_BAND:
        payload["phi0_constructor"] = "build_flower_phi0 + narrow_band_projection"
        payload["phi0_projection_band_cells"] = float(formula_projection_band_cells)
    else:
        payload["phi0_constructor"] = exact_sdf_method
        payload["sdf_constructor"] = exact_sdf_method
        payload["sdf_mp_dps"] = int(exact_sdf_mp_dps)
        payload["sdf_newton_tol"] = exact_sdf_newton_tol
        payload["sdf_newton_max_iter"] = int(exact_sdf_newton_max_iter)
    payload["grid_indexing"] = "xy"
    payload["reinitializer_cfl"] = float(cfl)
    payload["reinitializer_eps_sign_factor"] = float(eps_sign_factor)
    payload["reinitializer_sign_mode"] = str(sign_mode)
    payload["reinitializer_impl"] = (
        f"explicit_reinitializer(space_order={space_order}, time_order={time_order}, sign_mode={sign_mode}, eps_sign_factor={eps_sign_factor})"
    )
    payload["reinitializer_space_order"] = int(space_order)
    payload["reinitializer_time_order"] = int(time_order)
    payload["numerical_formula"] = "eq3_standard"
    payload["stencil_encoding"] = "training_order"
    payload["sampling_rule"] = "current_interface_nodes"
    payload["target_rule"] = "analytic_projection_current_nodes"
    return payload


# ── 核心：构建测试载荷 ───────────────────────────────────────

def _build_initial_field(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    h: float,
    a: float,
    b: float,
    p: int,
    mode: str,
    formula_projection_band_cells: float,
    exact_sdf_method: str,
    exact_sdf_mp_dps: int,
    exact_sdf_newton_tol: float | None,
    exact_sdf_newton_max_iter: int,
) -> np.ndarray:
    if mode == TEST_DATA_MODE_FORMULA:
        return build_flower_phi0(X, Y, a, b, p)

    if mode == TEST_DATA_MODE_FORMULA_PROJECTION_BAND:
        phi0 = build_flower_phi0(X, Y, a, b, p)
        band_mask = np.abs(phi0) <= float(formula_projection_band_cells) * h
        rows, cols = np.where(band_mask)
        if rows.size == 0:
            return phi0

        xy = np.column_stack((X[rows, cols], Y[rows, cols]))
        theta_proj = find_projection_theta(xy, a, b, p)
        radius = b + a * np.cos(p * theta_proj)
        curve_x = radius * np.cos(theta_proj)
        curve_y = radius * np.sin(theta_proj)
        distance = np.sqrt((xy[:, 0] - curve_x) ** 2 + (xy[:, 1] - curve_y) ** 2)
        corrected = phi0.copy()
        corrected[rows, cols] = np.sign(phi0[rows, cols]) * distance
        return corrected

    if mode != TEST_DATA_MODE_EXACT_SDF:
        raise ValueError(f"Unsupported test-data mode: {mode!r}")

    if exact_sdf_method == "vectorized_exact_sdf":
        return vectorized_exact_sdf(
            X,
            Y,
            a,
            b,
            p,
            tol=1e-11 if exact_sdf_newton_tol is None else float(exact_sdf_newton_tol),
            max_iter=exact_sdf_newton_max_iter,
        )
    if exact_sdf_method == "high_precision_exact_sdf":
        return high_precision_exact_sdf(
            X,
            Y,
            a,
            b,
            p,
            dps=exact_sdf_mp_dps,
            tol=exact_sdf_newton_tol,
            max_iter=exact_sdf_newton_max_iter,
        )
    raise ValueError(
        "Unsupported exact_sdf_method="
        f"{exact_sdf_method!r}. Expected 'vectorized_exact_sdf' or 'high_precision_exact_sdf'."
    )


def _build_test_payloads(
    cfg: dict,
    *,
    mode: str,
    cfl: float,
    eps_sign_factor: float,
    sign_mode: str,
    time_order: int,
    space_order: int,
    formula_projection_band_cells: float,
    exact_sdf_method: str,
    exact_sdf_mp_dps: int,
    exact_sdf_newton_tol: float | None,
    exact_sdf_newton_max_iter: int,
    test_iters: list[int],
    ) -> dict[int, dict[str, np.ndarray]]:
    X, Y, h = build_grid(float(cfg["L"]), int(cfg["N"]), indexing="xy")
    reinit = LevelSetReinitializer(
        indexing="xy",
        cfl=cfl,
        eps_sign_factor=eps_sign_factor,
        sign_mode=sign_mode,
        time_order=time_order,
        space_order=space_order,
    )
    a, b, p = float(cfg["a"]), float(cfg["b"]), int(cfg["p"])
    phi0 = _build_initial_field(
        X,
        Y,
        h=h,
        a=a,
        b=b,
        p=p,
        mode=mode,
        formula_projection_band_cells=formula_projection_band_cells,
        exact_sdf_method=exact_sdf_method,
        exact_sdf_mp_dps=exact_sdf_mp_dps,
        exact_sdf_newton_tol=exact_sdf_newton_tol,
        exact_sdf_newton_max_iter=exact_sdf_newton_max_iter,
    )
    payloads: dict[int, dict[str, np.ndarray]] = {}

    for n_iter in test_iters:
        phi = reinit.reinitialize(phi0, h, n_iter)
        indices = _get_interface_indices(phi)
        xy = _indices_to_xy(indices, X, Y) if len(indices) else np.zeros((0, 2))
        theta_p = find_projection_theta(xy, a, b, p) if len(indices) else np.zeros((0,))
        hk_tgt = hkappa_analytic(theta_p, h, a, b, p) if len(indices) else np.zeros((0,))
        hk_fd = compute_hkappa(phi, indices, h, indexing="xy")
        stencils = extract_3x3_stencils(phi, indices)

        payloads[n_iter] = {
            "indices": indices,
            "xy": xy,
            "theta_proj": theta_p,
            "stencils_raw": stencils,
            "hkappa_target": hk_tgt,
            "hkappa_fd": hk_fd,
        }
    return payloads


# ── 公共入口 ────────────────────────────────────────────────

def generate_test_datasets(
    output_dir: Union[str, Path],
    *,
    config: GenerateConfig | None = None,
    mode_override: str | None = None,
) -> None:
    cfg = config or load_generate_config()
    td = cfg.test_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_iters = list(td.test_iters)
    scenarios = [s.as_dict() for s in td.scenarios]
    mode = mode_override or td.mode

    banner = (
        "[generate test] "
        f"mode={mode} | cfl={td.cfl} | "
        f"eps_sign_factor={td.eps_sign_factor} | "
        f"time_order={td.time_order} | space_order={td.space_order} | sign_mode={td.sign_mode}"
    )
    if mode == TEST_DATA_MODE_EXACT_SDF:
        banner += (
            f" | exact_sdf_method={td.exact_sdf_method}"
            f" | mp_dps={td.exact_sdf_mp_dps}"
            f" | newton_max_iter={td.exact_sdf_newton_max_iter}"
        )
    elif mode == TEST_DATA_MODE_FORMULA_PROJECTION_BAND:
        banner += f" | band_cells={td.formula_projection_band_cells}"
    print(banner)

    for scenario_cfg in scenarios:
        exp_id = scenario_cfg["exp_id"]
        exp_dir = output_dir / exp_id
        print(f"\n  Experiment: {exp_id}")

        _save_meta(
            exp_dir,
            _build_meta(
                scenario_cfg,
                mode=mode,
                cfl=td.cfl,
                eps_sign_factor=td.eps_sign_factor,
                sign_mode=td.sign_mode,
                time_order=td.time_order,
                space_order=td.space_order,
                formula_projection_band_cells=td.formula_projection_band_cells,
                exact_sdf_method=td.exact_sdf_method,
                exact_sdf_mp_dps=td.exact_sdf_mp_dps,
                exact_sdf_newton_tol=td.exact_sdf_newton_tol,
                exact_sdf_newton_max_iter=td.exact_sdf_newton_max_iter,
            ),
        )
        payloads = _build_test_payloads(
                scenario_cfg,
                mode=mode,
                cfl=td.cfl,
                eps_sign_factor=td.eps_sign_factor,
                sign_mode=td.sign_mode,
                time_order=td.time_order,
                space_order=td.space_order,
            formula_projection_band_cells=td.formula_projection_band_cells,
            exact_sdf_method=td.exact_sdf_method,
            exact_sdf_mp_dps=td.exact_sdf_mp_dps,
            exact_sdf_newton_tol=td.exact_sdf_newton_tol,
            exact_sdf_newton_max_iter=td.exact_sdf_newton_max_iter,
            test_iters=test_iters,
        )

        for n_iter in test_iters:
            p = payloads[n_iter]
            _save_iter_h5(exp_dir, n_iter, p)
            M = p["stencils_raw"].shape[0]
            print(f"    iter={n_iter:>3d}  M={M:>6d}  "
                  f"hk_target∈[{p['hkappa_target'].min():.4f}, {p['hkappa_target'].max():.4f}]  "
                  f"hk_fd∈[{p['hkappa_fd'].min():.4f}, {p['hkappa_fd'].max():.4f}]")

    print(f"\n[generate test] Done. Output: {output_dir}")
