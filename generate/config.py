"""generate 模块配置加载器。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"

TEST_DATA_MODE_FORMULA = "formula_phi0"
TEST_DATA_MODE_FORMULA_PROJECTION_BAND = "formula_phi0_projection_band"
TEST_DATA_MODE_EXACT_SDF = "exact_sdf"
REINIT_SIGN_MODE_FROZEN = "frozen_phi0"
REINIT_SIGN_MODE_DYNAMIC = "dynamic_phi"
REINIT_SIGN_MODES = (
    REINIT_SIGN_MODE_FROZEN,
    REINIT_SIGN_MODE_DYNAMIC,
)
TEST_DATA_MODES = (
    TEST_DATA_MODE_FORMULA,
    TEST_DATA_MODE_FORMULA_PROJECTION_BAND,
    TEST_DATA_MODE_EXACT_SDF,
)


def _normalize_test_data_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or TEST_DATA_MODE_FORMULA).strip()
    if mode not in TEST_DATA_MODES:
        raise ValueError(
            f"Unsupported test_data.mode={mode!r}. "
            f"Expected one of: {', '.join(TEST_DATA_MODES)}"
        )
    return mode


def _normalize_reinit_sign_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or REINIT_SIGN_MODE_FROZEN).strip()
    if mode not in REINIT_SIGN_MODES:
        raise ValueError(
            f"Unsupported reinit sign_mode={mode!r}. "
            f"Expected one of: {', '.join(REINIT_SIGN_MODES)}"
        )
    return mode


# ── dataclass 定义 ──────────────────────────────────────────

@dataclass(frozen=True)
class TrainDataConfig:
    resolutions: tuple[int, ...]
    geometry_seed: int
    variations: int
    cfl: float
    eps_weno: float
    eps_sign_factor: float
    sign_mode: str
    time_order: int
    space_order: int
    reinit_steps: tuple[int, ...]


@dataclass(frozen=True)
class TestScenarioConfig:
    exp_id: str
    blueprint_id: str
    experiment_type: str
    rho_model: int
    L: float
    N: int
    h: float
    a: float
    b: float
    p: int

    def as_dict(self) -> dict[str, object]:
        return {
            "exp_id": self.exp_id,
            "blueprint_id": self.blueprint_id,
            "experiment_type": self.experiment_type,
            "rho_model": self.rho_model,
            "L": self.L, "N": self.N, "h": self.h,
            "a": self.a, "b": self.b, "p": self.p,
        }


@dataclass(frozen=True)
class TestDataConfig:
    mode: str
    cfl: float
    eps_sign_factor: float
    sign_mode: str
    time_order: int
    space_order: int
    formula_projection_band_cells: float
    exact_sdf_method: str
    exact_sdf_mp_dps: int
    exact_sdf_newton_tol: float | None
    exact_sdf_newton_max_iter: int
    test_iters: tuple[int, ...]
    scenarios: tuple[TestScenarioConfig, ...]


@dataclass(frozen=True)
class GenerateConfig:
    data_dir: Path
    train_data: TrainDataConfig
    test_data: TestDataConfig


# ── 加载函数 ────────────────────────────────────────────────

def load_generate_config(config_path: str | Path | None = None) -> GenerateConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _CONFIG_DIR.parent / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    td = raw["train_data"]
    train_data = TrainDataConfig(
        resolutions=tuple(int(r) for r in td["resolutions"]),
        geometry_seed=int(td["geometry_seed"]),
        variations=int(td["variations"]),
        cfl=float(td["cfl"]),
        eps_weno=float(td["eps_weno"]),
        eps_sign_factor=float(td["eps_sign_factor"]),
        sign_mode=_normalize_reinit_sign_mode(td.get("sign_mode", REINIT_SIGN_MODE_FROZEN)),
        time_order=int(td.get("time_order", 3)),
        space_order=int(td.get("space_order", 5)),
        reinit_steps=tuple(int(s) for s in td["reinit_steps"]),
    )

    td2 = raw["test_data"]
    scenarios = tuple(
        TestScenarioConfig(
            exp_id=str(s["exp_id"]),
            blueprint_id=str(s.get("blueprint_id", s["exp_id"])),
            experiment_type=str(s["experiment_type"]),
            rho_model=int(s["rho_model"]),
            L=float(s["L"]),
            N=int(s["N"]),
            h=float(s["h"]),
            a=float(s["a"]),
            b=float(s["b"]),
            p=int(s["p"]),
        )
        for s in td2["scenarios"]
    )
    test_data = TestDataConfig(
        mode=_normalize_test_data_mode(td2.get("mode", TEST_DATA_MODE_FORMULA)),
        cfl=float(td2["cfl"]),
        eps_sign_factor=float(td2.get("eps_sign_factor", 1.0)),
        sign_mode=_normalize_reinit_sign_mode(td2.get("sign_mode", REINIT_SIGN_MODE_FROZEN)),
        time_order=int(td2.get("time_order", 3)),
        space_order=int(td2.get("space_order", 5)),
        formula_projection_band_cells=float(td2.get("formula_projection_band_cells", 2.0)),
        exact_sdf_method=str(td2.get("exact_sdf_method", "high_precision_exact_sdf")),
        exact_sdf_mp_dps=int(td2.get("exact_sdf_mp_dps", 80)),
        exact_sdf_newton_tol=(
            None if td2.get("exact_sdf_newton_tol") is None else float(td2.get("exact_sdf_newton_tol"))
        ),
        exact_sdf_newton_max_iter=int(td2.get("exact_sdf_newton_max_iter", 100)),
        test_iters=tuple(int(i) for i in td2["test_iters"]),
        scenarios=scenarios,
    )

    data_dir_raw = raw.get("data_dir", "data")
    data_dir = Path(data_dir_raw)
    if not data_dir.is_absolute():
        data_dir = _CONFIG_DIR.parent / data_dir

    return GenerateConfig(data_dir=data_dir, train_data=train_data, test_data=test_data)
