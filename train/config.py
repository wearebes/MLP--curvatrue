"""train 模块配置加载器。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    default_hidden_dim: int
    hidden_dim_overrides: dict[int, int]

    def hidden_dim_for(self, rho: int) -> int:
        return int(self.hidden_dim_overrides.get(int(rho), self.default_hidden_dim))


@dataclass(frozen=True)
class TrainingConfig:
    resolutions: tuple[int, ...]
    data_dir: Path
    model_dir: Path
    seed: int
    batch_size: int
    max_epochs: int
    patience: int
    optimizer: OptimizerConfig
    model: ModelConfig


def load_training_config(
    config_path: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> TrainingConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _CONFIG_DIR.parent / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    opt = raw["optimizer"]
    betas_raw = opt["betas"]
    optimizer = OptimizerConfig(
        lr=float(opt["lr"]),
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        eps=float(opt["eps"]),
        weight_decay=float(opt["weight_decay"]),
    )

    mdl = raw["model"]
    model = ModelConfig(
        input_dim=int(mdl["input_dim"]),
        default_hidden_dim=int(mdl["default_hidden_dim"]),
        hidden_dim_overrides={int(k): int(v) for k, v in mdl.get("hidden_dim_overrides", {}).items()},
    )

    data_dir = Path(raw.get("data_dir", "data"))
    model_dir = Path(raw.get("model_dir", "model"))
    if not data_dir.is_absolute():
        data_dir = _CONFIG_DIR.parent / data_dir
    if not model_dir.is_absolute():
        model_dir = _CONFIG_DIR.parent / model_dir

    config = TrainingConfig(
        resolutions=tuple(int(r) for r in raw["resolutions"]),
        data_dir=data_dir,
        model_dir=model_dir,
        seed=int(raw["seed"]),
        batch_size=int(raw["batch_size"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        optimizer=optimizer,
        model=model,
    )
    if not overrides:
        return config

    normalized: dict[str, object] = {}
    if "batch_size" in overrides and overrides["batch_size"] is not None:
        normalized["batch_size"] = int(overrides["batch_size"])
    if not normalized:
        return config
    return replace(config, **normalized)
