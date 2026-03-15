from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    max_epochs: int
    patience: int
    batch_size: int
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: dict[int, int]

    def hidden_dim_for(self, rho: int) -> int:
        try:
            return self.hidden_dims[rho]
        except KeyError as exc:
            supported = ", ".join(str(key) for key in sorted(self.hidden_dims))
            raise KeyError(f"Unsupported rho={rho}. Supported values: {supported}") from exc


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: OptimizerConfig
    training: TrainingRuntimeConfig
    model: ModelConfig


def load_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    raw_values: dict[str, str] = {}

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid config line {line_number}: {raw_line}")
        key, value = stripped.split("=", 1)
        raw_values[key.strip()] = value.strip()

    required_keys = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "max_epochs",
        "patience",
        "batch_size",
        "seed",
        "hidden_dim_256",
        "hidden_dim_266",
        "hidden_dim_276",
    }
    missing = sorted(required_keys - raw_values.keys())
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    optimizer = OptimizerConfig(
        lr=float(raw_values["lr"]),
        betas=_parse_betas(raw_values["betas"]),
        eps=float(raw_values["eps"]),
        weight_decay=float(raw_values["weight_decay"]),
    )
    
    training = TrainingRuntimeConfig(
        max_epochs=int(raw_values["max_epochs"]),
        patience=int(raw_values["patience"]),
        batch_size=int(raw_values["batch_size"]),
        seed=int(raw_values["seed"]),
    )

    model = ModelConfig(
        hidden_dims={
            256: int(raw_values["hidden_dim_256"]),
            266: int(raw_values["hidden_dim_266"]),
            276: int(raw_values["hidden_dim_276"]),
        }
    )
    return TrainingConfig(optimizer=optimizer, training=training, model=model)


def _parse_betas(raw_value: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"betas must contain exactly two comma-separated values, got: {raw_value}")
    return float(parts[0]), float(parts[1])
