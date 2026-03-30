from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ModelConfig


class CurvatureMLP(nn.Module):
    """Shared MLP used for all rho-specific training runs."""

    def __init__(self, hidden_units: int, *, input_dim: int = 9):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.MLP(inputs)


def init_linear_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model_for_rho(rho: int, model_config: ModelConfig) -> CurvatureMLP:
    model = CurvatureMLP(
        hidden_units=model_config.hidden_dim_for(rho),
        input_dim=model_config.input_dim,
    )
    model.apply(init_linear_weights)
    return model


def load_model_for_rho(
    rho: int,
    model_config: ModelConfig,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> CurvatureMLP:
    model = CurvatureMLP(
        hidden_units=model_config.hidden_dim_for(rho),
        input_dim=model_config.input_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_zscore_stats(rho: int, model_dir: str | Path) -> dict[str, np.ndarray]:
    csv_path = Path(model_dir) / f"zscore_stats_{rho}.csv"
    mu_values: list[float] = []
    sigma_values: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {"feature_index", "mu", "sigma"}
        if set(reader.fieldnames or []) != expected_columns:
            raise ValueError(f"Unexpected z-score stats schema in {csv_path}")
        for row in reader:
            mu_values.append(float(row["mu"]))
            sigma_values.append(float(row["sigma"]))

    return {
        "mu": np.array(mu_values, dtype=np.float32),
        "sigma": np.array(sigma_values, dtype=np.float32),
    }


def load_inference_bundle(
    rho: int,
    model_config: ModelConfig,
    *,
    model_dir: str | Path,
    map_location="cpu",
):
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / f"model_rho{rho}.pth"
    model = load_model_for_rho(rho, model_config, checkpoint_path=checkpoint_path, map_location=map_location)
    stats = load_zscore_stats(rho, model_dir=model_dir)
    return model, stats
