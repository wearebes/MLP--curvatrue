from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import ModelConfig


class CurvatureMLP(nn.Module):
    """Shared MLP used for all rho-specific training runs."""

    def __init__(self, hidden_units: int):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9, hidden_units),
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
        nn.init.normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model_for_rho(rho: int, model_config: ModelConfig) -> CurvatureMLP:
    model = CurvatureMLP(hidden_units=model_config.hidden_dim_for(rho))
    model.apply(init_linear_weights)
    return model


def load_model_for_rho(
    rho: int,
    model_config: ModelConfig,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> CurvatureMLP:
    model = CurvatureMLP(hidden_units=model_config.hidden_dim_for(rho))
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
