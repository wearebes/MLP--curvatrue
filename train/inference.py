from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .model_structure import load_model_for_rho


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
