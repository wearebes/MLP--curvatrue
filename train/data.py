from __future__ import annotations

import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

VALID_DATA_LOADING_MODES = {"stream", "in_memory"}

class HDF5RegressionDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        indices: np.ndarray,
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        in_memory: bool = False,
    ):
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.in_memory = in_memory
        self._handle: h5py.File | None = None
        if self.in_memory:
            with h5py.File(self.h5_path, "r") as handle:
                self.x_data = np.asarray(handle["X"][:], dtype=np.float32)
                self.y_data = np.asarray(handle["Y"][:], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        if self.in_memory:
            x = self.x_data[idx].copy()
            y = self.y_data[idx].copy()
        else:
            handle = self._require_handle()
            x = np.asarray(handle["X"][idx], dtype=np.float32)
            y = np.asarray(handle["Y"][idx], dtype=np.float32)
        x = (x - self.mu) / self.sigma
        if y.ndim == 0:
            y = np.asarray([y], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def _require_handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()

def _compute_train_stats(
    h5_path: Path,
    train_indices: np.ndarray,
    *,
    feature_dim: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    s = np.zeros(feature_dim, dtype=np.float64)
    ssq = np.zeros(feature_dim, dtype=np.float64)
    n = 0
    with h5py.File(h5_path, "r") as handle:
        ds = handle["X"]
        for start in range(0, train_indices.shape[0], chunk_size):
            chunk = np.asarray(ds[train_indices[start:start + chunk_size]], dtype=np.float64)
            s += chunk.sum(axis=0)
            ssq += np.square(chunk).sum(axis=0)
            n += int(chunk.shape[0])
    mu = s / n
    sigma = np.sqrt(np.maximum(ssq / n - np.square(mu), 0.0))
    zero = sigma == 0.0
    sigma = sigma.astype(np.float32, copy=False)
    sigma[zero] = 1.0
    return mu.astype(np.float32, copy=False), sigma, int(zero.sum())


def resolve_data_loading_mode(mode: str) -> dict[str, object]:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_DATA_LOADING_MODES:
        raise ValueError(
            f"Unsupported data loading mode: {mode}. "
            f"Expected one of {sorted(VALID_DATA_LOADING_MODES)}"
        )
    return {
        "mode": normalized,
        "in_memory": normalized == "in_memory",
        "description": (
            "in_memory=True (dataset cached in RAM)"
            if normalized == "in_memory"
            else "in_memory=False (low-memory HDF5 streaming mode)"
        ),
    }


def resolve_pin_memory(pin_memory: str | bool | None) -> bool:
    if isinstance(pin_memory, bool):
        return pin_memory
    normalized = str(pin_memory or "auto").strip().lower()
    if normalized == "auto":
        return torch.cuda.is_available()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(
        f"Unsupported pin_memory value: {pin_memory}. Expected auto/true/false."
    )

def inspect_training_h5(
    h5_path: str | Path,
    *,
    expected_input_dim: int | None = None,
) -> dict[str, object]:
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    with h5py.File(path, "r") as handle:
        missing = [name for name in ("X", "Y") if name not in handle]
        if missing:
            raise KeyError(f"Training data file {path} is missing dataset(s): {', '.join(missing)}")

        x_ds = handle["X"]
        y_ds = handle["Y"]
        if x_ds.ndim != 2:
            raise ValueError(f"Expected X to be rank-2 in {path}, got shape {x_ds.shape}")
        if y_ds.ndim < 1:
            raise ValueError(f"Expected Y to be at least rank-1 in {path}, got shape {y_ds.shape}")

        total_samples = int(x_ds.shape[0])
        feature_dim = int(x_ds.shape[1])
        target_samples = int(y_ds.shape[0])
        target_shape = tuple(int(v) for v in y_ds.shape)

    if total_samples <= 0:
        raise ValueError(f"Training data file {path} contains no samples.")
    if total_samples != target_samples:
        raise ValueError(
            f"Training data file {path} has mismatched sample counts: "
            f"X has {total_samples}, Y has {target_samples}"
        )
    if expected_input_dim is not None and feature_dim != expected_input_dim:
        raise ValueError(
            f"Training data file {path} has feature_dim={feature_dim}, expected {expected_input_dim}"
        )

    return {
        "total_samples": total_samples,
        "feature_dim": feature_dim,
        "target_shape": target_shape,
    }


def validate_training_dataset(
    dataset_dir: str | Path,
    rho_values: list[int],
    *,
    expected_input_dim: int | None = None,
) -> dict[str, object]:
    dataset_path = Path(dataset_dir)
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for rho in rho_values:
        data_path = dataset_path / f"train_rho{rho}.h5"
        try:
            info = inspect_training_h5(data_path, expected_input_dim=expected_input_dim)
            results.append(
                {
                    "rho": int(rho),
                    "path": str(data_path),
                    "status": "ok",
                    **info,
                }
            )
        except Exception as exc:
            failure = {
                "rho": int(rho),
                "path": str(data_path),
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            results.append(failure)
            failures.append(failure)
    return {
        "dataset_dir": str(dataset_path),
        "requested_rhos": [int(rho) for rho in rho_values],
        "ok_count": sum(1 for row in results if row["status"] == "ok"),
        "failure_count": len(failures),
        "entries": results,
    }

def build_dataloaders(
    h5_path: str | Path,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 0,
    data_loading_mode: str = "stream",
    stats_chunk_size: int = 65536,
    pin_memory: str | bool = "auto",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    h5_path = Path(h5_path)
    h5_info = inspect_training_h5(h5_path)
    n = int(h5_info["total_samples"])
    feature_dim = int(h5_info["feature_dim"])
    resolved_mode = resolve_data_loading_mode(data_loading_mode)
    resolved_pin_memory = resolve_pin_memory(pin_memory)
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    tr_idx = perm[:train_n]
    val_idx = perm[train_n:train_n + val_n]
    te_idx = perm[train_n + val_n:]

    mu, sigma, zero_sigma_features = _compute_train_stats(
        h5_path,
        np.sort(tr_idx),
        feature_dim=feature_dim,
        chunk_size=stats_chunk_size,
    )

    def _loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            HDF5RegressionDataset(
                h5_path,
                indices,
                mu=mu,
                sigma=sigma,
                in_memory=bool(resolved_mode["in_memory"]),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=resolved_pin_memory,
            drop_last=False,
        )

    meta: dict[str, object] = {
        "N_total": n,
        "N_train": train_n,
        "N_val": val_n,
        "N_test": n - train_n - val_n,
        "batch_size": batch_size,
        "mu": torch.from_numpy(mu.copy()),
        "sigma": torch.from_numpy(sigma.copy()),
        "zero_sigma_features": zero_sigma_features,
        "feature_dim": feature_dim,
        "target_shape": h5_info["target_shape"],
        "data_loading_mode": str(resolved_mode["mode"]),
        "data_loading_description": str(resolved_mode["description"]),
        "num_workers": int(num_workers),
        "stats_chunk_size": int(stats_chunk_size),
        "pin_memory": bool(resolved_pin_memory),
    }
    return _loader(tr_idx, True), _loader(val_idx, False), _loader(te_idx, False), meta
