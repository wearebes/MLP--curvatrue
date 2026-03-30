"""
train/trainer.py — 训练完整流水线。

合并自：dataloader.py + train.py + worker.py

公共 API
--------
run_training_worker(rho, device_str, config, data_dir, model_dir)
"""
from __future__ import annotations

import copy
import csv
import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig, load_training_config
from .model import build_model_for_rho


# ═══════════════════════════════════════════════════════════════════════════
# § 1  Dataset & DataLoader
# ═══════════════════════════════════════════════════════════════════════════

class HDF5RegressionDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        indices: np.ndarray,
        *,
        mu: np.ndarray,
        sigma: np.ndarray,
        in_memory: bool = True,
    ):
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.in_memory = in_memory
        self._handle: h5py.File | None = None
        if self.in_memory:
            with h5py.File(self.h5_path, "r") as f:
                self.x_data = np.asarray(f["X"][:], dtype=np.float32)
                self.y_data = np.asarray(f["Y"][:], dtype=np.float32)

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
    h5_path: Path, train_indices: np.ndarray, *, feature_dim: int, chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    s = np.zeros(feature_dim, dtype=np.float64)
    ssq = np.zeros(feature_dim, dtype=np.float64)
    n = 0
    with h5py.File(h5_path, "r") as f:
        ds = f["X"]
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


def build_dataloaders(
    h5_path: str | Path,
    batch_size: int,
    *,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, object]]:
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        n = int(f["X"].shape[0])
        fdim = int(f["X"].shape[1])
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    tr_idx, val_idx, te_idx = perm[:train_n], perm[train_n:train_n+val_n], perm[train_n+val_n:]

    mu, sigma, zs = _compute_train_stats(h5_path, np.sort(tr_idx), feature_dim=fdim, chunk_size=65536)
    pin = torch.cuda.is_available()

    def _loader(indices, shuffle):
        return DataLoader(
            HDF5RegressionDataset(h5_path, indices, mu=mu, sigma=sigma, in_memory=True),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=pin, drop_last=False,
        )

    meta: dict[str, object] = {
        "N_total": n, "N_train": train_n, "N_val": val_n, "N_test": n - train_n - val_n,
        "batch_size": batch_size, "mu": torch.from_numpy(mu.copy()),
        "sigma": torch.from_numpy(sigma.copy()), "zero_sigma_features": zs,
    }
    return _loader(tr_idx, True), _loader(val_idx, False), _loader(te_idx, False), meta


# ═══════════════════════════════════════════════════════════════════════════
# § 2  Training loop
# ═══════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_mae = 0.0
    total_count = 0
    max_ae = 0.0
    num_batches = len(loader)
    for batch_idx, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        if pred.shape != yb.shape:
            yb = yb.view_as(pred)
        loss = criterion(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ae = torch.abs(pred - yb)
        max_ae = max(max_ae, torch.max(ae).item())
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_mae += ae.mean().item() * bs
        total_count += bs
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            print(f"  [Train] {batch_idx+1}/{num_batches} batches...", end="\r", flush=True)
    print(" " * 80, end="\r", flush=True)
    mse = total_loss / total_count
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": total_mae / total_count, "max_ae": max_ae}


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_mae = 0.0
    total_count = 0
    max_ae = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb)
        if pred.shape != yb.shape:
            yb = yb.view_as(pred)
        loss = criterion(pred, yb)
        ae = torch.abs(pred - yb)
        max_ae = max(max_ae, torch.max(ae).item())
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_mae += ae.mean().item() * bs
        total_count += bs
    mse = total_loss / total_count
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": total_mae / total_count, "max_ae": max_ae}


def fit_regression_model(
    rho: int,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    *,
    config: TrainingConfig,
    device: torch.device,
    save_path,
    log_message: Callable[[str], None] | None = None,
) -> dict[str, object]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    started_at = time.time()
    stop_epoch = None
    stop_info = None
    h_train, h_val, h_test = [], [], []

    def emit(msg: str) -> None:
        print(msg, flush=True)
        if log_message is not None:
            log_message(msg)

    for epoch in range(1, config.max_epochs + 1):
        tr = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = _evaluate(model, val_loader, criterion, device)
        te = _evaluate(model, test_loader, criterion, device)
        h_train.append(tr["mse"]); h_val.append(va["mse"]); h_test.append(te["mse"])

        improved = va["mae"] < best_val_mae
        if improved:
            best_val_mae = va["mae"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]["lr"]
        emit(
            f"[rho={rho}][Epoch {epoch:03d}/{config.max_epochs}] "
            f"LR={lr:.3e} | "
            f"Train: MSE={tr['mse']:.6e}, MAE={tr['mae']:.6e} | "
            f"Val: MAE={va['mae']:.6e} | "
            f"Test: MSE={te['mse']:.6e}, MAE={te['mae']:.6e}"
            f"{' <- best' if improved else ''}"
        )

        if patience_counter >= config.patience:
            emit(f"[rho={rho}] Early stopping at epoch {epoch} (patience={config.patience}).")
            stop_epoch = epoch
            stop_info = f"Early stopping at epoch {stop_epoch}"
            break

    elapsed = time.time() - started_at
    model.load_state_dict(best_state)
    te = _evaluate(model, test_loader, criterion, device)

    # plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(h_train) + 1)
    plt.plot(epochs, h_train, label="Train MSE", color="blue")
    plt.plot(epochs, h_val, label="Val MSE", color="orange")
    plt.plot(epochs, h_test, label="Test MSE", color="green")
    if best_epoch > 0:
        plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best ({best_epoch})")
    plt.yscale("log"); plt.xlabel("Epochs"); plt.ylabel("MSE (log)")
    plt.title(f"rho={rho}"); plt.legend(); plt.grid(True, which="both", ls="-", alpha=0.2)
    plot_path = Path(save_path).parent / f"training_curves_rho{rho}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close()
    emit(f"[rho={rho}] Training finished in {elapsed:.2f}s | best_epoch={best_epoch} | best_val_mae={best_val_mae:.6e}")

    return {
        "rho": rho, "best_epoch": best_epoch, "best_val_mae": best_val_mae,
        "test_mse": te["mse"], "test_mae": te["mae"], "test_max_ae": te["max_ae"],
        "elapsed_seconds": elapsed, "stop_epoch": stop_epoch, "stop_info": stop_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# § 3  Worker (subprocess entry point)
# ═══════════════════════════════════════════════════════════════════════════

def _append_log(path: str | Path, msg: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip("\n") + "\n")


def _to_flat_list(values) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    else:
        values = list(values)
    return [float(v) for v in values]


def save_zscore_stats(rho: int, mu, sigma, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"zscore_stats_{rho}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "mu", "sigma"])
        for i, (m, s) in enumerate(zip(_to_flat_list(mu), _to_flat_list(sigma), strict=True)):
            writer.writerow([i, float(m), float(s)])
    return csv_path


def run_training_worker(
    rho: int,
    device_str: str,
    config_path: str | None,
    config_overrides: dict[str, object] | None,
    data_dir_str: str,
    model_dir_str: str,
) -> dict:
    device = torch.device(device_str)
    cfg = load_training_config(config_path, overrides=config_overrides)
    data_dir = Path(data_dir_str)
    model_dir = Path(model_dir_str)

    set_all_seeds(cfg.seed)
    data_path = data_dir / f"train_rho{rho}.h5"
    ckpt_path = model_dir / f"model_rho{rho}.pth"
    log_path = model_dir / f"training_rho{rho}.log"

    def emit(msg: str) -> None:
        print(f"[rho={rho}] {msg}")
        _append_log(log_path, msg)

    emit("=" * 72)
    emit(f"Starting training for rho={rho} on device={device}")
    emit("=" * 72)

    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_path, batch_size=cfg.batch_size, seed=cfg.seed, num_workers=0,
    )
    emit(f"Data: total={meta['N_total']} train={meta['N_train']} val={meta['N_val']} test={meta['N_test']}")

    model = build_model_for_rho(rho, cfg.model)
    save_zscore_stats(rho, meta["mu"], meta["sigma"], model_dir)

    result = fit_regression_model(
        rho, model, train_loader, val_loader, test_loader,
        config=cfg, device=device, save_path=ckpt_path,
        log_message=lambda msg: _append_log(log_path, msg),
    )
    return result
