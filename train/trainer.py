"""
train/trainer.py - unified training pipeline.
"""
from __future__ import annotations

import copy
import math
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG_PATH, TrainingConfig, load_training_config
from .data import build_dataloaders, inspect_training_h5
from .model import build_model_for_rho
from .utils import (
    NullLogger,
    _append_log,
    collect_runtime_context,
    save_zscore_stats,
    set_all_seeds,
    write_json,
)

RESOURCE_FAILURE_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


def infer_failure_category(exc: BaseException) -> str:
    exc_type = type(exc).__name__.lower()
    message = str(exc).lower()
    combined = f"{exc_type}: {message}"
    if any(pattern in combined for pattern in RESOURCE_FAILURE_PATTERNS):
        return "resource_oom"
    if "cuda is required for training but is unavailable" in combined:
        return "cuda_unavailable"
    if "cuda" in combined or "cublas" in combined or "cudnn" in combined:
        return "cuda_runtime"
    if exc_type in {"filenotfounderror", "keyerror", "valueerror"}:
        return "input_data"
    return "worker_failure"


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_mae = 0.0
    total_count = 0
    max_ae = 0.0
    num_batches = len(loader)
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        if pred.shape != yb.shape:
            yb = yb.view_as(pred)
        loss = criterion(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ae = torch.abs(pred - yb)
        max_ae = max(max_ae, torch.max(ae).item())
        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_mae += ae.mean().item() * batch_size
        total_count += batch_size
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            print(f"  [Train] {batch_idx + 1}/{num_batches} batches...", end="\r", flush=True)
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
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        if pred.shape != yb.shape:
            yb = yb.view_as(pred)
        loss = criterion(pred, yb)
        ae = torch.abs(pred - yb)
        max_ae = max(max_ae, torch.max(ae).item())
        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_mae += ae.mean().item() * batch_size
        total_count += batch_size
    mse = total_loss / total_count
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": total_mae / total_count, "max_ae": max_ae}


def _build_failure_payload(
    *,
    rho: int,
    dataset: str,
    run_id: str,
    batch_size: int,
    runtime_context: dict[str, Any],
    started_at: datetime,
    exc: BaseException,
) -> dict[str, Any]:
    finished_at = datetime.now()
    return {
        "status": "failed",
        "rho": int(rho),
        "dataset": dataset,
        "run_id": run_id,
        "batch_size": int(batch_size),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": max((finished_at - started_at).total_seconds(), 0.0),
        "runtime_context": runtime_context,
        "exception_type": type(exc).__name__,
        "failure_category": infer_failure_category(exc),
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


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
    tracker_logger: Any | None = None,
    log_message: Callable[[str], None] | None = None,
) -> dict[str, object]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    tracker = tracker_logger or NullLogger()

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    started_at = time.time()
    stop_epoch = None
    stop_info = None

    def emit(msg: str) -> None:
        print(msg, flush=True)
        if log_message is not None:
            log_message(msg)

    emit(f"[rho={rho}] Early stopping monitors validation MAE only.")
    emit(f"[rho={rho}] Test metrics are logged for diagnostics only.")

    try:
        for epoch in range(1, config.max_epochs + 1):
            tr = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            va = _evaluate(model, val_loader, criterion, device)
            te = _evaluate(model, test_loader, criterion, device)

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

            tracker.log(
                {
                    "train/mse": tr["mse"],
                    "train/mae": tr["mae"],
                    "train/max_ae": tr["max_ae"],
                    "val/mse": va["mse"],
                    "val/mae": va["mae"],
                    "val/max_ae": va["max_ae"],
                    "test/mse": te["mse"],
                    "test/mae": te["mae"],
                    "test/max_ae": te["max_ae"],
                    "optimizer/lr": lr,
                },
                step=epoch,
            )

            if patience_counter >= config.patience:
                emit(f"[rho={rho}] Early stopping at epoch {epoch} (patience={config.patience}).")
                stop_epoch = epoch
                stop_info = f"Early stopping at epoch {stop_epoch}"
                break

        elapsed = time.time() - started_at
        model.load_state_dict(best_state)
        te = _evaluate(model, test_loader, criterion, device)

        emit(
            f"[rho={rho}] Training finished in {elapsed:.2f}s | "
            f"best_epoch={best_epoch} | best_val_mae={best_val_mae:.6e}"
        )
    finally:
        tracker.finish()

    return {
        "status": "success",
        "rho": rho,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mse": te["mse"],
        "test_mae": te["mae"],
        "test_max_ae": te["max_ae"],
        "elapsed_seconds": elapsed,
        "stop_epoch": stop_epoch,
        "stop_info": stop_info,
    }


def run_training_worker(
    rho: int,
    device_str: str,
    config_path: str | None,
    config_overrides: dict[str, object] | None,
    data_dir_str: str,
    model_dir_str: str,
    *,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device_str)
    cfg = load_training_config(config_path, overrides=config_overrides)
    data_dir = Path(data_dir_str)
    model_dir = Path(model_dir_str)
    model_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()

    set_all_seeds(cfg.seed)
    data_path = data_dir / f"train_rho{rho}.h5"
    ckpt_path = model_dir / f"model_rho{rho}.pth"
    log_path = model_dir / f"training_rho{rho}.log"
    dataset_name = data_dir.name
    run_id = model_dir.name
    resolved_config_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH.resolve()
    runtime_context = collect_runtime_context(device=str(device))

    def emit(msg: str) -> None:
        print(f"[rho={rho}] {msg}")
        _append_log(log_path, msg)

    try:
        emit("=" * 72)
        emit(f"Run header | dataset={dataset_name} | run_id={run_id} | rho={rho}")
        emit(f"Resolved config path: {resolved_config_path}")
        emit(f"Resolved data file  : {data_path.resolve()}")
        emit(f"Resolved model dir  : {model_dir.resolve()}")
        emit(f"Device              : {device}")
        emit(f"Batch size          : {cfg.batch_size}")
        emit(f"Seed                : {cfg.seed}")
        emit(f"Runtime context     : {runtime_context}")
        emit(
            "Data loading config : "
            f"mode={cfg.data_loading.mode} "
            f"num_workers={cfg.data_loading.num_workers} "
            f"stats_chunk_size={cfg.data_loading.stats_chunk_size} "
            f"pin_memory={cfg.data_loading.pin_memory}"
        )
        emit("=" * 72)

        h5_info = inspect_training_h5(data_path, expected_input_dim=cfg.model.input_dim)
        emit(
            f"Input validation OK | samples={h5_info['total_samples']} "
            f"feature_dim={h5_info['feature_dim']} target_shape={h5_info['target_shape']}"
        )

        train_loader, val_loader, test_loader, meta = build_dataloaders(
            data_path,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            num_workers=cfg.data_loading.num_workers,
            data_loading_mode=cfg.data_loading.mode,
            stats_chunk_size=cfg.data_loading.stats_chunk_size,
            pin_memory=cfg.data_loading.pin_memory,
        )
        emit(
            f"Data split | total={meta['N_total']} train={meta['N_train']} "
            f"val={meta['N_val']} test={meta['N_test']} "
            f"feature_dim={meta['feature_dim']} zero_sigma_features={meta['zero_sigma_features']} "
            f"mode={meta['data_loading_mode']} num_workers={meta['num_workers']} "
            f"pin_memory={meta['pin_memory']}"
        )

        model = build_model_for_rho(rho, cfg.model)
        save_zscore_stats(rho, meta["mu"], meta["sigma"], model_dir)

        result = fit_regression_model(
            rho,
            model,
            train_loader,
            val_loader,
            test_loader,
            config=cfg,
            device=device,
            save_path=ckpt_path,
            tracker_logger=NullLogger(),
            log_message=lambda msg: _append_log(log_path, msg),
        )
        payload: dict[str, Any] = {
            **result,
            "dataset": dataset_name,
            "run_id": run_id,
            "batch_size": int(cfg.batch_size),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "runtime_context": runtime_context,
        }
        if metrics_path is not None:
            write_json(metrics_path, payload)
        emit(f"SUCCESS footer | rho={rho} | checkpoint={ckpt_path.resolve()}")
        return payload
    except Exception as exc:
        emit("FAILURE footer | training worker raised an exception")
        _append_log(log_path, traceback.format_exc())
        payload = _build_failure_payload(
            rho=rho,
            dataset=dataset_name,
            run_id=run_id,
            batch_size=int(cfg.batch_size),
            runtime_context=runtime_context,
            started_at=started_at,
            exc=exc,
        )
        if metrics_path is not None:
            write_json(metrics_path, payload)
        raise
