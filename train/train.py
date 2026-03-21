from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from .config import OptimizerConfig, TrainingRuntimeConfig


#set the random seed for reproducibility across random, numpy, and torch ( CPU and CUDA).
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module, #through the defined model in model_structure
    loader,
    optimizer: torch.optim.Optimizer,  #adam
    criterion: nn.Module,     # the forward
    device: torch.device,
) -> dict[str, float]:    # find the model for different rho
    model.train()   # start train

    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    max_ae = 0.0

<<<<<<< Updated upstream
    # to device cuda, and non_blocking=True means that the data transfer can be 
    # asynchronous with respect to the host, which can improve performance when using pinned memory.
=======

>>>>>>> Stashed changes
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        predictions = model(x_batch)
        if predictions.shape != y_batch.shape:
            y_batch = y_batch.view_as(predictions)

        loss = criterion(predictions, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_ae = torch.abs(predictions - y_batch)
        max_ae = max(max_ae, torch.max(batch_ae).item())

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_mae += batch_ae.mean().item() * batch_size
        total_count += batch_size

    mse = total_loss / total_count
    rmse = math.sqrt(mse)
    mae = total_mae / total_count
    return {"mse": mse, "rmse": rmse, "mae": mae, "max_ae": max_ae}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    max_ae = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        predictions = model(x_batch)
        if predictions.shape != y_batch.shape:
            y_batch = y_batch.view_as(predictions)

        loss = criterion(predictions, y_batch)
        batch_ae = torch.abs(predictions - y_batch)
        max_ae = max(max_ae, torch.max(batch_ae).item())

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_mae += batch_ae.mean().item() * batch_size
        total_count += batch_size

    mse = total_loss / total_count
    rmse = math.sqrt(mse)
    mae = total_mae / total_count
    return {"mse": mse, "rmse": rmse, "mae": mae, "max_ae": max_ae}


def fit_regression_model(
    rho: int,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    *,
    optimizer_config: OptimizerConfig,
    training_config: TrainingRuntimeConfig,
    device: torch.device,
    save_path,
    log_message: Callable[[str], None] | None = None,
) -> dict[str, object]:
    model = model.to(device)
    criterion = nn.MSELoss()   #the loss is mse
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config.lr,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
    )

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    started_at = time.time()
    stop_epoch = None
    stop_info = None

    def emit(message: str) -> None:
        print(message, flush=True)
        if log_message is not None:
            log_message(message)

    for epoch in range(1, training_config.max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer,criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        improved = val_metrics["mae"] < best_val_mae
        if improved:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = " <- best" if improved else ""
        current_lr = optimizer.param_groups[0]["lr"]
        emit(
            f"[rho={rho}][Epoch {epoch:03d}/{training_config.max_epochs}] "
            f"LR={current_lr:.3e} | "
            f"Train: MSE={train_metrics['mse']:.6e}, RMSE={train_metrics['rmse']:.6e}, "
            f"MAE={train_metrics['mae']:.6e}, MaxAE={train_metrics['max_ae']:.6e} | "
            f"Val: MSE={val_metrics['mse']:.6e}, RMSE={val_metrics['rmse']:.6e}, "
            f"MAE={val_metrics['mae']:.6e}, MaxAE={val_metrics['max_ae']:.6e} | "
            f"Test: MSE={test_metrics['mse']:.6e}, RMSE={test_metrics['rmse']:.6e}, "
            f"MAE={test_metrics['mae']:.6e}, MaxAE={test_metrics['max_ae']:.6e}"
            f"{marker}"
        )

        if epochs_without_improvement >= training_config.patience:
            emit(
                f"[rho={rho}] Early stopping triggered at epoch {epoch} "
                f"(patience={training_config.patience})."
            )
            stop_epoch = epoch
            stop_info = (
                f"Early stopping at epoch {stop_epoch}: "
                f"LR={current_lr:.3e} | "
                f"Train: MSE={train_metrics['mse']:.6e}, RMSE={train_metrics['rmse']:.6e}, "
                f"MAE={train_metrics['mae']:.6e}, MaxAE={train_metrics['max_ae']:.6e} | "
                f"Val: MSE={val_metrics['mse']:.6e}, RMSE={val_metrics['rmse']:.6e}, "
                f"MAE={val_metrics['mae']:.6e}, MaxAE={val_metrics['max_ae']:.6e} | "
                f"Test: MSE={test_metrics['mse']:.6e}, RMSE={test_metrics['rmse']:.6e}, "
                f"MAE={test_metrics['mae']:.6e}, MaxAE={test_metrics['max_ae']:.6e}"
            )
            break

    elapsed_seconds = time.time() - started_at
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)

    emit(f"[rho={rho}] Training finished in {elapsed_seconds:.2f}s")
    emit(f"[rho={rho}] Best epoch: {best_epoch}")
    emit(f"[rho={rho}] Best val MAE: {best_val_mae:.6e}")
    emit(
        f"[rho={rho}] Test MSE={test_metrics['mse']:.6e} | "
        f"Test MAE={test_metrics['mae']:.6e} | "
        f"Test MaxAE={test_metrics['max_ae']:.6e}"
    )

    return {
        "rho": rho,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_max_ae": test_metrics["max_ae"],
        "elapsed_seconds": elapsed_seconds,
        "stop_epoch": stop_epoch,
        "stop_info": stop_info,
    }
