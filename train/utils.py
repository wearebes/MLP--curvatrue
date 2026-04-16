from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


class NullLogger:
    """Fallback logger used when tracking is disabled or unavailable."""

    def log(self, *args, **kwargs) -> None:
        pass

    def finish(self, *args, **kwargs) -> None:
        pass


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_log(path: str | Path, msg: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(msg.rstrip("\n") + "\n")


def _to_flat_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    else:
        values = list(values)
    return [float(v) for v in values]


def save_zscore_stats(rho: int, mu: Any, sigma: Any, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"zscore_stats_{rho}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "mu", "sigma"])
        for index, (mu_value, sigma_value) in enumerate(
            zip(_to_flat_list(mu), _to_flat_list(sigma), strict=True)
        ):
            writer.writerow([index, float(mu_value), float(sigma_value)])
    return csv_path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _query_nvidia_smi() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    rows: list[dict[str, str]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mb": parts[3],
                "memory_used_mb": parts[4],
                "utilization_gpu_pct": parts[5],
            }
        )
    return rows


def collect_runtime_context(device: str | None = None) -> dict[str, Any]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CPUS_PER_TASK",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    env = {key: os.environ.get(key, "") for key in keys}
    return {
        "device": device or "",
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "env": env,
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_available": torch.cuda.is_available(),
        "gpu_snapshot": _query_nvidia_smi(),
    }


def build_cuda_diagnostics() -> dict[str, Any]:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "LD_LIBRARY_PATH",
        "PATH",
        "CONDA_DEFAULT_ENV",
        "SLURM_JOB_ID",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
    ]
    env = {key: os.environ.get(key, "") for key in env_keys}
    diagnostics: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "env": env,
        "gpu_snapshot": _query_nvidia_smi(),
    }
    if torch.cuda.is_available():
        try:
            diagnostics["current_device"] = int(torch.cuda.current_device())
            diagnostics["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception as exc:  # pragma: no cover - diagnostic path
            diagnostics["device_query_error"] = f"{type(exc).__name__}: {exc}"
    return diagnostics


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_environment_report(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "node": platform.node(),
        },
        "paths": {
            "cwd": str(Path.cwd()),
            "config_path": str(config_path) if config_path else "",
            "data_dir": str(data_dir) if data_dir else "",
            "model_dir": str(model_dir) if model_dir else "",
        },
        "packages": {
            "torch": getattr(torch, "__version__", None),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "h5py": _package_version("h5py"),
            "pyyaml": _package_version("PyYAML"),
            "swanlab": _package_version("swanlab"),
        },
        "runtime_context": collect_runtime_context(),
        "cuda": build_cuda_diagnostics(),
    }
    return report


def aggregate_dataset_swanlab(
    *,
    tracking_mode: str,
    project: str,
    run_id: str,
    dataset: str,
    batch_size: int,
    requested_rhos: list[int],
    entries: list[dict[str, Any]],
    logdir: Path,
) -> bool:
    if tracking_mode.lower() in {"disabled", "off", "none"}:
        return False

    try:
        import swanlab
    except ImportError:
        print("[Warning] SwanLab library not found. Dataset-level tracking skipped.")
        return False

    tracker = swanlab.init(
        project=project,
        name=run_id,
        logdir=str(logdir),
        mode=tracking_mode,
    )
    try:
        tracker.log(
            {
                "run/requested_rho_count": len(requested_rhos),
                "run/requested_batch_size": int(batch_size),
                "run/success_count": sum(1 for entry in entries if entry.get("status") == "success"),
                "run/failure_count": sum(1 for entry in entries if entry.get("status") != "success"),
            },
            step=0,
        )
        for index, entry in enumerate(entries, start=1):
            status = str(entry.get("status", "unknown"))
            payload: dict[str, Any] = {
                "rho/value": int(entry.get("rho", -1)),
                "rho/order": index,
                "rho/status_code": 1 if status == "success" else 0,
                "rho/batch_size": int(entry.get("batch_size", batch_size)),
            }
            if status == "success":
                payload.update(
                    {
                        "rho/best_epoch": int(entry["best_epoch"]),
                        "rho/best_val_mae": float(entry["best_val_mae"]),
                        "rho/test_mse": float(entry["test_mse"]),
                        "rho/test_mae": float(entry["test_mae"]),
                        "rho/test_max_ae": float(entry["test_max_ae"]),
                        "rho/elapsed_seconds": float(entry["elapsed_seconds"]),
                    }
                )
            tracker.log(payload, step=index)
    finally:
        tracker.finish()
    return True
