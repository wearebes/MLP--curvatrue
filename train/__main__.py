"""
train/__main__.py - ``python -m train`` entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

os.environ["PYTHONUNBUFFERED"] = "1"

import torch

from .config import DEFAULT_CONFIG_PATH, TrainingConfig, TrainingConfigError, load_training_config
from .data import validate_training_dataset
from .trainer import run_training_worker
from .utils import (
    aggregate_dataset_swanlab,
    build_cuda_diagnostics,
    build_environment_report,
    read_json,
    write_json,
)

try:
    import pandas as pd
except ImportError:
    pd = None

RESULT_COLUMNS = [
    "rho",
    "best_epoch",
    "best_val_mae",
    "test_mse",
    "test_mae",
    "test_max_ae",
    "elapsed_seconds",
]

EXPECTED_RHO_ARTIFACTS = (
    "model_rho{rho}.pth",
    "zscore_stats_{rho}.csv",
    "training_rho{rho}.log",
)
RESOURCE_FAILURE_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)
RESOURCE_FAILURE_CATEGORIES = {"resource_oom", "cuda_unavailable", "cuda_runtime", "resource_failure"}


def format_results_table(rows: list[dict[str, object]]) -> str:
    if pd is not None:
        frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        formatters = {
            "best_val_mae": lambda v: f"{float(v):.6e}",
            "test_mse": lambda v: f"{float(v):.6e}",
            "test_mae": lambda v: f"{float(v):.6e}",
            "test_max_ae": lambda v: f"{float(v):.6e}",
            "elapsed_seconds": lambda v: f"{float(v):.2f}",
        }
        return frame.to_string(index=False, formatters=formatters)

    prepared = []
    for row in rows:
        prepared.append(
            {
                "rho": str(row["rho"]),
                "best_epoch": str(row["best_epoch"]),
                "best_val_mae": f"{float(row['best_val_mae']):.6e}",
                "test_mse": f"{float(row['test_mse']):.6e}",
                "test_mae": f"{float(row['test_mae']):.6e}",
                "test_max_ae": f"{float(row['test_max_ae']):.6e}",
                "elapsed_seconds": f"{float(row['elapsed_seconds']):.2f}",
            }
        )
    if not prepared:
        return "No results."
    widths = {c: max(len(c), *(len(r[c]) for r in prepared)) for c in RESULT_COLUMNS}
    lines = [" ".join(c.rjust(widths[c]) for c in RESULT_COLUMNS)]
    for row in prepared:
        lines.append(" ".join(row[c].rjust(widths[c]) for c in RESULT_COLUMNS))
    return "\n".join(lines)


def save_results_to_txt(results: list[dict[str, object]], output_path: str | Path) -> str:
    rows = [{c: r[c] for c in RESULT_COLUMNS} for r in results]
    table = format_results_table(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table + "\n", encoding="utf-8")
    return table


def ensure_output_directory_is_fresh(model_dir: Path) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse existing non-empty run directory: {model_dir}. "
            "Choose a new --run-id before rerunning this training job."
        )
    model_dir.mkdir(parents=True, exist_ok=True)


def write_train_config_snapshot(
    *,
    model_dir: Path,
    run_id: str,
    dataset: str,
    rho_values: list[int],
    batch_size: int,
    max_epochs: int,
    patience: int,
    optimizer_lr: float,
    optimizer_weight_decay: float,
    device: str,
    data_loading: dict[str, object],
    tracking: dict[str, object],
) -> None:
    dump = {
        "run_id": run_id,
        "dataset_source": dataset,
        "trained_at": datetime.now().isoformat(),
        "rho_values": list(rho_values),
        "device": device,
        "training_parameters": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "optimizer": {
                "lr": optimizer_lr,
                "weight_decay": optimizer_weight_decay,
            },
        },
        "data_loading": data_loading,
        "tracking": tracking,
    }
    with (model_dir / "train_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dump, handle, sort_keys=False)


def validate_rho_artifacts(model_dir: Path, rho: int) -> list[str]:
    missing: list[str] = []
    for template in EXPECTED_RHO_ARTIFACTS:
        expected_path = model_dir / template.format(rho=rho)
        if not expected_path.exists():
            missing.append(expected_path.name)
    return missing


def build_failure_record(rho: int, payload: dict[str, Any]) -> dict[str, object]:
    return {
        "rho": int(rho),
        "failure_category": str(payload.get("failure_category", classify_failure_payload(payload))),
        "exception_type": str(payload.get("exception_type", "RuntimeError")),
        "message": str(payload.get("message", "Unknown failure")),
        "traceback": str(payload.get("traceback", "")),
    }


def classify_failure_payload(payload: dict[str, Any]) -> str:
    explicit = payload.get("failure_category")
    if explicit:
        return str(explicit)

    exc_type = str(payload.get("exception_type", "")).lower()
    message = str(payload.get("message", "")).lower()
    combined = f"{exc_type}: {message}"
    if "missing worker metrics file" in combined:
        return "missing_metrics"
    if "missing expected artifacts" in combined:
        return "missing_artifact"
    if any(pattern in combined for pattern in RESOURCE_FAILURE_PATTERNS):
        return "resource_oom"
    if "cuda is required for training but is unavailable" in combined:
        return "cuda_unavailable"
    if "cuda" in combined or "cublas" in combined or "cudnn" in combined:
        return "cuda_runtime"
    if exc_type in {"filenotfounderror", "keyerror", "valueerror"}:
        return "input_data"
    return "worker_failure"


def should_trigger_resource_protection(payload: dict[str, Any]) -> bool:
    return classify_failure_payload(payload) in RESOURCE_FAILURE_CATEGORIES


def write_failure_summaries(
    *,
    model_dir: Path,
    dataset: str,
    run_id: str,
    requested_rhos: list[int],
    successes: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset": dataset,
        "run_id": run_id,
        "requested_rhos": requested_rhos,
        "successful_rhos": [int(row["rho"]) for row in successes],
        "failures": failures,
    }
    (model_dir / "training_failure_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"Training run failed for dataset={dataset} run_id={run_id}",
        f"Generated at: {payload['generated_at']}",
        f"Requested rhos: {requested_rhos}",
        f"Successful rhos: {payload['successful_rhos']}",
        "",
    ]
    for failure in failures:
        lines.append(
            f"rho={failure['rho']} | {failure['failure_category']} | "
            f"{failure['exception_type']}: {failure['message']}"
        )
        tb_lines = str(failure["traceback"]).strip().splitlines()
        if tb_lines:
            lines.append(tb_lines[-1])
        lines.append("")

    (model_dir / "training_failure_summary.txt").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def metrics_path_for(model_dir: Path, rho: int) -> Path:
    return model_dir / f"metrics_rho{rho}.json"


def default_dataset_run_id(dataset: str, batch_size: int) -> str:
    return f"{dataset}_batch{batch_size}"


def choose_run_id(base_run_id: str, model_root: Path) -> str:
    candidate = base_run_id
    model_dir = model_root / candidate
    if model_dir.exists() and any(model_dir.iterdir()):
        suffix = os.environ.get("SLURM_JOB_ID", "local")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = f"{base_run_id}_job{suffix}_{timestamp}"
    return candidate


def validate_dataset_inputs(*, data_dir: Path, rho_values: list[int]) -> None:
    report = validate_training_dataset(data_dir, rho_values)
    failures = [entry for entry in report["entries"] if entry["status"] != "ok"]
    if failures:
        messages = [f"{item['path']} ({item['exception_type']}: {item['message']})" for item in failures]
        raise FileNotFoundError("Dataset validation failed: " + "; ".join(messages))


def build_dataset_validation_report(
    *,
    cfg: TrainingConfig,
    dataset: str,
    rho_values: list[int],
) -> dict[str, object]:
    report = validate_training_dataset(
        cfg.data_dir / dataset,
        rho_values,
        expected_input_dim=cfg.model.input_dim,
    )
    report.update(
        {
            "dataset": dataset,
            "expected_input_dim": int(cfg.model.input_dim),
            "data_loading": {
                "mode": cfg.data_loading.mode,
                "num_workers": int(cfg.data_loading.num_workers),
                "stats_chunk_size": int(cfg.data_loading.stats_chunk_size),
                "pin_memory": cfg.data_loading.pin_memory,
            },
        }
    )
    return report


def build_schedule_preflight_report(
    *,
    config_path: str,
    base_config_overrides: dict[str, object],
    datasets: list[str],
    batch_sizes: list[int],
    rho_values: list[int],
    requested_rho_max_concurrent: int,
    require_cuda: bool,
) -> dict[str, object]:
    if not datasets:
        raise ValueError("--datasets must be non-empty in --preflight-schedule mode.")
    if not batch_sizes:
        raise ValueError("--batch-sizes must be non-empty in --preflight-schedule mode.")
    if requested_rho_max_concurrent < 1:
        raise ValueError("--rho-max-concurrent must be >= 1 in --preflight-schedule mode.")

    base_cfg = load_training_config(config_path, overrides=base_config_overrides)
    env_report = build_environment_report(
        config_path=Path(config_path).resolve(),
        data_dir=base_cfg.data_dir,
        model_dir=base_cfg.model_dir,
    )
    issues: list[str] = []
    warnings: list[str] = []
    batch_reports: list[dict[str, object]] = []

    if require_cuda and not bool(env_report["cuda"]["cuda_available"]):
        issues.append("CUDA is unavailable during schedule preflight.")
    slurm_issue = detect_slurm_gpu_allocation_issue(env_report)
    if slurm_issue:
        issues.append(slurm_issue)
    gpu_memory_report = build_gpu_memory_guard_report(requested_rho_max_concurrent)
    if int(gpu_memory_report["effective_rho_max_concurrent"]) < int(requested_rho_max_concurrent):
        warnings.append(
            "Requested rho concurrency exceeds current GPU free-memory guard. "
            f"requested={requested_rho_max_concurrent} "
            f"effective={gpu_memory_report['effective_rho_max_concurrent']}"
        )

    for batch_size in batch_sizes:
        overrides = dict(base_config_overrides)
        overrides["batch_size"] = int(batch_size)
        cfg_for_batch = load_training_config(config_path, overrides=overrides)
        dataset_reports: list[dict[str, object]] = []
        for dataset in datasets:
            report = build_dataset_validation_report(
                cfg=cfg_for_batch,
                dataset=dataset,
                rho_values=rho_values,
            )
            dataset_reports.append(report)
            if int(report["failure_count"]) > 0:
                issues.append(
                    f"dataset={dataset} batch_size={batch_size} has "
                    f"{report['failure_count']} data validation failure(s)"
                )
        batch_reports.append(
            {
                "batch_size": int(batch_size),
                "data_dir": str(cfg_for_batch.data_dir),
                "model_dir": str(cfg_for_batch.model_dir),
                "datasets": dataset_reports,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "datasets": datasets,
        "batch_sizes": [int(v) for v in batch_sizes],
        "rho_values": [int(v) for v in rho_values],
        "requested_rho_max_concurrent": int(requested_rho_max_concurrent),
        "require_cuda": bool(require_cuda),
        "environment": env_report,
        "gpu_memory_guard": gpu_memory_report,
        "batches": batch_reports,
        "warning_count": len(warnings),
        "warnings": warnings,
        "issue_count": len(issues),
        "issues": issues,
        "status": "ok" if not issues else "failed",
    }


def append_state_line(state_file: Path, message: str) -> None:
    with state_file.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


def _safe_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(str(value))
    except (TypeError, ValueError):
        return None


def detect_slurm_gpu_allocation_issue(env_report: dict[str, object]) -> str | None:
    cuda_report = env_report.get("cuda", {})
    if not isinstance(cuda_report, dict):
        return None
    env = cuda_report.get("env", {})
    if not isinstance(env, dict):
        return None
    slurm_job_id = str(env.get("SLURM_JOB_ID", "")).strip()
    if not slurm_job_id:
        return None
    slurm_job_gpus = str(env.get("SLURM_JOB_GPUS", "")).strip()
    slurm_step_gpus = str(env.get("SLURM_STEP_GPUS", "")).strip()
    if slurm_job_gpus or slurm_step_gpus:
        return None
    gpu_snapshot = cuda_report.get("gpu_snapshot", [])
    if isinstance(gpu_snapshot, list) and gpu_snapshot:
        return (
            "Slurm job is present but neither SLURM_JOB_GPUS nor SLURM_STEP_GPUS is set. "
            "The node exposes GPUs via nvidia-smi, but this step may not actually own a GPU allocation."
        )
    return (
        "Slurm job is present but neither SLURM_JOB_GPUS nor SLURM_STEP_GPUS is set. "
        "GPU allocation looks inconsistent."
    )


def build_gpu_memory_guard_report(requested_rho_max_concurrent: int) -> dict[str, object]:
    diagnostics = build_cuda_diagnostics()
    snapshot = diagnostics.get("gpu_snapshot", [])
    report: dict[str, object] = {
        "requested_rho_max_concurrent": int(requested_rho_max_concurrent),
        "effective_rho_max_concurrent": int(requested_rho_max_concurrent),
        "status": "not_applicable",
        "reason": "",
        "gpu_index": None,
        "memory_total_mb": None,
        "memory_used_mb": None,
        "memory_free_mb": None,
        "min_free_mb_per_worker": _safe_int(os.environ.get("TRAIN_GPU_MIN_FREE_MB_PER_WORKER")) or 3000,
        "free_memory_reserve_mb": _safe_int(os.environ.get("TRAIN_GPU_FREE_MEMORY_RESERVE_MB")) or 1024,
    }
    if int(requested_rho_max_concurrent) <= 1:
        report["status"] = "skipped"
        report["reason"] = "Requested rho concurrency <= 1."
        return report
    if not bool(diagnostics.get("cuda_available")):
        report["status"] = "skipped"
        report["reason"] = "CUDA unavailable; GPU memory guard skipped."
        return report
    if not isinstance(snapshot, list) or not snapshot:
        report["status"] = "skipped"
        report["reason"] = "No GPU snapshot available from nvidia-smi."
        return report

    selected = snapshot[0]
    total_mb = _safe_int(selected.get("memory_total_mb"))
    used_mb = _safe_int(selected.get("memory_used_mb"))
    if total_mb is None or used_mb is None:
        report["status"] = "skipped"
        report["reason"] = "GPU memory values could not be parsed."
        return report

    free_mb = max(total_mb - used_mb, 0)
    min_free_mb_per_worker = int(report["min_free_mb_per_worker"])
    reserve_mb = int(report["free_memory_reserve_mb"])
    available_after_reserve = max(free_mb - reserve_mb, 0)
    supported_workers = max(1, available_after_reserve // max(min_free_mb_per_worker, 1))
    effective = max(1, min(int(requested_rho_max_concurrent), int(supported_workers)))

    report.update(
        {
            "status": "ok" if effective == int(requested_rho_max_concurrent) else "reduced",
            "gpu_index": selected.get("index"),
            "memory_total_mb": total_mb,
            "memory_used_mb": used_mb,
            "memory_free_mb": free_mb,
            "effective_rho_max_concurrent": effective,
            "reason": (
                "Current free GPU memory is below the requested concurrency guard."
                if effective < int(requested_rho_max_concurrent)
                else "Current free GPU memory satisfies the requested concurrency guard."
            ),
        }
    )
    return report


def append_gpu_snapshot(state_file: Path, label: str) -> None:
    payload = {
        "label": label,
        "captured_at": datetime.now().isoformat(),
        "cuda": build_cuda_diagnostics(),
    }
    append_state_line(state_file, json.dumps(payload, ensure_ascii=True))


def detect_resource_failure(model_dir: Path, rho: int) -> bool:
    payload_path = metrics_path_for(model_dir, rho)
    if payload_path.exists():
        payload = read_json(payload_path)
        if should_trigger_resource_protection(payload):
            return True

    log_path = model_dir / f"training_rho{rho}.log"
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(pattern in text for pattern in RESOURCE_FAILURE_PATTERNS)


def launch_worker_process(
    *,
    dataset: str,
    run_id: str,
    rho: int,
    config_path: str,
    config_overrides: dict[str, object],
    allow_cpu: bool,
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "train",
        "--dataset",
        dataset,
        "--run-id",
        run_id,
        "--config",
        config_path,
        "--rho",
        str(rho),
        "--worker-mode",
    ]
    if "batch_size" in config_overrides:
        command.extend(["--batch-size", str(config_overrides["batch_size"])])
    if "max_epochs" in config_overrides:
        command.extend(["--max-epochs", str(config_overrides["max_epochs"])])
    if "patience" in config_overrides:
        command.extend(["--patience", str(config_overrides["patience"])])
    if allow_cpu:
        command.append("--allow-cpu")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(command, env=env)


def prepare_run_directory(
    *,
    cfg: TrainingConfig,
    model_dir: Path,
    run_id: str,
    dataset: str,
    rho_values: list[int],
    device: str,
) -> None:
    ensure_output_directory_is_fresh(model_dir)
    write_train_config_snapshot(
        model_dir=model_dir,
        run_id=run_id,
        dataset=dataset,
        rho_values=rho_values,
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        patience=cfg.patience,
        optimizer_lr=cfg.optimizer.lr,
        optimizer_weight_decay=cfg.optimizer.weight_decay,
        device=device,
        data_loading={
            "mode": cfg.data_loading.mode,
            "num_workers": int(cfg.data_loading.num_workers),
            "stats_chunk_size": int(cfg.data_loading.stats_chunk_size),
            "pin_memory": cfg.data_loading.pin_memory,
        },
        tracking={
            "mode": cfg.tracking.mode,
            "project": cfg.tracking.project,
            "logdir_name": cfg.tracking.logdir_name,
        },
    )


def _load_worker_payloads(
    *,
    model_dir: Path,
    dataset: str,
    run_id: str,
    requested_rhos: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stop_info_list: list[str] = []

    for rho in requested_rhos:
        path = metrics_path_for(model_dir, rho)
        if not path.exists():
            payload = {
                "status": "failed",
                "rho": int(rho),
                "dataset": dataset,
                "run_id": run_id,
                "batch_size": int(batch_size),
                "exception_type": "FileNotFoundError",
                "failure_category": "missing_metrics",
                "message": f"Missing worker metrics file: {path.name}",
                "traceback": "",
            }
        else:
            payload = read_json(path)

        payload.setdefault("rho", int(rho))
        payload.setdefault("dataset", dataset)
        payload.setdefault("run_id", run_id)
        payload.setdefault("batch_size", int(batch_size))
        entries.append(payload)

        if payload.get("status") == "success":
            missing_artifacts = validate_rho_artifacts(model_dir, int(rho))
            if missing_artifacts:
                failure_payload = {
                    "status": "failed",
                    "rho": int(rho),
                    "dataset": dataset,
                    "run_id": run_id,
                    "batch_size": int(batch_size),
                    "exception_type": "RuntimeError",
                    "failure_category": "missing_artifact",
                    "message": "Missing expected artifacts: " + ", ".join(missing_artifacts),
                    "traceback": "",
                }
                write_json(path, failure_payload)
                entries[-1] = failure_payload
                failures.append(build_failure_record(int(rho), failure_payload))
                continue

            row = {column: payload[column] for column in RESULT_COLUMNS}
            result_rows.append(row)
            if payload.get("stop_info"):
                stop_info_list.append(f"[rho={rho}] {payload['stop_info']}")
        else:
            failures.append(build_failure_record(int(rho), payload))

    return entries, result_rows, failures, stop_info_list


def finalize_dataset_run(
    *,
    cfg: TrainingConfig,
    model_dir: Path,
    dataset: str,
    run_id: str,
    requested_rhos: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries, result_rows, failures, stop_info_list = _load_worker_payloads(
        model_dir=model_dir,
        dataset=dataset,
        run_id=run_id,
        requested_rhos=requested_rhos,
        batch_size=batch_size,
    )

    write_json(
        model_dir / "dataset_metrics_summary.json",
        {
            "generated_at": datetime.now().isoformat(),
            "dataset": dataset,
            "run_id": run_id,
            "batch_size": int(batch_size),
            "requested_rhos": requested_rhos,
            "entries": entries,
        },
    )

    if stop_info_list:
        (model_dir / "stop_summary.txt").write_text(
            "\n".join(stop_info_list) + "\n",
            encoding="utf-8",
        )

    if result_rows:
        output_name = "training_summary.txt" if not failures else "training_summary_partial.txt"
        output_path = model_dir / output_name
        save_results_to_txt(result_rows, output_path)

    if failures:
        write_failure_summaries(
            model_dir=model_dir,
            dataset=dataset,
            run_id=run_id,
            requested_rhos=requested_rhos,
            successes=result_rows,
            failures=failures,
        )

    aggregate_dataset_swanlab(
        tracking_mode=cfg.tracking.mode,
        project=cfg.tracking.project,
        run_id=run_id,
        dataset=dataset,
        batch_size=batch_size,
        requested_rhos=requested_rhos,
        entries=entries,
        logdir=model_dir / cfg.tracking.logdir_name,
    )
    return result_rows, failures


def run_dataset_lifecycle(
    *,
    cfg: TrainingConfig,
    dataset: str,
    run_id: str,
    rho_values: list[int],
    rho_max_concurrent: int,
    config_path: str,
    config_overrides: dict[str, object],
    allow_cpu: bool,
    device: str,
) -> dict[str, Any]:
    if rho_max_concurrent < 1:
        raise ValueError("--rho-max-concurrent must be >= 1.")

    gpu_guard_report = build_gpu_memory_guard_report(rho_max_concurrent)
    effective_rho_max_concurrent = int(gpu_guard_report["effective_rho_max_concurrent"])
    if effective_rho_max_concurrent < rho_max_concurrent:
        print(
            f"[dataset={dataset}][run_id={run_id}] reducing rho concurrency from "
            f"{rho_max_concurrent} to {effective_rho_max_concurrent} due to GPU memory guard",
            flush=True,
        )
        rho_max_concurrent = effective_rho_max_concurrent

    data_dir = cfg.data_dir / dataset
    model_dir = cfg.model_dir / run_id
    state_file = model_dir / "dataset_job_state.txt"
    validate_dataset_inputs(data_dir=data_dir, rho_values=rho_values)

    prepare_run_directory(
        cfg=cfg,
        model_dir=model_dir,
        run_id=run_id,
        dataset=dataset,
        rho_values=rho_values,
        device=device,
    )

    started_at = datetime.now().isoformat()
    state_file.write_text(
        "\n".join(
            [
                f"dataset: {dataset}",
                f"run_id: {run_id}",
                f"batch_size: {cfg.batch_size}",
                "requested_rhos: " + " ".join(str(rho) for rho in rho_values),
                f"effective_rho_max_concurrent: {rho_max_concurrent}",
                f"started_at: {started_at}",
                f"job_id: {os.environ.get('SLURM_JOB_ID', 'local')}",
                f"python_executable: {sys.executable}",
                f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
                f"slurm_cpus_per_task: {os.environ.get('SLURM_CPUS_PER_TASK', '<unset>')}",
                "--- worker_states ---",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation_report = build_dataset_validation_report(
        cfg=cfg,
        dataset=dataset,
        rho_values=rho_values,
    )
    append_state_line(
        state_file,
        json.dumps(
            {
                "label": "dataset_validation_report",
                "captured_at": datetime.now().isoformat(),
                "report": validation_report,
            },
            ensure_ascii=True,
        ),
    )
    append_state_line(
        state_file,
        json.dumps(
            {
                "label": "gpu_memory_guard",
                "captured_at": datetime.now().isoformat(),
                "report": gpu_guard_report,
            },
            ensure_ascii=True,
        ),
    )
    append_gpu_snapshot(state_file, "gpu_snapshot_before_dataset")
    print(
        f"[dataset={dataset}][run_id={run_id}] "
        f"starting dataset lifecycle with rho concurrency={rho_max_concurrent}",
        flush=True,
    )

    active_workers: list[dict[str, Any]] = []
    dataset_failed = False
    resource_protection_triggered = False

    def drain_next_worker() -> None:
        nonlocal dataset_failed, resource_protection_triggered
        worker = active_workers.pop(0)
        rho = int(worker["rho"])
        proc = worker["proc"]
        exit_code = proc.wait()
        status_label = "success" if exit_code == 0 else "failed"
        if exit_code != 0:
            dataset_failed = True
            if detect_resource_failure(model_dir, rho):
                resource_protection_triggered = True
        append_state_line(
            state_file,
            (
                f"rho={rho} pid={proc.pid} status={status_label} exit_code={exit_code} "
                f"started_at={worker['started_at']} finished_at={datetime.now().isoformat()}"
            ),
        )

    for rho in rho_values:
        while len(active_workers) >= rho_max_concurrent:
            drain_next_worker()

        proc = launch_worker_process(
            dataset=dataset,
            run_id=run_id,
            rho=int(rho),
            config_path=config_path,
            config_overrides=config_overrides,
            allow_cpu=allow_cpu,
        )
        started_worker_at = datetime.now().isoformat()
        active_workers.append({"rho": int(rho), "proc": proc, "started_at": started_worker_at})
        append_state_line(
            state_file,
            f"rho={rho} pid={proc.pid} status=started exit_code= started_at={started_worker_at}",
        )

    while active_workers:
        drain_next_worker()

    append_gpu_snapshot(state_file, "gpu_snapshot_after_dataset")
    result_rows, failures = finalize_dataset_run(
        cfg=cfg,
        model_dir=model_dir,
        dataset=dataset,
        run_id=run_id,
        requested_rhos=rho_values,
        batch_size=cfg.batch_size,
    )
    if failures:
        dataset_failed = True

    append_state_line(state_file, "--- dataset_summary ---")
    append_state_line(state_file, f"finished_at: {datetime.now().isoformat()}")
    append_state_line(
        state_file,
        "aggregate_status: " + ("success" if not failures else "failed"),
    )
    append_state_line(
        state_file,
        "dataset_status: " + ("success" if not dataset_failed else "partial_failure"),
    )
    append_state_line(
        state_file,
        "resource_protection_triggered: "
        + ("true" if resource_protection_triggered else "false"),
    )

    if result_rows:
        print("\nTraining Summary:")
        for line in format_results_table(result_rows).splitlines():
            print(line)

    if failures:
        print("\nTraining failures:")
        for failure in failures:
            print(
                f"  rho={failure['rho']} | {failure['failure_category']} | {failure['exception_type']}: "
                f"{failure['message']}"
            )

    result_payload = {
        "dataset": dataset,
        "run_id": run_id,
        "batch_size": int(cfg.batch_size),
        "requested_rhos": [int(rho) for rho in rho_values],
        "rho_max_concurrent": int(rho_max_concurrent),
        "exit_code": 1 if dataset_failed else 0,
        "dataset_failed": bool(dataset_failed),
        "resource_protection_triggered": bool(resource_protection_triggered),
        "failure_categories": sorted({str(failure["failure_category"]) for failure in failures}),
        "state_file": str(state_file),
        "summary_file": str(
            model_dir / ("training_summary_partial.txt" if failures else "training_summary.txt")
        )
        if result_rows
        else "",
    }

    if dataset_failed:
        print(
            f"[dataset={dataset}][run_id={run_id}] completed with failures",
            flush=True,
        )
        return result_payload

    print(f"[dataset={dataset}][run_id={run_id}] completed successfully", flush=True)
    return result_payload


def write_schedule_summary(
    *,
    model_root: Path,
    batch_sizes: list[int],
    requested_datasets: list[str],
    requested_rhos: list[int],
    initial_rho_max_concurrent: int,
    entries: list[dict[str, Any]],
) -> Path:
    schedule_dir = model_root / "_schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = schedule_dir / f"schedule_summary_{timestamp}.json"
    write_json(
        output_path,
        {
            "generated_at": datetime.now().isoformat(),
            "batch_sizes": [int(v) for v in batch_sizes],
            "requested_datasets": requested_datasets,
            "requested_rhos": [int(v) for v in requested_rhos],
            "initial_rho_max_concurrent": int(initial_rho_max_concurrent),
            "entries": entries,
        },
    )
    return output_path


def _path_status(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.exists()}


def verify_run_artifacts(model_dir: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "run_dir": str(model_dir),
        "status": "ok",
        "issues": [],
        "shared_files": {},
        "rho_entries": [],
        "tracking": {},
    }
    issues: list[str] = []
    shared_files: dict[str, object] = {}

    for name in ("train_config.yaml", "dataset_job_state.txt", "dataset_metrics_summary.json"):
        path = model_dir / name
        shared_files[name] = _path_status(path)
        if not path.exists():
            issues.append(f"Missing shared file: {name}")

    summary_payload: dict[str, Any] = {}
    if (model_dir / "dataset_metrics_summary.json").exists():
        summary_payload = read_json(model_dir / "dataset_metrics_summary.json")
    requested_rhos = [int(v) for v in summary_payload.get("requested_rhos", [])]
    entries_by_rho = {
        int(entry["rho"]): entry for entry in summary_payload.get("entries", []) if "rho" in entry
    }

    success_count = 0
    failure_count = 0
    for rho in requested_rhos:
        metrics_path = metrics_path_for(model_dir, rho)
        metrics_payload = read_json(metrics_path) if metrics_path.exists() else {}
        status = str(
            metrics_payload.get(
                "status",
                entries_by_rho.get(rho, {}).get("status", "missing"),
            )
        )
        rho_report = {
            "rho": int(rho),
            "status": status,
            "files": {
                f"metrics_rho{rho}.json": _path_status(metrics_path),
                f"training_rho{rho}.log": _path_status(model_dir / f"training_rho{rho}.log"),
            },
        }
        if status == "success":
            success_count += 1
            for name in (
                f"model_rho{rho}.pth",
                f"zscore_stats_{rho}.csv",
                f"training_rho{rho}.log",
                f"metrics_rho{rho}.json",
            ):
                path = model_dir / name
                rho_report["files"][name] = _path_status(path)
                if not path.exists():
                    issues.append(f"rho={rho} missing expected success artifact: {name}")
        else:
            failure_count += 1
            if not metrics_path.exists():
                issues.append(f"rho={rho} missing metrics file")
        report["rho_entries"].append(rho_report)

    summary_name = ""
    if success_count > 0 and failure_count > 0:
        summary_name = "training_summary_partial.txt"
    elif success_count > 0:
        summary_name = "training_summary.txt"
    if summary_name:
        shared_files[summary_name] = _path_status(model_dir / summary_name)
        if not (model_dir / summary_name).exists():
            issues.append(f"Missing summary file: {summary_name}")

    if failure_count > 0:
        for name in ("training_failure_summary.json", "training_failure_summary.txt"):
            shared_files[name] = _path_status(model_dir / name)
            if not (model_dir / name).exists():
                issues.append(f"Missing failure summary: {name}")

    stop_summary = model_dir / "stop_summary.txt"
    shared_files["stop_summary.txt"] = _path_status(stop_summary)

    train_config_payload: dict[str, Any] = {}
    if (model_dir / "train_config.yaml").exists():
        train_config_payload = yaml.safe_load((model_dir / "train_config.yaml").read_text(encoding="utf-8")) or {}
    tracking_info = train_config_payload.get("tracking", {})
    tracking_mode = str(tracking_info.get("mode", "unknown"))
    logdir_name = str(tracking_info.get("logdir_name", "swanlab"))
    swanlab_dir = model_dir / logdir_name
    tracking_report = {
        "mode": tracking_mode,
        "logdir": str(swanlab_dir),
        "exists": swanlab_dir.exists(),
        "non_empty": swanlab_dir.exists() and any(swanlab_dir.iterdir()),
    }
    if tracking_mode.lower() not in {"disabled", "off", "none", "unknown"} and not swanlab_dir.exists():
        issues.append(f"SwanLab logdir missing for enabled tracking: {swanlab_dir}")

    report["shared_files"] = shared_files
    report["tracking"] = tracking_report
    report["issue_count"] = len(issues)
    report["issues"] = issues
    report["status"] = "ok" if not issues else "failed"
    return report


def verify_schedule_summary(schedule_path: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "schedule_path": str(schedule_path),
        "status": "ok",
        "issues": [],
        "entries": [],
    }
    issues: list[str] = []
    if not schedule_path.exists():
        return {
            **report,
            "status": "failed",
            "issues": [f"Schedule summary file not found: {schedule_path}"],
            "issue_count": 1,
        }

    payload = read_json(schedule_path)
    entries = payload.get("entries", [])
    for index, entry in enumerate(entries, start=1):
        state_file = str(entry.get("state_file", ""))
        summary_file = str(entry.get("summary_file", ""))
        entry_report = {
            "index": index,
            "dataset": str(entry.get("dataset", "")),
            "run_id": str(entry.get("run_id", "")),
            "exit_code": int(entry.get("exit_code", 1)),
            "dataset_failed": bool(entry.get("dataset_failed", True)),
            "rho_max_concurrent": int(entry.get("rho_max_concurrent", 0)),
            "state_file_exists": bool(state_file) and Path(state_file).exists(),
            "summary_file_exists": bool(summary_file) and Path(summary_file).exists(),
        }
        if state_file and not Path(state_file).exists():
            issues.append(f"Entry {index} missing state file: {state_file}")
        if not entry_report["dataset_failed"] and summary_file and not Path(summary_file).exists():
            issues.append(f"Entry {index} missing summary file: {summary_file}")
        report["entries"].append(entry_report)

    report["issue_count"] = len(issues)
    report["issues"] = issues
    report["status"] = "ok" if not issues else "failed"
    return report


def run_training_schedule(
    *,
    config_path: str,
    base_config_overrides: dict[str, object],
    datasets: list[str],
    batch_sizes: list[int],
    rho_values: list[int],
    initial_rho_max_concurrent: int,
    allow_cpu: bool,
    device: str,
) -> int:
    if not datasets:
        raise ValueError("--datasets must be non-empty in --schedule-run mode.")
    if not batch_sizes:
        raise ValueError("--batch-sizes must be non-empty in --schedule-run mode.")
    if initial_rho_max_concurrent < 1:
        raise ValueError("--rho-max-concurrent must be >= 1.")

    current_rho_max_concurrent = int(initial_rho_max_concurrent)
    overall_status = 0
    schedule_entries: list[dict[str, Any]] = []
    print(
        f"[schedule-run] starting with datasets={datasets} batch_sizes={batch_sizes} "
        f"rhos={rho_values} initial_rho_max_concurrent={current_rho_max_concurrent}",
        flush=True,
    )

    last_cfg: TrainingConfig | None = None
    for batch_size in batch_sizes:
        print(f"[schedule-run] batch_size={batch_size}", flush=True)
        per_batch_overrides = dict(base_config_overrides)
        per_batch_overrides["batch_size"] = int(batch_size)
        cfg_for_batch = load_training_config(config_path, overrides=per_batch_overrides)
        last_cfg = cfg_for_batch

        for dataset in datasets:
            try:
                run_id = choose_run_id(
                    default_dataset_run_id(dataset, cfg_for_batch.batch_size),
                    cfg_for_batch.model_dir,
                )
                result = run_dataset_lifecycle(
                    cfg=cfg_for_batch,
                    dataset=dataset,
                    run_id=run_id,
                    rho_values=rho_values,
                    rho_max_concurrent=current_rho_max_concurrent,
                    config_path=config_path,
                    config_overrides=per_batch_overrides,
                    allow_cpu=allow_cpu,
                    device=device,
                )
            except Exception as exc:
                failure_category = classify_failure_payload(
                    {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                result = {
                    "dataset": dataset,
                    "run_id": "",
                    "batch_size": int(batch_size),
                    "requested_rhos": [int(rho) for rho in rho_values],
                    "rho_max_concurrent": int(current_rho_max_concurrent),
                    "exit_code": 1,
                    "dataset_failed": True,
                    "resource_protection_triggered": failure_category in RESOURCE_FAILURE_CATEGORIES,
                    "failure_categories": [failure_category],
                    "state_file": "",
                    "summary_file": "",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                print(
                    f"[schedule-run][dataset={dataset}][batch_size={batch_size}] "
                    f"failed before completion: {type(exc).__name__}: {exc}",
                    flush=True,
                )

            schedule_entries.append(result)
            if result["exit_code"] != 0:
                overall_status = 1
                if current_rho_max_concurrent > 1:
                    current_rho_max_concurrent = 1
                    print(
                        f"[schedule-run][dataset={dataset}][batch_size={batch_size}] "
                        "downgrading future datasets to rho concurrency=1",
                        flush=True,
                    )

    assert last_cfg is not None
    summary_path = write_schedule_summary(
        model_root=last_cfg.model_dir,
        batch_sizes=batch_sizes,
        requested_datasets=datasets,
        requested_rhos=rho_values,
        initial_rho_max_concurrent=initial_rho_max_concurrent,
        entries=schedule_entries,
    )
    print(f"[schedule-run] summary written to {summary_path}", flush=True)
    return overall_status


def resolve_execution_device(*, require_cuda: bool, allow_cpu: bool) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda and not allow_cpu:
        diagnostics = build_cuda_diagnostics()
        raise RuntimeError(
            "CUDA is required for training but is unavailable. "
            f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=True)}"
        )
    return "cpu"


def build_config_failure_report(
    *,
    config_path: str,
    message: str,
    datasets: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    rho_values: list[int] | None = None,
    require_cuda: bool | None = None,
) -> dict[str, object]:
    report = build_environment_report(config_path=Path(config_path).resolve())
    report["status"] = "failed"
    report["config_error"] = message
    if datasets is not None:
        report["datasets"] = list(datasets)
    if batch_sizes is not None:
        report["batch_sizes"] = [int(v) for v in batch_sizes]
    if rho_values is not None:
        report["rho_values"] = [int(v) for v in rho_values]
    if require_cuda is not None:
        report["require_cuda"] = bool(require_cuda)
    return report


def main() -> None:
    args = parse_args()

    if args.verify_schedule_path:
        report = verify_schedule_summary(Path(args.verify_schedule_path))
        print(json.dumps(report, indent=2, ensure_ascii=True))
        if str(report["status"]) != "ok":
            raise SystemExit(1)
        return

    if args.preflight_schedule:
        try:
            preflight_batch_sizes = (
                args.batch_sizes
                if args.batch_sizes
                else [load_training_config(args.config, overrides=args.config_overrides).batch_size]
            )
            report = build_schedule_preflight_report(
                config_path=args.config,
                base_config_overrides=args.config_overrides,
                datasets=args.datasets,
                batch_sizes=preflight_batch_sizes,
                rho_values=args.rho_values,
                requested_rho_max_concurrent=args.rho_max_concurrent,
                require_cuda=args.require_cuda,
            )
        except TrainingConfigError as exc:
            report = build_config_failure_report(
                config_path=args.config,
                message=str(exc),
                datasets=args.datasets,
                batch_sizes=args.batch_sizes,
                rho_values=args.rho_values,
                require_cuda=args.require_cuda,
            )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        if str(report["status"]) != "ok":
            raise SystemExit(1)
        return

    try:
        cfg = load_training_config(args.config, overrides=args.config_overrides)
    except TrainingConfigError as exc:
        if args.env_check:
            report = build_config_failure_report(
                config_path=args.config,
                message=str(exc),
            )
            print(json.dumps(report, indent=2, ensure_ascii=True))
            raise SystemExit(1)
        raise SystemExit(f"ERROR: {exc}")

    if args.env_check:
        report = build_environment_report(
            config_path=Path(args.config).resolve(),
            data_dir=cfg.data_dir,
            model_dir=cfg.model_dir,
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        if args.require_cuda and not report["cuda"]["cuda_available"]:
            raise SystemExit(1)
        return

    if args.validate_data:
        report = build_dataset_validation_report(
            cfg=cfg,
            dataset=args.dataset,
            rho_values=args.rho_values,
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        if int(report["failure_count"]) > 0:
            raise SystemExit(1)
        return

    if args.verify_run:
        if not args.run_id:
            raise SystemExit("--verify-run requires --run-id.")
        report = verify_run_artifacts(cfg.model_dir / args.run_id)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        if str(report["status"]) != "ok":
            raise SystemExit(1)
        return

    training_mode = not any(
        [
            args.prepare_run,
            args.aggregate_run,
            args.env_check,
            args.preflight_schedule,
            args.validate_data,
            args.verify_run,
            bool(args.verify_schedule_path),
        ]
    )
    device = resolve_execution_device(
        require_cuda=training_mode,
        allow_cpu=args.allow_cpu,
    )

    if args.schedule_run:
        exit_code = run_training_schedule(
            config_path=args.config,
            base_config_overrides=args.config_overrides,
            datasets=args.datasets,
            batch_sizes=args.batch_sizes if args.batch_sizes else [cfg.batch_size],
            rho_values=args.rho_values,
            initial_rho_max_concurrent=args.rho_max_concurrent,
            allow_cpu=args.allow_cpu,
            device=device,
        )
        raise SystemExit(exit_code)

    data_dir = cfg.data_dir / args.dataset
    if args.dataset_run:
        run_id = (
            args.run_id
            if args.run_id
            else choose_run_id(default_dataset_run_id(args.dataset, cfg.batch_size), cfg.model_dir)
        )
    else:
        run_id = args.run_id if args.run_id else f"{args.dataset}_bs{cfg.batch_size}"
    model_dir = cfg.model_dir / run_id

    if args.dataset_run:
        result = run_dataset_lifecycle(
            cfg=cfg,
            dataset=args.dataset,
            run_id=run_id,
            rho_values=args.rho_values,
            rho_max_concurrent=args.rho_max_concurrent,
            config_path=args.config,
            config_overrides=args.config_overrides,
            allow_cpu=args.allow_cpu,
            device=device,
        )
        raise SystemExit(int(result["exit_code"]))

    if args.prepare_run:
        prepare_run_directory(
            cfg=cfg,
            model_dir=model_dir,
            run_id=run_id,
            dataset=args.dataset,
            rho_values=args.rho_values,
            device=device,
        )
        print(f"Prepared run directory: {model_dir}")
        return

    if args.aggregate_run:
        result_rows, failures = finalize_dataset_run(
            cfg=cfg,
            model_dir=model_dir,
            dataset=args.dataset,
            run_id=run_id,
            requested_rhos=args.rho_values,
            batch_size=cfg.batch_size,
        )
        if result_rows:
            print("\nTraining Summary:")
            for line in format_results_table(result_rows).splitlines():
                print(line)
        if failures:
            print("\nTraining failures:")
            for failure in failures:
                print(
                    f"  rho={failure['rho']} | {failure['failure_category']} | {failure['exception_type']}: "
                    f"{failure['message']}"
                )
            raise SystemExit(1)
        return

    if args.worker_mode:
        if len(args.rho_values) != 1:
            raise SystemExit("--worker-mode expects exactly one rho value.")
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Worker mode requires an existing run directory prepared in advance: {model_dir}"
            )
        rho = int(args.rho_values[0])
        metrics_path = metrics_path_for(model_dir, rho)
        run_training_worker(
            rho,
            device,
            args.config,
            args.config_overrides,
            str(data_dir),
            str(model_dir),
            metrics_path=metrics_path,
        )
        missing_artifacts = validate_rho_artifacts(model_dir, rho)
        if missing_artifacts:
            failure_payload = {
                "status": "failed",
                "rho": rho,
                "dataset": args.dataset,
                "run_id": run_id,
                "batch_size": int(cfg.batch_size),
                "exception_type": "RuntimeError",
                "failure_category": "missing_artifact",
                "message": "Missing expected artifacts: " + ", ".join(missing_artifacts),
                "traceback": "",
            }
            write_json(metrics_path, failure_payload)
            raise RuntimeError(failure_payload["message"])
        return

    prepare_run_directory(
        cfg=cfg,
        model_dir=model_dir,
        run_id=run_id,
        dataset=args.dataset,
        rho_values=args.rho_values,
        device=device,
    )
    print(f"Running SEQUENTIAL training for rhos: {args.rho_values} on device: {device}")

    for rho in args.rho_values:
        try:
            print(f"[Main] Starting training for rho={rho}...")
            run_training_worker(
                rho,
                device,
                args.config,
                args.config_overrides,
                str(data_dir),
                str(model_dir),
                metrics_path=metrics_path_for(model_dir, int(rho)),
            )
            missing_artifacts = validate_rho_artifacts(model_dir, int(rho))
            if missing_artifacts:
                raise RuntimeError(
                    f"rho={rho} completed but missing expected artifacts: "
                    f"{', '.join(missing_artifacts)}"
                )
            print(f"[Main] Completed rho={rho}.")
        except Exception as exc:
            print(f"[Main] FAILED rho={rho}: {type(exc).__name__}: {exc}")
            if not args.sequential and len(args.rho_values) == 1:
                break

    result_rows, failures = finalize_dataset_run(
        cfg=cfg,
        model_dir=model_dir,
        dataset=args.dataset,
        run_id=run_id,
        requested_rhos=args.rho_values,
        batch_size=cfg.batch_size,
    )
    if result_rows:
        print("\nTraining Summary:")
        for line in format_results_table(result_rows).splitlines():
            print(line)

    if failures:
        print("\nTraining failures:")
        for failure in failures:
            print(
                f"  rho={failure['rho']} | {failure['failure_category']} | {failure['exception_type']}: "
                f"{failure['message']}"
            )
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    cfg = load_training_config()
    parser = argparse.ArgumentParser(description="Train rho-specific curvature MLP models.")
    parser.add_argument("--dataset", type=str, default="", help="Dataset ID (e.g., ds01)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset IDs for --schedule-run mode.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Optional Run ID")
    parser.add_argument(
        "--rho",
        nargs="+",
        type=int,
        default=list(cfg.resolutions),
        help=f"rho values to train (default: {list(cfg.resolutions)})",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config yaml path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs for quick testing.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Retained for backwards compatibility with the sequential CLI entry point.",
    )
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help="Run a single-rho worker inside an already-prepared run directory.",
    )
    parser.add_argument(
        "--prepare-run",
        action="store_true",
        help="Create a fresh run directory and write shared dataset-level metadata.",
    )
    parser.add_argument(
        "--aggregate-run",
        action="store_true",
        help="Aggregate rho worker metrics into dataset summaries and SwanLab offline logs.",
    )
    parser.add_argument(
        "--dataset-run",
        action="store_true",
        help="Run the full dataset lifecycle: prepare, launch rho workers, wait, and aggregate.",
    )
    parser.add_argument(
        "--schedule-run",
        action="store_true",
        help="Run multiple datasets/batch sizes in Python and apply downgrade policy internally.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[],
        help="Batch sizes for --schedule-run mode.",
    )
    parser.add_argument(
        "--rho-max-concurrent",
        type=int,
        default=3,
        help="Maximum number of rho workers to run concurrently in --dataset-run or --schedule-run mode.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback when CUDA is unavailable. Training modes require CUDA by default.",
    )
    parser.add_argument(
        "--env-check",
        action="store_true",
        help="Print environment diagnostics for CPU/GPU setup without starting training.",
    )
    parser.add_argument(
        "--preflight-schedule",
        action="store_true",
        help="Validate schedule inputs, data files, and environment before starting a schedule-run.",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate dataset HDF5 inputs and print a structured report without starting training.",
    )
    parser.add_argument(
        "--verify-run",
        action="store_true",
        help="Verify dataset-level run artifacts under model/<run-id>.",
    )
    parser.add_argument(
        "--verify-schedule-path",
        default="",
        help="Verify a schedule summary JSON file produced by --schedule-run.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="With --env-check, return a non-zero exit code when CUDA is unavailable.",
    )
    args = parser.parse_args()
    args.rho_values = args.rho

    active_modes = [
        args.worker_mode,
        args.prepare_run,
        args.aggregate_run,
        args.dataset_run,
        args.schedule_run,
        args.env_check,
        args.preflight_schedule,
        args.validate_data,
        args.verify_run,
        bool(args.verify_schedule_path),
    ]
    if sum(1 for value in active_modes if value) > 1:
        parser.error(
            "--worker-mode, --prepare-run, --aggregate-run, --dataset-run, --schedule-run, "
            "--env-check, --preflight-schedule, --validate-data, --verify-run, "
            "and --verify-schedule-path are mutually exclusive."
        )

    if (
        not args.env_check
        and not args.schedule_run
        and not args.preflight_schedule
        and not args.verify_schedule_path
        and not args.verify_run
        and not args.dataset
    ):
        parser.error(
            "--dataset is required unless using env-check, preflight-schedule, "
            "schedule-run, verify-run, or verify-schedule-path."
        )

    args.config_overrides = {}
    if args.batch_size is not None:
        args.config_overrides["batch_size"] = int(args.batch_size)
    if args.max_epochs is not None:
        args.config_overrides["max_epochs"] = int(args.max_epochs)
    if args.patience is not None:
        args.config_overrides["patience"] = int(args.patience)

    return args


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
