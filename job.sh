#!/bin/bash
set -euo pipefail
umask "${JOB_UMASK:-002}"

#SBATCH --job-name=pde_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

SCRIPT_PATH=${BASH_SOURCE[0]:-$0}
REPO_ROOT=$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
LOG_ROOT_DIR="$REPO_ROOT/logs"
if [ -n "${LOG_ID:-}" ]; then
    EFFECTIVE_LOG_ID="$LOG_ID"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    EFFECTIVE_LOG_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    EFFECTIVE_LOG_ID="${SLURM_JOB_ID}"
else
    EFFECTIVE_LOG_ID="local_$(date +%Y%m%d_%H%M%S)"
fi
JOB_LOG_DIR="$LOG_ROOT_DIR/$EFFECTIVE_LOG_ID"

cd "$REPO_ROOT" || exit 1
mkdir -p "$LOG_ROOT_DIR" "$JOB_LOG_DIR"
mkdir -p "$LOG_ROOT_DIR/slurm"
chmod u+rwx "$JOB_LOG_DIR" >/dev/null 2>&1 || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS_STR=${DATASETS:-"data02_001 data02_002 data02_004 data02_005"}
RHOS_STR=${RHOS:-"256 266 276"}
BATCH_SIZES_STR=${BATCH_SIZES:-"32"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"code1-gpu"}
CONDA_ENV_FILE=${CONDA_ENV_FILE:-"environment.yaml"}
AUTO_CREATE_CONDA_ENV=${AUTO_CREATE_CONDA_ENV:-1}
AUTO_UPDATE_CONDA_ENV=${AUTO_UPDATE_CONDA_ENV:-0}
TRAIN_CONFIG_PATH=${TRAIN_CONFIG_PATH:-"train/config.yaml"}
RHO_MAX_CONCURRENT_DEFAULT=${RHO_MAX_CONCURRENT:-3}
CURRENT_RHO_MAX_CONCURRENT=$RHO_MAX_CONCURRENT_DEFAULT
GPU_PREFLIGHT_MAX_RETRIES=${GPU_PREFLIGHT_MAX_RETRIES:-3}
GPU_PREFLIGHT_RETRY_SLEEP_SECONDS=${GPU_PREFLIGHT_RETRY_SLEEP_SECONDS:-5}
SUMMARY_LOG="$JOB_LOG_DIR/job_summary.log"
SCHEDULE_PREFLIGHT_LOG="$JOB_LOG_DIR/schedule_preflight.json"
GPU_PREFLIGHT_LOG="$JOB_LOG_DIR/gpu_preflight.log"
JOB_CONTEXT_LOG="$JOB_LOG_DIR/job_context.env"
CONDA_BOOTSTRAP_LOG="$JOB_LOG_DIR/conda_bootstrap.log"
ENV_REPORT_LOG="$JOB_LOG_DIR/environment_report.json"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a RHOS <<< "$RHOS_STR"
read -r -a BATCH_SIZES <<< "$BATCH_SIZES_STR"

: > "$SUMMARY_LOG"
: > "$JOB_CONTEXT_LOG"

log_job() {
    echo "$*" | tee -a "$SUMMARY_LOG"
}

log_error_and_exit() {
    echo "ERROR: $*" >&2
    log_job "ERROR: $*"
    exit 1
}

on_job_exit() {
    local rc=$?
    trap - EXIT
    log_job ""
    log_job "Job finished at $(date --iso-8601=seconds) with exit_code=$rc"
}

trap on_job_exit EXIT

gpu_preflight_check() {
    : > "$GPU_PREFLIGHT_LOG"
    local attempt
    for ((attempt=1; attempt<=GPU_PREFLIGHT_MAX_RETRIES; attempt++)); do
        {
            echo "[$(date --iso-8601=seconds)] gpu_preflight attempt=${attempt}/${GPU_PREFLIGHT_MAX_RETRIES}"
            echo "hostname=$(hostname)"
            echo "log_id=${EFFECTIVE_LOG_ID}"
            echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
            echo "slurm_job_id=${SLURM_JOB_ID:-<unset>}"
            echo "slurm_job_gpus=${SLURM_JOB_GPUS:-<unset>}"
            echo "slurm_step_gpus=${SLURM_STEP_GPUS:-<unset>}"
            if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_JOB_GPUS:-}" ] && [ -z "${SLURM_STEP_GPUS:-}" ]; then
                echo "ERROR: Slurm job is present but neither SLURM_JOB_GPUS nor SLURM_STEP_GPUS is set."
                echo "This node may expose GPUs through nvidia-smi without this step actually owning a GPU allocation."
            fi
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "--- nvidia-smi ---"
                nvidia-smi || true
            else
                echo "nvidia-smi unavailable"
            fi
            echo "--- torch cuda probe ---"
        } >> "$GPU_PREFLIGHT_LOG"

        if python - <<'PY' >> "$GPU_PREFLIGHT_LOG" 2>&1
import json
import os
import sys
import torch

slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
slurm_job_gpus = os.environ.get("SLURM_JOB_GPUS", "").strip()
slurm_step_gpus = os.environ.get("SLURM_STEP_GPUS", "").strip()
payload = {
    "torch_version": getattr(torch, "__version__", ""),
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "slurm_job_id": slurm_job_id,
    "slurm_job_gpus": slurm_job_gpus,
    "slurm_step_gpus": slurm_step_gpus,
}
print(json.dumps(payload, ensure_ascii=True))
if slurm_job_id and not slurm_job_gpus and not slurm_step_gpus:
    sys.exit(3)
sys.exit(0 if payload["cuda_available"] else 2)
PY
        then
            log_job "CUDA preflight passed. Details: $GPU_PREFLIGHT_LOG"
            return 0
        fi

        if [ "$attempt" -lt "$GPU_PREFLIGHT_MAX_RETRIES" ]; then
            {
                echo "retrying after ${GPU_PREFLIGHT_RETRY_SLEEP_SECONDS}s due to failed CUDA preflight"
                echo
            } >> "$GPU_PREFLIGHT_LOG"
            sleep "$GPU_PREFLIGHT_RETRY_SLEEP_SECONDS"
        fi
    done

    log_job "ERROR: CUDA preflight failed after ${GPU_PREFLIGHT_MAX_RETRIES} attempts. See $GPU_PREFLIGHT_LOG"
    cat "$GPU_PREFLIGHT_LOG" >&2
    exit 1
}

resolve_conda() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return 0
    fi
    local conda_path
    for conda_path in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$conda_path" ]; then
            # shellcheck disable=SC1090
            source "$conda_path"
            return 0
        fi
    done
    return 1
}

resolve_conda || {
    echo "ERROR: conda command not found." >&2
    exit 1
}

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"
}

bootstrap_conda_env() {
    : > "$CONDA_BOOTSTRAP_LOG"
    {
        echo "[$(date --iso-8601=seconds)] conda bootstrap"
        echo "env_name=$CONDA_ENV_NAME"
        echo "env_file=$CONDA_ENV_FILE"
        echo "auto_create=$AUTO_CREATE_CONDA_ENV"
        echo "auto_update=$AUTO_UPDATE_CONDA_ENV"
    } >> "$CONDA_BOOTSTRAP_LOG"

    if [ ! -f "$REPO_ROOT/$CONDA_ENV_FILE" ] && [ ! -f "$CONDA_ENV_FILE" ]; then
        log_error_and_exit "CONDA_ENV_FILE does not exist: $CONDA_ENV_FILE"
    fi

    local resolved_env_file="$CONDA_ENV_FILE"
    if [ -f "$REPO_ROOT/$CONDA_ENV_FILE" ]; then
        resolved_env_file="$REPO_ROOT/$CONDA_ENV_FILE"
    fi

    if conda_env_exists; then
        log_job "Conda env already exists: $CONDA_ENV_NAME"
        if [ "$AUTO_UPDATE_CONDA_ENV" = "1" ]; then
            log_job "Updating conda env from $resolved_env_file"
            if ! conda env update --name "$CONDA_ENV_NAME" --file "$resolved_env_file" >> "$CONDA_BOOTSTRAP_LOG" 2>&1; then
                log_error_and_exit "Failed to update conda env $CONDA_ENV_NAME. See $CONDA_BOOTSTRAP_LOG"
            fi
        fi
        return 0
    fi

    if [ "$AUTO_CREATE_CONDA_ENV" != "1" ]; then
        log_error_and_exit "Conda env $CONDA_ENV_NAME is missing and AUTO_CREATE_CONDA_ENV=0"
    fi

    log_job "Creating conda env $CONDA_ENV_NAME from $resolved_env_file"
    if ! conda env create --name "$CONDA_ENV_NAME" --file "$resolved_env_file" >> "$CONDA_BOOTSTRAP_LOG" 2>&1; then
        log_error_and_exit "Failed to create conda env $CONDA_ENV_NAME. See $CONDA_BOOTSTRAP_LOG"
    fi
}

bootstrap_conda_env

conda activate "$CONDA_ENV_NAME" || log_error_and_exit "failed to activate conda env $CONDA_ENV_NAME"

chmod u+x "$SCRIPT_PATH" >/dev/null 2>&1 || true
{
    echo "TIMESTAMP=$(date --iso-8601=seconds)"
    echo "LOG_ID=$EFFECTIVE_LOG_ID"
    echo "REPO_ROOT=$REPO_ROOT"
    echo "JOB_LOG_DIR=$JOB_LOG_DIR"
    echo "HOSTNAME=$(hostname)"
    echo "UNAME=$(uname -a)"
    echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
    echo "CONDA_ENV_FILE=$CONDA_ENV_FILE"
    echo "AUTO_CREATE_CONDA_ENV=$AUTO_CREATE_CONDA_ENV"
    echo "AUTO_UPDATE_CONDA_ENV=$AUTO_UPDATE_CONDA_ENV"
    echo "TRAIN_CONFIG_PATH=$TRAIN_CONFIG_PATH"
    echo "RHO_MAX_CONCURRENT=$CURRENT_RHO_MAX_CONCURRENT"
    echo "DATASETS=$DATASETS_STR"
    echo "RHOS=$RHOS_STR"
    echo "BATCH_SIZES=$BATCH_SIZES_STR"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
    echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
    echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "CONDA_VERSION=$(conda --version 2>/dev/null || true)"
    echo "PYTHON_VERSION=$(python --version 2>/dev/null || true)"
    if [ -f /etc/os-release ]; then
        echo "--- /etc/os-release ---"
        cat /etc/os-release
    fi
} >> "$JOB_CONTEXT_LOG"

if python -m train --env-check > "$ENV_REPORT_LOG"; then
    log_job "Environment report written to $ENV_REPORT_LOG"
else
    log_error_and_exit "Failed to generate environment report. See $ENV_REPORT_LOG"
fi

gpu_preflight_check

if [ "${#DATASETS[@]}" -eq 0 ] || [ "${#RHOS[@]}" -eq 0 ] || [ "${#BATCH_SIZES[@]}" -eq 0 ]; then
    echo "ERROR: DATASETS, RHOS, and BATCH_SIZES must all be non-empty." >&2
    exit 1
fi

if [ ! -f "$REPO_ROOT/$TRAIN_CONFIG_PATH" ] && [ ! -f "$TRAIN_CONFIG_PATH" ]; then
    echo "ERROR: TRAIN_CONFIG_PATH does not exist: $TRAIN_CONFIG_PATH" >&2
    exit 1
fi

log_job "Job started at $(date --iso-8601=seconds)"
log_job "Host: $(hostname)"
log_job "Job ID: ${SLURM_JOB_ID:-local}"
log_job "Datasets: ${DATASETS[*]}"
log_job "Batch sizes: ${BATCH_SIZES[*]}"
log_job "Rhos: ${RHOS[*]}"
log_job "Conda env: ${CONDA_ENV_NAME}"
log_job "Conda env file: ${CONDA_ENV_FILE}"
log_job "Train config: ${TRAIN_CONFIG_PATH}"
log_job "Initial RHO_MAX_CONCURRENT: ${CURRENT_RHO_MAX_CONCURRENT}"
log_job "Log ID: ${EFFECTIVE_LOG_ID}"
log_job "Job log dir: ${JOB_LOG_DIR}"
log_job "Summary log: ${SUMMARY_LOG}"
log_job "Conda bootstrap log: ${CONDA_BOOTSTRAP_LOG}"
log_job "Environment report: ${ENV_REPORT_LOG}"

if python -m train \
    --config "$TRAIN_CONFIG_PATH" \
    --datasets "${DATASETS[@]}" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --rho "${RHOS[@]}" \
    --rho-max-concurrent "$CURRENT_RHO_MAX_CONCURRENT" \
    --preflight-schedule \
    --require-cuda > "$SCHEDULE_PREFLIGHT_LOG"; then
    log_job "Schedule preflight passed. Details: $SCHEDULE_PREFLIGHT_LOG"
else
    log_job "ERROR: schedule preflight failed. See $SCHEDULE_PREFLIGHT_LOG"
    cat "$SCHEDULE_PREFLIGHT_LOG" >&2
    exit 1
fi

if python -m train \
    --config "$TRAIN_CONFIG_PATH" \
    --datasets "${DATASETS[@]}" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --rho "${RHOS[@]}" \
    --schedule-run \
    --rho-max-concurrent "$CURRENT_RHO_MAX_CONCURRENT"; then
    OVERALL_STATUS=0
else
    OVERALL_STATUS=1
fi
exit "$OVERALL_STATUS"
