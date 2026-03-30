#!/bin/bash
set -uo pipefail

#SBATCH --job-name=pde_train_generic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

# SLURM experiment scheduler for dataset-level training batches.
# Responsibilities:
#   - expand dataset / rho / batch-size combinations at the shell layer
#   - assign datasets to a fixed set of GPUs in predictable grouped blocks
#   - launch one Python training process per dataset
#   - keep the dataset-internal rho parallelism inside `python -m train`
#   - continue other datasets when one fails, then emit a job-level summary
#
# Default layout: 2 GPUs, 4 datasets, 2 datasets per GPU.
#
# Examples:
#   sbatch --export=ALL,BATCH_SIZES="32" job.sh
#   sbatch --export=ALL,BATCH_SIZES="128",GPU_IDS="0 1" job.sh
#   sbatch --export=ALL,BATCH_SIZES="32 128 1024" job.sh

cd "$(dirname "$0")" || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

DATASETS_STR=${DATASETS:-"data02_001 data02_002 data02_004 data02_005"}
RHOS_STR=${RHOS:-"256 266 276"}
BATCH_SIZES_STR=${BATCH_SIZES:-"32"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"jq"}
GPU_IDS_STR=${GPU_IDS:-""}
DATASETS_PER_GPU=${DATASETS_PER_GPU:-2}
SUMMARY_LOG="logs/job_summary_${SLURM_JOB_ID:-local}.log"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a RHOS <<< "$RHOS_STR"
read -r -a BATCH_SIZES <<< "$BATCH_SIZES_STR"

if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "ERROR: DATASETS is empty." >&2
    exit 1
fi
if [ "${#BATCH_SIZES[@]}" -eq 0 ]; then
    echo "ERROR: BATCH_SIZES is empty." >&2
    exit 1
fi

: > "$SUMMARY_LOG"

log_job() {
    echo "$*" | tee -a "$SUMMARY_LOG"
}

resolve_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi
    local candidates=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/.bashrc"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate" ]; then
            # shellcheck disable=SC1090
            source "$candidate"
            if command -v conda >/dev/null 2>&1; then
                return 0
            fi
        fi
    done
    return 1
}

resolve_gpus() {
    local resolved=()
    if [ -n "$GPU_IDS_STR" ]; then
        read -r -a resolved <<< "$GPU_IDS_STR"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        local visible="${CUDA_VISIBLE_DEVICES// /}"
        IFS=',' read -r -a resolved <<< "$visible"
    elif [ -n "${SLURM_STEP_GPUS:-}" ]; then
        local step_gpus="${SLURM_STEP_GPUS// /}"
        IFS=',' read -r -a resolved <<< "$step_gpus"
    elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
        local job_gpus="${SLURM_JOB_GPUS// /}"
        IFS=',' read -r -a resolved <<< "$job_gpus"
    else
        resolved=(0 1)
    fi

    if [ "${#resolved[@]}" -gt 2 ] && [ -z "$GPU_IDS_STR" ]; then
        resolved=("${resolved[@]:0:2}")
    fi

    GPU_LIST=("${resolved[@]}")
    NUM_GPUS=${#GPU_LIST[@]}
}

assign_gpu_for_index() {
    local dataset_index=$1
    local gpu_group=$((dataset_index / DATASETS_PER_GPU))
    local gpu_idx=$((gpu_group % NUM_GPUS))
    printf '%s\n' "${GPU_LIST[$gpu_idx]}"
}

print_assignment_plan() {
    local batch_size=$1
    local i
    local dataset
    local gpu_id

    log_job "Assignment plan for batch_size=${batch_size}:"
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu_id=$(assign_gpu_for_index "$i")
        log_job "  ${dataset} -> GPU ${gpu_id} | rhos: ${RHOS[*]}"
    done
}

validate_datasets() {
    VALID_DATASETS=()
    local dataset
    local rho
    local missing
    for dataset in "${DATASETS[@]}"; do
        missing=0
        for rho in "${RHOS[@]}"; do
            if [ ! -f "data/${dataset}/train_rho${rho}.h5" ]; then
                echo "WARNING: Missing data/${dataset}/train_rho${rho}.h5, skipping ${dataset}." >&2
                missing=1
                break
            fi
        done
        if [ "$missing" -eq 0 ]; then
            VALID_DATASETS+=("$dataset")
        fi
    done

    if [ "${#VALID_DATASETS[@]}" -eq 0 ]; then
        echo "ERROR: No valid datasets were found." >&2
        exit 1
    fi
}

run_training() {
    local dataset=$1
    local gpu_id=$2
    local batch_size=$3
    local run_id="${dataset}_batch${batch_size}"
    local log_file="logs/train_${run_id}_${SLURM_JOB_ID:-local}.log"

    echo "[${run_id}] starting on GPU ${gpu_id}"
    {
        echo "=========================================================="
        echo "Job ID      : ${SLURM_JOB_ID:-local}"
        echo "Host        : $(hostname)"
        echo "Start time  : $(date)"
        echo "Dataset     : ${dataset}"
        echo "Batch size  : ${batch_size}"
        echo "Run ID      : ${run_id}"
        echo "GPU         : ${gpu_id}"
        echo "Rhos        : ${RHOS[*]}"
        echo "Scheduler   : dataset-level shell scheduler"
        echo "Inner rho parallelism handled by python -m train"
        echo "=========================================================="
        echo "Command:"
        echo "python -m train --dataset ${dataset} --run-id ${run_id} --batch-size ${batch_size} --rho ${RHOS[*]}"
        echo
    } >> "$log_file"

    export CUDA_VISIBLE_DEVICES="$gpu_id"

    if conda run -n "$CONDA_ENV_NAME" --no-capture-output \
        python -m train --dataset "$dataset" --run-id "$run_id" --batch-size "$batch_size" --rho "${RHOS[@]}" \
        >> "$log_file" 2>&1; then
        echo "[${run_id}] finished successfully on GPU ${gpu_id}"
        echo "[$(date)] SUCCESS" >> "$log_file"
    else
        local status=$?
        echo "[${run_id}] failed on GPU ${gpu_id} (status=${status})" >&2
        echo "[$(date)] ERROR status=${status}" >> "$log_file"
        nvidia-smi -i "$gpu_id" >> "$log_file" 2>&1 || true
        return "$status"
    fi
}

wait_for_group() {
    local overall=0
    local i
    local status

    BATCH_SUCCESSES=()
    BATCH_FAILURES=()
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            log_job "-> ${PID_LABELS[$i]} completed"
            BATCH_SUCCESSES+=("${PID_LABELS[$i]}")
        else
            status=$?
            log_job "-> ${PID_LABELS[$i]} failed (exit=${status})"
            BATCH_FAILURES+=("${PID_LABELS[$i]} (exit=${status})")
            overall=1
        fi
    done
    return "$overall"
}

resolve_conda || {
    echo "ERROR: conda command not found." >&2
    exit 1
}

resolve_gpus
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: no GPUs resolved." >&2
    exit 1
fi

validate_datasets
DATASETS=("${VALID_DATASETS[@]}")
JOB_SUCCESSES=()
JOB_FAILURES=()

if [ $((NUM_GPUS * DATASETS_PER_GPU)) -lt "${#DATASETS[@]}" ]; then
    log_job "WARNING: ${#DATASETS[@]} datasets requested, but only ${NUM_GPUS} GPU(s) x ${DATASETS_PER_GPU} dataset(s)/GPU configured."
    log_job "WARNING: Datasets will be assigned in grouped blocks and then cycled across GPUs."
fi

log_job "Job started at $(date)"
log_job "Host: $(hostname)"
log_job "Job ID: ${SLURM_JOB_ID:-local}"
log_job "Datasets: ${DATASETS[*]}"
log_job "Batch sizes: ${BATCH_SIZES[*]}"
log_job "Rhos: ${RHOS[*]}"
log_job "Conda env: ${CONDA_ENV_NAME}"
log_job "GPUs: ${GPU_LIST[*]}"
log_job "Datasets per GPU: ${DATASETS_PER_GPU}"
log_job "Summary log: ${SUMMARY_LOG}"

OVERALL_STATUS=0
for batch_size in "${BATCH_SIZES[@]}"; do
    log_job ""
    log_job "===== batch_size=${batch_size} ====="
    PIDS=()
    PID_LABELS=()
    print_assignment_plan "$batch_size"

    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        gpu_id=$(assign_gpu_for_index "$i")

        run_training "$dataset" "$gpu_id" "$batch_size" &
        PIDS+=($!)
        PID_LABELS+=("${dataset}_batch${batch_size}@gpu${gpu_id}")
    done

    if ! wait_for_group; then
        OVERALL_STATUS=1
    fi

    if [ "${#BATCH_SUCCESSES[@]}" -gt 0 ]; then
        log_job "Completed for batch_size=${batch_size}: ${BATCH_SUCCESSES[*]}"
        JOB_SUCCESSES+=("${BATCH_SUCCESSES[@]}")
    fi
    if [ "${#BATCH_FAILURES[@]}" -gt 0 ]; then
        log_job "Failed for batch_size=${batch_size}: ${BATCH_FAILURES[*]}"
        JOB_FAILURES+=("${BATCH_FAILURES[@]}")
    fi
done

log_job ""
log_job "===== summary ====="
for batch_size in "${BATCH_SIZES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        run_id="${dataset}_batch${batch_size}"
        log_file="logs/train_${run_id}_${SLURM_JOB_ID:-local}.log"
        if [ -f "$log_file" ]; then
            log_job "--- ${run_id} ---"
            grep -E "(best_epoch|Final Training Summary|SUCCESS|ERROR|test_mse|test_mae)" "$log_file" | tail -20 | tee -a "$SUMMARY_LOG" || true
        fi
    done
done

log_job ""
if [ "${#JOB_SUCCESSES[@]}" -gt 0 ]; then
    log_job "Successful dataset runs: ${JOB_SUCCESSES[*]}"
fi
if [ "${#JOB_FAILURES[@]}" -gt 0 ]; then
    log_job "Failed dataset runs: ${JOB_FAILURES[*]}"
else
    log_job "Failed dataset runs: none"
fi

log_job "Job finished at $(date)"
exit "$OVERALL_STATUS"
