#!/bin/bash
set -euo pipefail

#SBATCH --job-name=pde_training_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.log
#SBATCH --error=slurm_%x_%j.err


# ============================================================================
# PDE Training Job Script for 北师港浸大 HPC Cluster
# 使用4块GPU并行训练四个数据集（data_001, data_002, data_004, data_005）
# 基于 HPC 使用手册 V2026313
# ============================================================================
# GPU队列说明：
#   - hpcgpu01: 4x V100 GPU
#   - hpcgpu04, hpcgpu05: 4x A100 GPU
# ============================================================================

cd "$(dirname "$0")" || exit 1
mkdir -p logs

# 默认训练四个训练数据集；可在提交时覆盖，例如：
# sbatch --export=ALL,DATASETS="data_001 data_002",RHOS="256 266 276",CONDA_ENV_NAME="jq" job.sh
DATASETS_STR=${DATASETS:-"data_001 data_002 data_004 data_005"}
RHOS_STR=${RHOS:-"256 266 276"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"jq"}

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a RHOS <<< "$RHOS_STR"


# ============================================================================
# ENVIRONMENT SETUP - 北师港浸大 HPC 环境配置
# ============================================================================
echo "=========================================="
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $SLURM_JOB_GPUS"
echo "=========================================="

if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/.bashrc" ]; then
        # shellcheck disable=SC1090
        source "$HOME/.bashrc"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found. Please load conda before submitting." >&2
    exit 1
fi

# 验证环境
echo "Conda env: $CONDA_ENV_NAME"
conda run -n "$CONDA_ENV_NAME" --no-capture-output python --version

echo "PyTorch:"
conda run -n "$CONDA_ENV_NAME" --no-capture-output \
    python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}')" \
    2>/dev/null || echo "  PyTorch not available"

echo "=========================================="

# ============================================================================
# TRAINING - 使用后台进程在4块GPU上并行训练四个数据集
# ============================================================================


# 获取GPU数量（优先使用 sbatch --gpus）
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    NUM_GPUS=4
fi

if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "ERROR: DATASETS is empty." >&2
    exit 1
fi

echo "Starting training on up to $NUM_GPUS GPUs in parallel..."
echo "Datasets to train: ${DATASETS[*]}"
echo "Rhos to train: ${RHOS[*]}"

for dataset in "${DATASETS[@]}"; do
    for rho in "${RHOS[@]}"; do
        expected_file="data/${dataset}/train_rho${rho}.h5"
        if [ ! -f "$expected_file" ]; then
            echo "ERROR: Missing training file: $expected_file" >&2
            exit 1
        fi
    done
done

# 创建后台训练函数
run_training() {
    local dataset=$1
    local gpu_id=$2
    local log_file="logs/train_${dataset}_${SLURM_JOB_ID}.log"
    
    echo "[$(date)] Starting training for $dataset on GPU $gpu_id"
    
    # 设置使用的GPU（每个后台任务绑定一个独立GPU）
    if CUDA_VISIBLE_DEVICES=$gpu_id \
        conda run -n "$CONDA_ENV_NAME" --no-capture-output \
        python -m train dataset "$dataset" --rho "${RHOS[@]}" >> "$log_file" 2>&1; then
        echo "[$(date)] Finished training for $dataset on GPU $gpu_id" | tee -a "$log_file"
    else
        echo "[$(date)] FAILED training for $dataset on GPU $gpu_id - check $log_file" | tee -a "$log_file"
        return 1
    fi
}

# 启动后台训练任务（每个数据集使用一个GPU）
PIDS=()
PID_DATASETS=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu_id=$((i % NUM_GPUS))

    run_training "$dataset" "$gpu_id" &
    PIDS+=($!)
    PID_DATASETS+=("$dataset")
    echo "Started training for $dataset on GPU $gpu_id (PID: $!)"
done

# 等待所有后台任务完成
echo "Waiting for all training jobs to complete..."
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    dataset="${PID_DATASETS[$i]}"
    if wait "$pid"; then
        echo "Process $pid ($dataset) finished successfully"
    else
        echo "Process $pid ($dataset) failed" >&2
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "One or more dataset training jobs failed." >&2
    exit 1
fi

# ============================================================================
# JOB COMPLETION
# ============================================================================
echo "=========================================="
echo "All training completed at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# 汇总训练结果
echo ""
echo "===== Training Summary ====="
for dataset in "${DATASETS[@]}"; do
    log_file="logs/train_${dataset}_${SLURM_JOB_ID}.log"
    if [ -f "$log_file" ]; then
        echo "--- $dataset ---"
        tail -20 "$log_file" | grep -E "(rho=|Final|Test MSE|Val MSE)" || echo "No summary found"
    fi
done

exit 0


