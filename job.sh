#!/bin/bash
set -euo pipefail

#SBATCH --job-name=pde_training_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.log
#SBATCH --error=slurm_%x_%j.err
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --nodelist=hpcgpu06


# PDE Training Job Script for 北师港浸大 HPC Cluster
# 使用4块GPU并行训练四个数据集（data_001, data_002, data_004, data_005）
# 基于 HPC 使用手册 V2026313
# GPU队列说明：
#   - hpcgpu01: 4x V100 GPU
#   - hpcgpu04, hpcgpu05: 4x A100 GPU

cd "$(dirname "$0")" || exit 1
mkdir -p logs

# 默认训练四个训练数据集；可在提交时覆盖，例如：
# sbatch --export=ALL,DATASETS="data_001 data_002",RHOS="256 266 276",CONDA_ENV_NAME="jq" job.sh
DATASETS_STR=${DATASETS:-"data_001 data_002 data_004 data_005"}
RHOS_STR=${RHOS:-"256 266 276"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"jq"}

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a RHOS <<< "$RHOS_STR"

# 定义要训练的数据集列表
DATASETS=("data_001" "data_002" "data_004" "data_005")

# ENVIRONMENT SETUP - 北师港浸大 HPC 环境配置
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-<empty>}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-<empty>}"

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

# FIX: 使用系统 CUDA 替代 conda 环境中的 PyTorch CUDA
echo "Loading system CUDA..."

# 加载系统 CUDA 模块
module load cuda/12.4 || module load cuda/12.2 || module load cuda/11.8

# 验证 CUDA 环境
echo "CUDA_HOME: ${CUDA_HOME:-<empty>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<empty>}"
which nvcc

# 卸载 conda 环境中不兼容的 PyTorch CUDA 包，强制使用 CPU 版本
pip uninstall -y nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12 2>/dev/null || true

# 验证 PyTorch - 重新安装 CPU 版本作为后备，或者使用系统 CUDA
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || {
    echo "Reinstalling PyTorch with system CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    python -c "import torch; print('New PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
}

# 验证环境
echo "Conda env: $CONDA_ENV_NAME"
conda run -n "$CONDA_ENV_NAME" --no-capture-output python --version
echo "Python: $(which python)"
python --version
nvidia-smi -L || { echo "Failed to query GPUs via nvidia-smi"; exit 2; }

echo "PyTorch:"
conda run -n "$CONDA_ENV_NAME" --no-capture-output \
    python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}')" \
    2>/dev/null || echo "  PyTorch not available"


# TRAINING - 使用后台进程在4块GPU上并行训练四个数据集


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
# Use the original submission directory in Slurm to locate project files reliably.
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$WORKDIR" || exit 1
mkdir -p logs

# 获取GPU数量。
# 优先使用当前进程可见设备（通常已被 Slurm/cgroup 重映射为本地索引）。
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
VISIBLE_GPUS="${VISIBLE_GPUS// /}"

ASSIGNED_GPUS="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
ASSIGNED_GPUS="${ASSIGNED_GPUS// /}"

if [ -n "$VISIBLE_GPUS" ]; then
    IFS=',' read -r -a GPU_LIST <<< "$VISIBLE_GPUS"
    NUM_GPUS=${#GPU_LIST[@]}
elif [ -n "$ASSIGNED_GPUS" ]; then
    IFS=',' read -r -a GPU_LIST <<< "$ASSIGNED_GPUS"
    NUM_GPUS=${#GPU_LIST[@]}
else
    GPU_LIST=()
    NUM_GPUS=4
fi
echo "Starting training on $NUM_GPUS GPUs in parallel..."
if [ "${#GPU_LIST[@]}" -gt 0 ]; then
    echo "Resolved GPU IDs for child processes: ${GPU_LIST[*]}"
fi
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
    local train_status
    
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
    # 设置使用的GPU - 使用 env 在 Python 启动前设置环境变量
    env CUDA_VISIBLE_DEVICES=$gpu_id PYTHONUNBUFFERED=1 python -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 86)" \
        || { echo "[$(date)] CUDA is not available for $dataset on GPU binding $gpu_id" | tee -a "$log_file"; return 86; }
    
    # 训练指定数据集的三个rho值
    env CUDA_VISIBLE_DEVICES=$gpu_id PYTHONUNBUFFERED=1 python -m train dataset "$dataset" --rho 256 266 276 2>&1 | tee "$log_file"
    train_status=${PIPESTATUS[0]}
    if [ "$train_status" -ne 0 ]; then
        echo "[$(date)] Training failed for $dataset on GPU $gpu_id (status: $train_status)"
        return "$train_status"
    fi
    
    echo "[$(date)] Finished training for $dataset on GPU $gpu_id"
}

# 启动后台训练任务（每个数据集使用一个GPU）
PIDS=()
PID_DATASETS=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu_id=$((i % NUM_GPUS))

    if [ "${#GPU_LIST[@]}" -gt "$i" ]; then
        gpu_id="${GPU_LIST[$i]}"
    else
        gpu_id=$i
    fi
    
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
overall_status=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    status=$?
    echo "Process $pid finished with status: $status"
    if [ "$status" -ne 0 ]; then
        overall_status=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "One or more dataset training jobs failed." >&2
    exit 1
fi

# JOB COMPLETION
echo "All training completed at $(date)"
echo "Job ID: $SLURM_JOB_ID"

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

exit "$overall_status"
