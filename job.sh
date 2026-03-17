#!/bin/bash
#SBATCH --job-name=pde_training_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err


# ============================================================================
# PDE Training Job Script for 北师港浸大 HPC Cluster
# 使用4块GPU并行训练四个数据集（data_001, data_002, data_004, data_005）
# 基于 HPC 使用手册 V2026313
# ============================================================================
# GPU队列说明：
#   - hpcgpu01: 4x V100 GPU
#   - hpcgpu04, hpcgpu05: 4x A100 GPU
# ============================================================================

# 定义要训练的数据集列表
DATASETS=("data_001" "data_002" "data_004" "data_005")

mkdir -p logs

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

# 激活 Conda 环境（根据实际修改环境名称）
source ~/.bashrc
conda activate ml

# 验证环境
echo "Python: $(which python)"
python --version

echo "PyTorch:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "  PyTorch not available"

echo "=========================================="

# ============================================================================
# TRAINING - 使用后台进程在4块GPU上并行训练四个数据集
# ============================================================================

cd "$(dirname "$0")" || exit 1

# 获取GPU数量
NUM_GPUS=${SLURM_JOB_GPUS:-4}
echo "Starting training on $NUM_GPUS GPUs in parallel..."
echo "Datasets to train: ${DATASETS[*]}"

# 创建后台训练函数
run_training() {
    local dataset=$1
    local gpu_id=$2
    local log_file="logs/train_${dataset}_${SLURM_JOB_ID}.log"
    
    echo "[$(date)] Starting training for $dataset on GPU $gpu_id"
    
    # 设置使用的GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # 训练指定数据集的三个rho值
    python -m train dataset "$dataset" --rho 256 266 276 2>&1 | tee "$log_file"
    
    echo "[$(date)] Finished training for $dataset on GPU $gpu_id"
}

# 启动后台训练任务（每个数据集使用一个GPU）
PIDS=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu_id=$i  # GPU 0, 1, 2, 3
    
    run_training "$dataset" "$gpu_id" &
    PIDS+=($!)
    echo "Started training for $dataset on GPU $gpu_id (PID: ${PIDS[-1]})"
done

# 等待所有后台任务完成
echo "Waiting for all training jobs to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Process $pid finished with status: $?"
done

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


