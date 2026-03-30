#!/bin/bash
set -uo pipefail

#SBATCH --job-name=pde_train_bs128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

BATCH_SIZES=128 bash "$(dirname "$0")/job.sh" "$@"
