# Neural Curvature Pipeline Usage Guide

This document provides instructions on how to use the newly refactored pipeline.

## 1. Generating Data

All data generation scripts are housed under `generate/`.
To generate training data, use:
```bash
python -m generate train --dataset-name ds01 --rho 64 128
```
To generate test/evaluation data:
```bash
python -m generate test --dataset-name ds01
```
默认测试生成会读取 [`generate/config.yaml`](/e:/Research/PDE/code1/generate/config.yaml) 里的 `test_data.mode`，当前默认是 `exact_sdf`。
当前支持：
- `formula_phi0`: 论文 irregular flower 口径，先用花形公式初值再做 reinit
- `exact_sdf`: 先构造 exact SDF 再做 reinit

也可以命令行覆盖，例如：
```bash
python -m generate test --dataset-name ds_formula --mode formula_phi0
python -m generate test --dataset-name ds_exact --mode exact_sdf
```

## 2. Training Models

Training is centralized in `train/`. It automatically picks up the dataset and trains the configured rhos in parallel.
```bash
python -m train --dataset ds01 --run-id run_ds01_bs128
```
All training artifacts, logs, and compiled weights are saved directly to `model/run_ds01_bs128/`.

## 3. Evaluation

Evaluation is centralized in `test/`. By referencing the `--run-id`, the script will automatically discover and evaluate all trained resolution models bound to that run.
```bash
python -m test neural --run-id run_ds01_bs128 --source ds01
```
The results are saved as `unified_evaluation.txt` inside `model/run_ds01_bs128/evals/`.
