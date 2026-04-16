# Validation Guide

## Purpose

This project now uses two validation tracks:

- CPU validation: development server, CLI checks, artifact checks, smoke runs.
- HPC GPU validation: scheduler, CUDA, multi-rho execution, resource downgrade behavior.

The CPU path should be run frequently. The HPC path should be run before or after meaningful scheduler or environment changes.

## CPU Validation

### Fast smoke

Run the integrated smoke script:

```bash
python test/smoke_train_pipeline.py
```

This checks:

- `env-check`
- `validate-data` success path
- `validate-data` failure path
- `dataset-run` in CPU-allowed mode
- `verify-run`
- `schedule-run` downgrade path
- `verify-schedule`

### Manual CPU checklist

```bash
python -m train --env-check
python -m train --dataset tiny_ds --config test/fixtures/phase2_smoke/train_config.yaml --rho 256 266 --validate-data
python -m train --dataset tiny_ds --config test/fixtures/phase2_smoke/train_config.yaml --run-id local_check --rho 256 266 --dataset-run --rho-max-concurrent 2 --allow-cpu --max-epochs 1 --patience 1
python -m train --config test/fixtures/phase2_smoke/train_config.yaml --verify-run --run-id local_check
```

Expected outcome:

- `env-check` prints a valid report.
- `validate-data` returns zero on valid data.
- `dataset-run` completes and writes model artifacts.
- `verify-run` returns `status: ok`.

## HPC GPU Validation

### Preflight

Before real training:

```bash
python -m train --env-check --require-cuda
```

Expected outcome:

- exit code `0`
- `cuda_available: true`
- `cuda_device_count >= 1`

### Single dataset smoke

```bash
python -m train \
  --dataset <dataset> \
  --config train/config.yaml \
  --rho 256 266 276 \
  --dataset-run \
  --rho-max-concurrent 1 \
  --max-epochs 1 \
  --patience 1
```

Expected outcome:

- run completes without CUDA initialization failure
- `verify-run` returns `status: ok`

### Concurrency smoke

```bash
python -m train \
  --dataset <dataset> \
  --config train/config.yaml \
  --rho 256 266 276 \
  --dataset-run \
  --rho-max-concurrent 3 \
  --max-epochs 1 \
  --patience 1
```

Expected outcome:

- all rho workers start
- `dataset_job_state.txt` records effective concurrency
- if resource failures occur, failure categories are visible in metrics/failure summaries

### Schedule smoke through `job.sh`

```bash
DATASETS="<dataset1> <dataset2>" \
RHOS="256 266 276" \
BATCH_SIZES="32" \
TRAIN_CONFIG_PATH="train/config.yaml" \
RHO_MAX_CONCURRENT=3 \
sbatch job.sh
```

Expected outcome:

- GPU preflight passes
- `job.sh` calls Python `--schedule-run`
- schedule summary is written under `model/_schedules/`

## Artifact Verification

### Verify a run

```bash
python -m train --config train/config.yaml --verify-run --run-id <run_id>
```

### Verify a schedule

```bash
python -m train --verify-schedule-path model/_schedules/<summary>.json
```

## Interpretation

- `validate-data` failure means input data problem, not trainer problem.
- `env-check --require-cuda` failure means environment or scheduler problem, not model problem.
- `verify-run` failure means artifact completeness problem.
- `verify-schedule` failure means schedule bookkeeping or referenced artifact problem.
