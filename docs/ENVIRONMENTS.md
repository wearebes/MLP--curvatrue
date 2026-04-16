# Environment Setup

Use one file only:

- [environment.yaml](/e:/Research/PDE/code1/environment.yaml:1)

Recommended:

```bash
conda env create -f environment.yaml
conda activate code1-gpu
python -m train --env-check --require-cuda
```

This environment now targets PyTorch official `cu126` wheels.

`job.sh` now defaults to:

- `CONDA_ENV_NAME=code1-gpu`
- `CONDA_ENV_FILE=environment.yaml`
- job logs include `environment_report.json`, `gpu_preflight.log`, and `schedule_preflight.json`
