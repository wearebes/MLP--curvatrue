# MLP Curvature Pipeline

## Generate training data

```bash
python -m generate train
```

This will:

- generate `data/train_rho256.h5`
- generate `data/train_rho266.h5`
- generate `data/train_rho276.h5`
- print sample counts and validation info in the terminal

## Generate test data

```bash
python -m generate test
```

This will:

- generate test experiment folders under `data/`
- keep the existing explicit test setups:
  - `data/smooth_256/`
  - `data/smooth_266/`
  - `data/smooth_276/`
  - `data/acute_276/`
- print sample counts and validation info in the terminal

## Train models

```bash
python -m train rho 256 266 276
```

This will:

- train 3 separate models for `rho=256`, `rho=266`, `rho=276`
- read:
  - `data/train_rho256.h5`
  - `data/train_rho266.h5`
  - `data/train_rho276.h5`
- save:
  - `model/model_rho256.pth`
  - `model/model_rho266.pth`
  - `model/model_rho276.pth`
  - `model/zscore_stats_256.csv`
  - `model/zscore_stats_266.csv`
  - `model/zscore_stats_276.csv`
- print epoch logs in the terminal
- print train / val / test metrics for each epoch in the terminal
- save one final training summary to:
  - `train/results/result_MMDD.txt`
- save the full epoch-by-epoch training log to:
  - `train/results/train_log_MMDD.txt`

If you are inside `train/`, use:

```bash
python __main__.py rho 256 266 276
```

## Evaluate numerical / neural results

```bash
python -m test numerical
python -m test neural
python -m test numerical neural
```

This will:

- evaluate `hk = h * kappa`, not raw `kappa`
- use a full-field `div(normal)` numerical baseline for explicit test experiments
- use the same full-field `div(normal)` route for `train Numerical`, computed on the fly in memory
- keep the training `.h5` files unchanged; intermediate full-field values are not saved back into `.h5`
- evaluate both generated train data and generated test data
- for neural evaluation, load the 3 saved models:
  - `model/model_rho256.pth`
  - `model/model_rho266.pth`
  - `model/model_rho276.pth`
- load the matching z-score stats:
  - `model/zscore_stats_256.csv`
  - `model/zscore_stats_266.csv`
  - `model/zscore_stats_276.csv`
- save one unified result table to:
  - `test/results/result_MMDD.txt`

## Recommended order

```bash
python -m generate train
python -m generate test
python -m train rho 256 266 276
python -m test numerical neural
python view_case.py acute_276

```
CUDA Driver Version: 580.126.09
Current PyTorch CUDA Version: 12.6
WARNING: PyTorch CUDA is not available! Reinstalling PyTorch...
Installing CUDA 12.1 compatible PyTorch...
