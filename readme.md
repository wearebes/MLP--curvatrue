# Neural Curvature MLP Pipeline

> 基于 Level-Set 重初始化的 MLP 曲率预测系统。从几何生成到模型评估的端到端流水线。

---

## Architecture

```
code1/
├── generate/              # Step 1-2: 数据生成
│   ├── config.yaml        # 数据生成配置（train_data + test_data）
│   ├── config.py          # 配置加载器
│   ├── pde.py             # Level-Set 重初始化核心（Godunov + WENO5 + TVD-RK3）
│   ├── numerics.py        # 数值工具（精确SDF、花形曲线、曲率计算）
│   ├── train_data.py      # 训练数据流水线（几何→场→重初始化→HDF5）
│   ├── test_data.py       # 测试数据流水线（高精度牛顿投影 exact_sdf → h·κ 目标）
│   └── __main__.py        # CLI 入口
│
├── train/                 # Step 3: 模型训练
│   ├── config.yaml        # 训练配置（optimizer, model, scheduler）
│   ├── config.py          # 配置加载器
│   ├── model.py           # CurvatureMLP（4层 ReLU, Xavier init）
│   ├── trainer.py         # 训练流水线（DataLoader→训练循环→多ρ并行）
│   └── __main__.py        # CLI 入口
│
├── test/                  # Step 4: 模型评估
│   ├── evaluator.py       # 评估主逻辑（numerical / neural / origin）
│   ├── utils.py           # NeuralPredictor, OriginPredictor, 指标工具
│   └── __main__.py        # CLI 入口
│
├── data/                  # 生成的 HDF5 数据
└── model/                 # 训练产出（权重 + z-score + 曲线图）
```

---

## Quick Start

### Prerequisites

```bash
conda activate jq
export PYTHONPATH=/path/to/code1    # 确保项目根目录在 Python 路径中
```

### Full Pipeline（4 步）

```bash
# Step 1: 生成训练数据
python -m generate train --dataset-name ds01

# Step 2: 生成测试数据
python -m generate test --dataset-name test01

# Step 3: 训练模型
python -m train --dataset ds01

# Step 4: 评估
python -m test numerical neural --data-split test
```

---

## Step 1 — Generate Training Data

```bash
python -m generate train --dataset-name <NAME> [--rho 256 266 276]
```

**流程**：CircleGeometry → SDF/nonSDF 场 → 重初始化(N步) → 3×3 stencil 提取 → HDF5 + 验证

| 参数             | 默认值        | 说明           |
| ---------------- | ------------- | -------------- |
| `--dataset-name` | *必填*        | 数据集文件夹名 |
| `--rho`          | `256 266 276` | 网格分辨率     |

**配置**（`generate/config.yaml`）：

```yaml
train_data:
  resolutions: [256, 266, 276]
  cfl: 0.5                        # 重初始化 CFL
  time_order: 3                   # TVD-RK3
  space_order: 5                  # WENO5
  reinit_steps: [5, 10, 15, 20]   # 重初始化步数变体
  geometry_seed: 42               # 几何随机种子
  variations: 5                   # 每个半径的中心偏移变体数
```

**输出**：`data/<NAME>/train_rho{ρ}.h5`

每个 HDF5 文件包含：
- `X` — 归一化 3×3 stencil `(N, 9)`
- `Y` — h·κ 标签 `(N, 1)`
- `reinit_steps` — 重初始化步数 `(N, 1)`
- `blueprint_idx`, `radius_idx` — 追踪元数据

---

## Step 2 — Generate Test Data

```bash
python -m generate test --dataset-name <NAME>
```

**流程**：花形曲线 `r(θ) = b + a·cos(pθ)` → 双精度高精度牛顿正交投影构造 exact SDF → 重初始化 → 解析投影 → h·κ 目标

**配置**（`generate/config.yaml`）：

```yaml
test_data:
  cfl: 0.5
  time_order: 3
  space_order: 5
  test_iters: [5, 10, 20]   # 评估的重初始化步数
  scenarios:
    - exp_id: smooth_256     # 实验 ID
      L: 0.207843            # 计算域半宽
      N: 107                 # 网格点数
      a: 0.05                # 花形基准半径
      b: 0.15                # 花形振幅
      p: 3                   # 花瓣数
```

**输出**：`data/<NAME>/<exp_id>/iter_{N}.h5` + `meta.json`

---

## Step 3 — Train Models

```bash
python -m train --dataset <NAME> [--rho 256 266 276] [--run-id <RUN>]
```

| 参数        | 默认值        | 说明                             |
| ----------- | ------------- | -------------------------------- |
| `--dataset` | *必填*        | 数据集 ID（对应 `data/<NAME>/`） |
| `--rho`     | `256 266 276` | 要训练的分辨率                   |
| `--run-id`  | 自动生成      | 模型保存路径标识                 |

**配置**（`train/config.yaml`）：

```yaml
seed: 42
batch_size: 128
max_epochs: 500
patience: 30                  # Early stopping 容忍轮数

optimizer:
  lr: 0.00015                 # Adam 学习率
  weight_decay: 0.0

model:
  input_dim: 9                # 3×3 stencil 展平
  default_hidden_dim: 128     # 隐藏层宽度
  hidden_dim_overrides:       # 按 ρ 覆盖
    266: 140
    276: 140
```

**模型结构**：`Linear(9→H) → ReLU → Linear(H→H) → ReLU → Linear(H→H) → ReLU → Linear(H→H) → ReLU → Linear(H→1)`

**初始化**：Xavier Uniform（权重） + Zeros（偏置）

**多 ρ 并行**：通过 `torch.multiprocessing` spawn 模式并行训练

**输出**（`model/<run_id>/`）：

```
model_rho256.pth              # 模型权重
zscore_stats_256.csv          # z-score 归一化参数
training_curves_rho256.png    # 训练曲线
training_rho256.log           # 训练日志
training_summary.txt          # 汇总表（.6e 科学计数法）
train_config.yaml             # 训练参数快照
```

---

## Step 4 — Evaluate

```bash
python -m test numerical neural --data-split test --run-id run01
```

**方法**：`numerical`（有限差分） | `neural`（MLP 推理） | `origin`/`paper`（原始模型）

| 参数           | 说明                                                                    |
| -------------- | ----------------------------------------------------------------------- |
| `methods`      | 一个或多个：`numerical` `neural` `origin`（`paper` 是 `origin` 别名）   |
| `--data-split` | 仅评估 `train` 或 `test`，省略则两者都评                                |
| `--run-id`     | 指定模型目录名（如 `model_data02_002`），省略则评估 `model/` 下所有模型 |
| `--source`     | 指定测试数据来源子目录（相对于 `data/`），见下方说明                    |

**数据选择逻辑**：

- **不传 `--source`**：默认在 `data/` 下查找含 `meta.json` 的子目录（如 `smooth_256/`、`acute_276/`）
- **传 `--source`**：路径相对于 `data/`，例如 `--source exact_sdf` → 查找 `data/exact_sdf/` 下的实验

```bash
# 用 data/ 下的默认测试数据
python -m test numerical neural --data-split test --run-id model_data02_002

# 用 data/exact_sdf/ 下的精确SDF测试数据
python -m test numerical neural --data-split test --run-id model_data02_002 --source exact_sdf

# 用 data/cfl05/ 下的测试数据
python -m test numerical neural --data-split test --run-id model_data02_002 --source cfl05
```

### 完整真实示例：从零到评估

```bash
# ── 环境准备 ──
conda activate jq
cd /mnt/e/Research/PDE/code1
export PYTHONPATH=.

# ── Step 1: 生成训练数据 ──
python -m generate train --dataset-name ds_smooth_01 --rho 256 266 276
# 输出：
#   data/ds_smooth_01/train_rho256.h5   (~800,000 samples)
#   data/ds_smooth_01/train_rho266.h5
#   data/ds_smooth_01/train_rho276.h5
#   data/ds_smooth_01/data_config.yaml

# ── Step 2: 生成测试数据（花形曲线 exact_sdf）──
python -m generate test --dataset-name test_flower_01
# 输出：
#   data/test_flower_01/smooth_256/meta.json
#   data/test_flower_01/smooth_256/iter_5.h5
#   data/test_flower_01/smooth_256/iter_10.h5
#   data/test_flower_01/smooth_256/iter_20.h5
#   data/test_flower_01/smooth_266/...
#   data/test_flower_01/smooth_276/...
#   data/test_flower_01/acute_276/...

# ── Step 3: 训练模型 ──
python -m train --dataset ds_smooth_01 --run-id run01
# 三个 ρ 并行训练，输出到 model/run01/
# 训练日志示例：
#   [rho=256][Epoch 001/500] LR=1.500e-04 | Train: MSE=2.345678e-03, MAE=3.456789e-02 | Val: MAE=3.210987e-02 | Test: MSE=2.109876e-03, MAE=3.098765e-02 <- best
#   [rho=256][Epoch 002/500] ...
#   [rho=256] Early stopping at epoch 180 (patience=30).
#   [rho=256] Training finished in 245.32s | best_epoch=150
#
# 输出文件：
#   model/run01/model_rho256.pth
#   model/run01/model_rho266.pth
#   model/run01/model_rho276.pth
#   model/run01/zscore_stats_256.csv
#   model/run01/training_curves_rho256.png
#   model/run01/training_summary.txt

# ── Step 4: 评估（核心）──

# 4a. 仅在测试数据上评估 numerical + neural
python -m test numerical neural --data-split test --run-id run01
python -m train --dataset data02_005 --run-id data02_005_batch256 --sequential
# 输出示例：
#   Evaluating test Neural using model directory: model/run01
#     smooth_256  iter=  5  Numerical: MAE=1.234567e-02  Neural: MAE=8.765432e-03
#     smooth_256  iter= 10  Numerical: MAE=9.876543e-03  Neural: MAE=5.432109e-03
#     smooth_256  iter= 20  Numerical: MAE=7.654321e-03  Neural: MAE=3.210987e-03
#     smooth_266  iter=  5  ...
#     acute_276   iter= 20  ...
#
#   Unified hk evaluation summary:
#   ┌────────────┬──────┬────────┬──────────────┬──────────────┬──────────────┐
#   │ experiment │ step │ method │     MAE      │     RMSE     │    MaxAE     │
#   ├────────────┼──────┼────────┼──────────────┼──────────────┼──────────────┤
#   │ smooth_256 │    5 │ Numer. │ 1.234567e-02 │ 1.567890e-02 │ 4.321098e-02 │
#   │ smooth_256 │    5 │ Neural │ 8.765432e-03 │ 1.098765e-02 │ 2.345678e-02 │
#   │ smooth_256 │   10 │ Numer. │ 9.876543e-03 │ 1.234567e-02 │ 3.456789e-02 │
#   │ smooth_256 │   10 │ Neural │ 5.432109e-03 │ 6.789012e-03 │ 1.567890e-02 │
#   │    ...     │  ... │  ...   │     ...      │     ...      │     ...      │
#   └────────────┴──────┴────────┴──────────────┴──────────────┴──────────────┘
#
#   Unified hk evaluation saved to model/run01/evals/unified_eval.csv

# 4b. 在训练数据上评估（验证训练质量）
python -m test numerical neural --data-split train --run-id run01

# 4c. 三方法对比（需要 model/origin/ 下有原始论文模型权重）
python -m test numerical neural origin --data-split test --run-id run01

# 4d. 指定自定义测试数据源
python -m test neural --data-split test --source test_flower_01 --run-id run01
```
python -m test numerical origin neural --data-split test --run-id model_data02_002 --source paper_formula_cfl095_eps25_dynsign

### 评估方法说明

| 方法        | 计算方式                                   | 用途                           |
| ----------- | ------------------------------------------ | ------------------------------ |
| `numerical` | 从场重建 3×3 stencil，有限差分公式计算 h·κ | 基准线：纯数值方法的精度上限   |
| `neural`    | 加载训练好的 MLP，z-score 归一化后推理 h·κ | 主结果：神经网络是否超越数值法 |
| `origin`    | 加载论文原始 Keras/TF 模型权重进行推理     | 对照：与原始论文的模型对比     |

---

## Key Design Decisions

| 设计点        | 选择                      | 原因                          |
| ------------- | ------------------------- | ----------------------------- |
| 训练 indexing | `ij`                      | 与 PyTorch tensor 布局一致    |
| 测试 indexing | `xy`                      | 与笛卡尔坐标一致（花形曲线）  |
| 配置          | 模块独立 YAML             | 无全局耦合，各模块可独立运行  |
| 精度输出      | `.6e` 科学计数法          | 保留 e-16 级别小值精度        |
| 数据对称      | ±stencil 镜像             | 利用 κ 的符号对称性扩充训练集 |
| 重初始化      | WENO5 + TVD-RK3 + Godunov | 高阶空间 + 高阶时间，数值稳定 |

---

## 数学报告

完整的数学表述、统一符号体系以及 `Method + Appendix` 体例的推导见 [docs/report.md](e:\Research\PDE\code1\docs\report.md)。
README 仅保留流水线与使用说明，不再承载完整的方法学推导。
