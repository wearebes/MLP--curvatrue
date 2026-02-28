# 环境配置与安装指南

## 快速开始（推荐：最小依赖）

如果只想跑模型训练，用这个更容易成功：

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 只装核心包，允许 pip 自动选择兼容版本
pip install -r requirements-minimal.txt
```

## 完整环境复现（需要精确匹配）

如果要 100% 复现开发环境，用完整的 `requirements.txt`：

```bash
pip install -r requirements.txt
```

**但可能遇到版本冲突**，原因：
- Python 版本不同（`requirements.txt` 固定在 Python 3.11）
- CUDA 版本差异（`torch==2.10.0+cu126` 要求 CUDA 12.6）
- 操作系统差异（包含 Windows 专有包）

## Windows GPU 支持（CUDA 12.6）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-minimal.txt
```

## CPU Only Mode

编辑 `requirements-minimal.txt`，改成：

```bash
torch==2.10.0  # 自动安装 CPU 版本
torchvision
pip install -r requirements-minimal.txt
```

## 遇到冲突时的解决方案

1. **删除冲突的包版本号**，让 pip 自动选择：
   ```bash
   pip install --upgrade torch tensorflow --no-deps
   ```

2. **使用 conda 环境文件**（如果装了 Conda）：
   ```bash
   conda env create -n ml_new -f environment.yml  # 用 environment.txt 重新生成 yml
   ```

3. **逐个安装关键包**：
   ```bash
   pip install torch
   pip install tensorflow keras
   pip install h5py
   # ...以此类推
   ```

## Python 版本要求

- 推荐：**Python 3.11+**
- 最低：**Python 3.9**
- 检查版本：`python --version`

## 常见错误

| 错误 | 解决方案 |
|------|--------|
| `No module named torch` | `pip install torch` |
| `CUDA version mismatch` | 用 CPU 版本：`pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `file:// path error` | 已修复，用新的 `requirements.txt` |

## 文件说明

- `requirements.txt` — 完整的 pip 包清单（精确版本）
- `requirements-minimal.txt` — 核心包只（推荐跨机器安装）
- `environment.txt` — Conda 格式（仅 Conda 用户）
