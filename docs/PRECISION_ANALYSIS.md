# 训练数据生成的精度分析

## 📊 快速答案

| 阶段 | 精度 | 说明 |
|------|------|------|
| **字段计算** | `float64` (双精度) | 生成 level-set 字段 |
| **存储格式** | `float32` (单精度) | HDF5 文件中保存 |
| **训练读入** | `float32` (单精度) | PyTorch 中加载 |

---

## 🔍 详细代码分析

### 1️⃣ 字段生成阶段（float64）

#### `generate/field_builder.py` line 12

```python
class LevelSetFieldBuilder:
    def __init__(self, dtype=np.float64):  # ✅ 默认 float64
        self.dtype = dtype
        self.spacing_atol = 1e-9
```

#### `generate/field_builder.py` line 38-42

```python
def _build_grid(self, rho: int):
    h = 1.0 / (rho - 1)
    x = np.linspace(0.0, 1.0, rho, dtype=self.dtype)  # ✅ dtype=float64
    y = np.linspace(0.0, 1.0, rho, dtype=self.dtype)  # ✅ dtype=float64
    X, Y = np.meshgrid(x, y, indexing='ij')
    return x, y, X, Y, float(h)
```

**结论**：所有 level-set 字段计算（phi, 梯度等）都用 **float64**

---

### 2️⃣ 文件存储阶段（float32）

#### `generate/dataset_compiler.py` line 17-20

```python
class HDF5DatasetCompiler:
    def __init__(self, h5_filepath: str, mode: str = "w"):
        if mode == "w":
            # ✅ X 和 Y 都存储为 float32
            self.X_dset = self.file.create_dataset(
                "X", shape=(0, 9), maxshape=(None, 9), 
                dtype=np.float32,  # ← 关键
                compression="gzip"
            )
            self.Y_dset = self.file.create_dataset(
                "Y", shape=(0, 1), maxshape=(None, 1), 
                dtype=np.float32,  # ← 关键
                compression="gzip"
            )
```

#### `generate/dataset_compiler.py` line 50-52

```python
@staticmethod
def extract_stencils(field_pack: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # ... 从 float64 的 phi 中提取 ...
    
    # ✅ 转换为 float32 后保存
    X_b = patches.astype(np.float32)      # float64 → float32
    Y_b = np.full((X_b.shape[0], 1), h_kappa, dtype=np.float32)
    
    # 增强
    X_batch_aug = np.vstack([X_b, -X_b])  # 仍然是 float32
    Y_batch_aug = np.vstack([Y_b, -Y_b])
    
    return X_batch_aug, Y_batch_aug  # 返回 float32
```

**结论**：保存到 HDF5 前转换为 **float32**（节省空间）

---

### 3️⃣ 训练读入阶段（float32）

#### `train/dataloader.py` line 28-29

```python
x_np = handle["X"][:]  # 从 HDF5 读入 (float32)
y_np = handle["Y"][:]  # 从 HDF5 读入 (float32)

# ✅ 转为 PyTorch tensor，保持 float32
features = torch.tensor(x_np, dtype=torch.float32)  # float32
targets = torch.tensor(y_np, dtype=torch.float32)   # float32
```

**结论**：在 PyTorch 中保持 **float32** 训练

---

## 📈 精度转换流程图

```
┌─────────────────────────────────────────────────────────┐
│ 1. 字段生成                                              │
│    Level-Set Field Builder                              │
│    ├─ 网格：float64                                     │
│    ├─ phi = (X-cx)² + (Y-cy)² - r²：float64            │
│    ├─ 重初始化：float64                                 │
│    └─ 梯度计算：float64                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Stencil 提取                                          │
│    ├─ 从 float64 的 phi 中提取 3×3 patches             │
│    ├─ 提取结果暂时：float64                             │
│    └─ 然后 astype(float32) 转换                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. HDF5 存储                                             │
│    ├─ X (9维 stencil)：float32                         │
│    ├─ Y (h*kappa)：float32                             │
│    └─ 压缩格式：gzip                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 训练加载                                              │
│    ├─ 从 HDF5 读入：float32                             │
│    ├─ 转为 PyTorch tensor：float32                      │
│    └─ 训练：float32                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🔢 精度损失分析

### float64 → float32 的精度损失

**float64**（双精度）：
- 尾数：52 位
- 精度：~15-17 位十进制数字

**float32**（单精度）：
- 尾数：23 位
- 精度：~6-7 位十进制数字

### 具体损失量级

对于典型的 stencil 值（~10^-3）：

```
float64: 1.234567890123456e-03  (15位精度)
float32: 1.234568e-03           (6位精度)
误差:    ~1e-09 (相对误差 ~1e-6)
```

**结论**：对于网格间距 h ~ 10^-3，float32 的相对精度损失约 **0.0001%**，可以接受

---

## ⚙️ 配置检查

### 检查 float64 是否被强制指定

```python
# generate/config.py
# （无 dtype 配置，使用默认 float64）

# generate/field_builder.py 
class LevelSetFieldBuilder:
    def __init__(self, dtype=np.float64):  # ✅ 硬编码 float64
```

### 检查 float32 是否被强制指定

```python
# generate/dataset_compiler.py
self.X_dset = self.file.create_dataset("X", ..., dtype=np.float32)
self.Y_dset = self.file.create_dataset("Y", ..., dtype=np.float32)

# train/dataloader.py
features = torch.tensor(x_np, dtype=torch.float32)
targets = torch.tensor(y_np, dtype=torch.float32)
```

✅ **都是硬编码的**，不可配置

---

## 📊 数据大小对比

假设 3.14M 样本（单个 rho）：

| 格式 | 数据量 | 压缩后 |
|------|--------|--------|
| **float64** | 3.14M × 9 × 8 B = 226 MB | ~ 100 MB |
| **float32** | 3.14M × 9 × 4 B = 113 MB | ~ 50 MB |
| **节省比例** | - | **50%** |

✅ float32 可以有效减少磁盘空间和 I/O 时间

---

## ⚠️ 重要注意事项

### 1️⃣ 精度选择的合理性

✅ **合理**，因为：
- 生成阶段（float64）确保了计算的高精度
- 存储阶段（float32）在可接受的误差范围内
- 训练任务不需要超过 float32 的精度

### 2️⃣ 测试数据的精度

检查测试数据的精度：

```python
# generate/test_data.py
def extract_3x3_stencils(phi: np.ndarray, indices: np.ndarray) -> np.ndarray:
    stencils = []
    for row, col in indices:
        patch = phi[row - 1:row + 2, col - 1:col + 2]
        stencils.append(patch.flatten())
    return np.array(stencils, dtype=np.float64)  # ← float64！
```

⚠️ **不一致**：训练数据是 float32，测试数据是 float64

### 3️⃣ 推理时的精度

训练时用 float32，推理时需要确保也用 float32：

```python
# train/inference.py (假设代码)
# 需要确保：features 是 float32，与训练时一致
```

---

## 📋 完整精度表

```
┌──────────────────┬──────────┬──────────────────────────┐
│ 阶段             │ 精度     │ 说明                     │
├──────────────────┼──────────┼──────────────────────────┤
│ 网格构建         │ float64  │ np.linspace(dtype=...)   │
│ Level-set 计算   │ float64  │ (X-cx)² + (Y-cy)² - r²   │
│ 重初始化         │ float64  │ WENO5 求解 PDE           │
│ Stencil 提取     │ float64  │ 从 phi[i-1:i+2, j-1:j+2] │
│ Stencil 转换     │ float32  │ astype(float32)          │
│ HDF5 存储        │ float32  │ create_dataset dtype=... │
│ 训练读入         │ float32  │ torch.tensor dtype=...   │
│ 模型计算         │ float32  │ PyTorch default          │
│ 测试数据         │ float64  │ ⚠️ 不一致！              │
└──────────────────┴──────────┴──────────────────────────┘
```

---

## 🎯 建议

### 如果想统一为 float64（更高精度）

修改 `generate/dataset_compiler.py`：
```python
self.X_dset = self.file.create_dataset("X", ..., dtype=np.float64)
self.Y_dset = self.file.create_dataset("Y", ..., dtype=np.float64)
```

修改 `train/dataloader.py`：
```python
features = torch.tensor(x_np, dtype=torch.float64)
targets = torch.tensor(y_np, dtype=torch.float64)
```

**代价**：内存和磁盘占用翻倍

### 如果想统一为 float32（当前配置）

✅ **已经这样做了**（除了测试数据例外）

### 如果想统一测试数据的精度

修改 `generate/test_data.py`：
```python
return np.array(stencils, dtype=np.float32)  # 改为 float32
```

---

## 总结

| 问题 | 答案 |
|------|------|
| 生成阶段精度？ | **float64**（高精度） |
| 存储精度？ | **float32**（节省空间） |
| 训练精度？ | **float32**（PyTorch 默认） |
| 是否一致？ | ⚠️ 部分不一致（测试数据用 float64） |
| 建议？ | ✅ 当前配置可接受，或统一改为 float64 |

