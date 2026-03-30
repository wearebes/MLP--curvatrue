# Neural Curvature Pipeline — 完整数学原理文档

> **目标**：用 MLP 从 3×3 level-set 模板 $\phi$ 预测量纲化曲率 $h\kappa$，
> 使其精度 MAE $\approx 10^{-6}$，明显优于经典有限差分法（MAE $\approx 10^{-3}$–$10^{-4}$）。

---

## 符号总表

| 符号 | 含义 | 典型数值 |
|---|---|---|
| $\Omega$ | 计算域 | $[0,1]^2$（训练）/ $[-L,L]^2$（测试） |
| $L$ | 测试域半宽 | $\approx 0.208$ |
| $N$ | 每边网格点数 | $\rho$（训练）/ $107$–$129$（测试） |
| $\rho$ | 训练分辨率参数 | $256, 266, 276$ |
| $h$ | 网格间距 | $1/(\rho-1)$，例如 $1/255 \approx 3.922\times10^{-3}$ |
| $\phi_{i,j}$ | level-set 函数值 | 有符号距离或代数值 |
| $\kappa$ | 界面平均曲率（有符号） | $\approx 1/r$ |
| $h\kappa$ | 量纲化曲率（无量纲） | $\in[-1,1]$ |
| $\Gamma$ | 零等值线 $\{\phi=0\}$ 即界面 | — |
| $a,b,p$ | flower 曲线参数 | $a=0.05,b=0.15,p=3$ |
| $\text{CFL}$ | Courant 数 | $0.5$（训练）/ $0.95$（测试） |
| $\varepsilon_s$ | smooth sign 正则化系数 | $1.0$（训练）/ $2.5$（测试）|

---

## 第一部分：训练数据生成

### 1.1 计算域与网格

训练域为单位正方形 $[0,1]^2$，采用 `ij` 索引约定：

$$
x_i = \frac{i}{\rho-1}, \quad y_j = \frac{j}{\rho-1}, \qquad i,j = 0,1,\ldots,\rho-1
$$

网格间距：
$$
h = \frac{1}{\rho - 1}
$$

**具体数值**（对应 `resolutions: [256, 266, 276]`）：

| $\rho$ | $h$ |
|---|---|
| $256$ | $1/255 \approx 3.9216\times10^{-3}$ |
| $266$ | $1/265 \approx 3.7736\times10^{-3}$ |
| $276$ | $1/275 \approx 3.6364\times10^{-3}$ |

---

### 1.2 圆形 Geometry 蓝图生成

#### 1.2.1 半径集合

给定分辨率 $\rho$，生成 $N_r$ 个均匀分布的圆半径：

$$
r_{\min} = 1.6\,h, \quad r_{\max} = 0.5 - 2.0\,h
$$

$$
N_r = \left\lfloor \frac{\rho - 8.2}{2} \right\rfloor + 1
$$

$$
r_k = r_{\min} + \frac{k}{N_r-1}(r_{\max} - r_{\min}), \quad k = 0, 1, \ldots, N_r-1
$$

**具体数值**（$\rho=256$）：
$$
r_{\min} = 1.6 \times \frac{1}{255} \approx 6.275\times10^{-3}, \quad
r_{\max} = 0.5 - \frac{2}{255} \approx 0.4922, \quad
N_r = \lfloor (256-8.2)/2 \rfloor + 1 = 124
$$

#### 1.2.2 圆心位置随机化

每个半径 $r_k$ 对应 $V=5$ 个 variation，圆心随机扰动：

$$
c_x, c_y \sim \mathcal{U}\!\left[0.5 - \frac{h}{2},\; 0.5 + \frac{h}{2}\right]
$$

此随机化使训练样本覆盖界面相对于网格的所有亚格子偏移位置，防止模型过拟合到网格对齐的界面。

**每个分辨率的蓝图总数**：$N_r \times V = 124 \times 5 = 620$（$\rho=256$）

#### 1.2.3 解析曲率标签

圆半径为 $r$ 的圆，曲率和量纲化曲率：

$$
\kappa = \frac{1}{r}, \qquad h\kappa = \frac{h}{r}
$$

---

### 1.3 Level-Set 初始场构建

#### 1.3.1 精确 SDF 初值

对圆心 $(c_x, c_y)$，半径 $r$，精确带符号距离函数（SDF）：

$$
\phi^{(0)}_{i,j} = \sqrt{(x_i - c_x)^2 + (y_j - c_y)^2} - r
$$

此场满足 Eikonal 方程 $|\nabla\phi^{(0)}| = 1$，是真正的 SDF。

#### 1.3.2 非 SDF 代数初值

非 SDF 形式：

$$
\phi^{(0)}_{i,j} = (x_i - c_x)^2 + (y_j - c_y)^2 - r^2
$$

此场 $|\nabla\phi^{(0)}| \neq 1$，需要重初始化才能用于曲率计算。

---

### 1.4 Level-Set 重初始化 PDE

重初始化目标：将任意 level-set 函数演化为带符号距离函数，即使 $|\nabla\phi| \to 1$。

#### 1.4.1 重初始化 PDE（连续形式）

$$
\frac{\partial \phi}{\partial \tau} + S(\phi_0)\left(|\nabla\phi| - 1\right) = 0
$$

其中 $\tau$ 是伪时间（不是物理时间），$\phi_0$ 是初始场，$S$ 是平滑符号函数：

$$
S(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0^2 + \varepsilon_s^2 h^2}}
$$

**具体数值**（训练，`eps_sign_factor=1.0`）：

$$
\varepsilon_s = 1.0, \quad \varepsilon = \varepsilon_s \cdot h = 1.0 \times \frac{1}{255} \approx 3.922\times10^{-3}
$$

**注**：`sign_mode=frozen_phi0` 表示 $S$ 在整个演化过程中固定使用 $\phi_0$；
`sign_mode=dynamic_phi` 表示每步用当前 $\phi$ 更新 $S$。

#### 1.4.2 Godunov 迎风 Hamiltonian 格式

对 Hamilton-Jacobi 方程 $|\nabla\phi| = G(\nabla\phi) = 1$，采用 Rouy-Tourin 型 Godunov 格式：

设 $D^-_x, D^+_x$ 为 $x$ 方向向后/向前差分，$D^-_y, D^+_y$ 同理，则离散的迎风梯度模：

对 $S \geq 0$（$\phi_0 > 0$，即界面外侧）：
$$
G_{i,j}^+ = \sqrt{\left[\max\!\left(\max(D^-_x\phi_{i,j},\, 0),\, -\min(D^+_x\phi_{i,j},\, 0)\right)\right]^2 + \left[\max\!\left(\max(D^-_y\phi_{i,j},\, 0),\, -\min(D^+_y\phi_{i,j},\, 0)\right)\right]^2}
$$

对 $S < 0$（$\phi_0 < 0$，即界面内侧）：
$$
G_{i,j}^- = \sqrt{\left[\max\!\left(\max(-D^-_x\phi_{i,j},\, 0),\, \min(D^+_x\phi_{i,j},\, 0)\right)\right]^2 + \left[\max\!\left(\max(-D^-_y\phi_{i,j},\, 0),\, \min(D^+_y\phi_{i,j},\, 0)\right)\right]^2}
$$

综合：
$$
G_{i,j} = \begin{cases} G_{i,j}^+ & S_{i,j} \geq 0 \\ G_{i,j}^- & S_{i,j} < 0 \end{cases}
$$

右端项：
$$
\mathcal{L}(\phi)_{i,j} = -S_{i,j}\left(G_{i,j} - 1\right)
$$

#### 1.4.3 空间导数：WENO5 格式

采用 5 阶 WENO（Hamilton-Jacobi 型）格式计算 $D^\pm$ 差分。

**一维 WENO5-HJ 公式**（以向后差分 $D^-$ 为例，输入为连续差分值 $v_k$）：

先计算 5 个第一阶差分：
$$
v_1 = \frac{\phi_{i-2}-\phi_{i-3}}{h}, \quad v_2 = \frac{\phi_{i-1}-\phi_{i-2}}{h}, \quad v_3 = \frac{\phi_i-\phi_{i-1}}{h}, \quad v_4 = \frac{\phi_{i+1}-\phi_i}{h}, \quad v_5 = \frac{\phi_{i+2}-\phi_{i+1}}{h}
$$

光滑性指标（smoothness indicators）：
$$
\beta_0 = \frac{13}{12}(v_1 - 2v_2 + v_3)^2 + \frac{1}{4}(v_1 - 4v_2 + 3v_3)^2
$$
$$
\beta_1 = \frac{13}{12}(v_2 - 2v_3 + v_4)^2 + \frac{1}{4}(v_2 - v_4)^2
$$
$$
\beta_2 = \frac{13}{12}(v_3 - 2v_4 + v_5)^2 + \frac{1}{4}(3v_3 - 4v_4 + v_5)^2
$$

理想权重：$d_0 = 0.1,\; d_1 = 0.6,\; d_2 = 0.3$（共 0.1+0.6+0.3=1）

非线性权重（$\varepsilon_w = 10^{-6}$）：
$$
\alpha_k = \frac{d_k}{(\beta_k + \varepsilon_w)^2}, \qquad \omega_k = \frac{\alpha_k}{\alpha_0+\alpha_1+\alpha_2}
$$

三个候选多项式重建（向后差分）：
$$
p_0 = \frac{1}{3}v_1 - \frac{7}{6}v_2 + \frac{11}{6}v_3, \qquad
p_1 = -\frac{1}{6}v_2 + \frac{5}{6}v_3 + \frac{1}{3}v_4, \qquad
p_2 = \frac{1}{3}v_3 + \frac{5}{6}v_4 - \frac{1}{6}v_5
$$

WENO5 重建：
$$
D^-\phi_{i} = \omega_0 p_0 + \omega_1 p_1 + \omega_2 p_2
$$

向前差分 $D^+$ 采用镜像反转（$v_k$ 序列颠倒），其余公式形式相同。

**边界处理**：全场以 `edge` 方式镜像填充 3 个边界单元，确保 WENO5 模板不越界。

#### 1.4.4 时间积分：3 阶 TVD Runge-Kutta（`time_order=3`）

设 CFL 步长：

$$
\Delta\tau = \text{CFL} \times h = 0.5 \times \frac{1}{\rho-1}
$$

**具体数值**（$\rho=256$）：
$$
\Delta\tau = 0.5 / 255 \approx 1.961\times10^{-3}
$$

TVD-RK3 三级格式（Shu-Osher 1988）：

$$
\phi^{(1)} = \phi^n + \Delta\tau\,\mathcal{L}(\phi^n)
$$
$$
\phi^{(2)} = \frac{3}{4}\phi^n + \frac{1}{4}\left[\phi^{(1)} + \Delta\tau\,\mathcal{L}(\phi^{(1)})\right]
$$
$$
\phi^{n+1} = \frac{1}{3}\phi^n + \frac{2}{3}\left[\phi^{(2)} + \Delta\tau\,\mathcal{L}(\phi^{(2)})\right]
$$

每级均用冻结（或动态）符号场 $S$，由 `sign_mode` 控制。

#### 1.4.5 重初始化步数

训练数据对非 SDF 初值运行 $n_s \in \{5, 10, 15, 20\}$ 步（`reinit_steps: [5, 10, 15, 20]`）。

**伪时间总量**（以 $\rho=256, n_s=20$ 为例）：
$$
\tau_{\text{total}} = n_s \times \Delta\tau = 20 \times 1.961\times10^{-3} \approx 0.03922 \approx 10h
$$

精确 SDF 初值直接记为 $n_s=0$（进 pipeline 时不做重初始化）。

---

### 1.5 界面相邻节点采样

#### 1.5.1 采样准则

对重初始化后的场 $\phi$，识别**界面相邻节点**：所有与 $\phi$ 发生符号变化的边相邻的节点：

水平方向符号变化：
$$
\phi_{i,j} \cdot \phi_{i,j+1} \leq 0
$$

垂直方向符号变化：
$$
\phi_{i,j} \cdot \phi_{i+1,j} \leq 0
$$

将这些边两端的节点均纳入集合 $\mathcal{I}$，并去除边界后得到采样索引集 $\mathcal{I} = \{(i,j)\}$。

#### 1.5.2 样本数量估计

对圆形界面，界面长度 $\approx 2\pi r$，相邻节点约为界面两侧各 1 层，合计约：
$$
M \approx \frac{2 \times 2\pi r}{h}
$$

**具体数值**（$\rho=256$，中等半径 $r\approx 0.25$）：
$$
M \approx \frac{4\pi \times 0.25}{1/255} \approx 800
$$

每个蓝图 1 个 SDF + 4 个 reinit 步数，共约 $620 \times 5 = 3100$ 个场，每场约 $800$ 个节点：原始样本 $\approx 3100 \times 800 = 2{,}480{,}000$。加上以下正负对称扩增，最终约 $5\times10^6$ 个样本对。

---

### 1.6 3×3 模板提取与编码

#### 1.6.1 模板提取

对每个采样节点 $(i,j)$，提取中心化 3×3 窗口：

$$
P_{i,j} = \{\phi_{i+\delta_i, j+\delta_j}\}_{(\delta_i,\delta_j)\in\{-1,0,1\}^2}
$$

#### 1.6.2 模板展平编码（training\_order）

代码中 `encode_patch_training_order` 把 2D patch 先**沿列翻转**再**转置**后展平：

```
patch_2d  (3×3 原始，row=i-axis, col=j-axis)

     j-1   j    j+1
i-1 [a00, a01, a02]
i   [a10, a11, a12]
i+1 [a20, a21, a22]

翻转列: [[a02,a01,a00],[a12,a11,a10],[a22,a21,a20]]
转置:   [[a02,a12,a22],[a01,a11,a21],[a00,a10,a20]]
展平:   [a02,a12,a22, a01,a11,a21, a00,a10,a20]
```

因此网络输入的 9 维向量按如下顺序对应网格：

| 向量下标 | 节点 | 网格位置 |
|---|---|---|
| 0 | $(i-1, j+1)$ | 右上 |
| 1 | $(i,\ \ j+1)$ | 右 |
| 2 | $(i+1, j+1)$ | 右下 |
| 3 | $(i-1, j\ \ )$ | 上 |
| 4 | $(i,\ \ j\ \ )$ | **中心** |
| 5 | $(i+1, j\ \ )$ | 下 |
| 6 | $(i-1, j-1)$ | 左上 |
| 7 | $(i,\ \ j-1)$ | 左 |
| 8 | $(i+1, j-1)$ | 左下 |

**注**：此编码与 `ij` 轴约定对应，代表网格中东南西北的物理方向。

#### 1.6.3 正负对称扩增

对每个节点提取的 9 元组 $\mathbf{x}$ 和标签 $y = h\kappa$，**同时**存入正样本和负样本：

$$
\mathcal{D} \ni (\mathbf{x},\, y) \quad \text{和} \quad (-\mathbf{x},\, -y)
$$

物理意义：将 level-set 场取负（$\phi \to -\phi$）等价于界面内外翻转，曲率也取负，因此网络天然应满足奇对称性 $f(-\mathbf{x}) = -f(\mathbf{x})$，此扩增强制约束。

#### 1.6.4 标签

每个节点的标签为（与圆半径 $r$ 和网格间距 $h$ 无关地均匀赋值）：

$$
y = h\kappa_{\text{analytic}} = \frac{h}{r}
$$

**典型范围**（$\rho=256$）：

$$
y_{\min} = \frac{h}{r_{\max}} = \frac{1/255}{0.4922} \approx 7.98\times10^{-3}, \qquad
y_{\max} = \frac{h}{r_{\min}} = \frac{1/255}{6.275\times10^{-3}} \approx 0.624
$$

---

### 1.7 数据集统计

HDF5 文件结构（`train_rho{rho}.h5`）：
- `X`：形状 $(N_{\text{total}}, 9)$，float32，原始模板值（未正则化）
- `Y`：形状 $(N_{\text{total}}, 1)$，float32，$h\kappa$ 标签
- `reinit_steps`：步数标签 $\in \{0,5,10,15,20\}$

因对称扩增，$\mathbb{E}[X] \approx 0$ 且 $\mathbb{E}[Y] \approx 0$（验证条件 $|\overline{X}| < 10^{-6}$，$|\overline{Y}| < 10^{-6}$）。

---

## 第二部分：测试数据生成

测试数据使用**flower 曲线**（非圆形），评估网络在更复杂几何上的泛化能力。

### 2.1 Flower 曲线定义

极坐标方程：
$$
r(\theta) = b + a\cos(p\theta), \quad \theta \in [0, 2\pi)
$$

**默认参数**（smooth 实验，`a=0.05, b=0.15, p=3`）：

曲线在极坐标系中形成 3 瓣花形，半径在 $[b-a, b+a] = [0.10, 0.20]$ 之间振荡。

**acute 实验**（`a=0.075, b=0.15, p=3`）：振荡更大，曲率更极端。

---

### 2.2 测试域与网格

测试测网格采用 `xy` 索引约定（行对应 $y$，列对应 $x$），域为 $[-L,L]^2$：

$$
x_j = -L + j\cdot h, \quad y_i = -L + i\cdot h, \qquad i,j = 0,\ldots,N-1
$$

$$
h = \frac{2L}{N-1}
$$

**具体参数表**（`config.yaml`）：

| 场景 | $\rho_{\text{model}}$ | $L$ | $N$ | $h$ |
|---|---|---|---|---|
| smooth\_256 | 256 | 0.207843 | 107 | $3.9216\times10^{-3}$ |
| smooth\_266 | 266 | 0.207547 | 111 | $3.7736\times10^{-3}$ |
| smooth\_276 | 276 | 0.207339 | 114 | $3.6697\times10^{-3}$ |
| acute\_276 | 276 | 0.232258 | 129 | $3.6290\times10^{-3}$ |

注意测试和训练的 $h$ 值匹配是刻意设计的：
$$
h_{\text{test}} = \frac{2L}{N-1} \approx \frac{1}{\rho-1} = h_{\text{train}}
$$

---

### 2.3 三种初值模式

#### 模式 A：`formula_phi0`（代数公式初值）

直接代入极坐标代数表达式：

$$
\phi^{(0)}(x,y) = \sqrt{x^2+y^2} - a\cos\!\left(p\arctan\!\frac{y}{x}\right) - b
$$

此场在界面处近似但**不精确**满足 Eikonal 方程，需重初始化。

#### 模式 B：`formula_phi0_projection_band`（带内修正）

在窄带 $|\phi^{(0)}| \leq \sigma h$（默认 $\sigma=2.0$，即 2 个格子宽度）内，用 Newton 投影修正为精确距离。

设 band 内点 $(x,y)$，求最近的曲线点 $\theta^*$ 使得 $\nabla_\theta \|\mathbf{c}(\theta) - (x,y)\|^2 = 0$（下详），得：

$$
\phi^{(0)}_{\text{corrected}}(x,y) = \text{sgn}(\phi^{(0)}_{\text{formula}}(x,y)) \times \left\|\mathbf{c}(\theta^*) - (x,y)\right\|_2
$$

带外保留代数值。

#### 模式 C：`exact_sdf`（精确 SDF 初值）

对全场每点进行 Newton 投影，计算精确 SDF。

---

### 2.4 Newton 投影求最近点参数 $\theta^*$

最近点问题等价于：

$$
f(\theta) = [\mathbf{c}(\theta)-(x,y)] \cdot \mathbf{c}'(\theta) = 0
$$

其中曲线点 $\mathbf{c}(\theta) = (r(\theta)\cos\theta, r(\theta)\sin\theta)$，

曲线参数化导数（$r(\theta) = b+a\cos(p\theta)$，$r'(\theta) = -ap\sin(p\theta)$，$r''(\theta) = -ap^2\cos(p\theta)$）：

$$
c_x(\theta) = r\cos\theta, \quad c_y(\theta) = r\sin\theta
$$
$$
c'_x(\theta) = r'\cos\theta - r\sin\theta, \quad c'_y(\theta) = r'\sin\theta + r\cos\theta
$$
$$
c''_x(\theta) = r''\cos\theta - 2r'\sin\theta - r\cos\theta, \quad c''_y(\theta) = r''\sin\theta + 2r'\cos\theta - r\sin\theta
$$

Newton 迭代（初值 $\theta_0 = \arctan2(y,x) \bmod 2\pi$）：

$$
f(\theta) = (c_x-x)c'_x + (c_y-y)c'_y
$$
$$
f'(\theta) = {c'_x}^2 + {c'_y}^2 + (c_x-x)c''_x + (c_y-y)c''_y
$$
$$
\theta_{k+1} = \left(\theta_k - \frac{f(\theta_k)}{f'(\theta_k)}\right) \bmod 2\pi
$$

终止条件：$|f(\theta)| \leq \tau_{\text{tol}} = 10^{-11}$（向量化版）或精度自适应（mpmath 版，有效位 `dps=80`）。

**高精度版本**（`high_precision_exact_sdf`）：用 mpmath 以 80 位十进制精度（约 266 位二进制）逐点迭代，保证 SDF 参考值正确至机器精度以内。

**有效精度估计**（80 dps，近 $10^{-80}$ 精度定义 $\tau$）：
$$
|f(\theta^*)| \leq 10^{-(80-10)} = 10^{-70}
$$

最终 SDF 值：
$$
\phi^{(0)}(x,y) = \text{sgn}\!\left(\phi^{(0)}_{\text{formula}}(x,y)\right) \times \left\|(c_x(\theta^*), c_y(\theta^*)) - (x,y)\right\|_2
$$

---

### 2.5 测试场重初始化

测试数据使用与训练相同的 WENO5 + TVD-RK3，但参数不同：

| 参数 | 训练 | 测试 |
|---|---|---|
| `cfl` | 0.5 | 0.95 |
| `eps_sign_factor` | 1.0 | 2.5 |
| `sign_mode` | `frozen_phi0` | `dynamic_phi` |
| `time_order` | 3 | 3 |
| `space_order` | 5 | 5 |

测试迭代步数：$n_s \in \{5, 10, 20\}$（`test_iters: [5, 10, 20]`）。

**测试步长**（$h \approx 3.922\times10^{-3}$）：
$$
\Delta\tau_{\text{test}} = 0.95 \times h \approx 3.726\times10^{-3}
$$

测试 CFL 接近 1，演化更激进，会产生更多界面漂移，测试难度更高。

**平滑符号函数**（测试，$\varepsilon_s=2.5$）：
$$
S(\phi) = \frac{\phi}{\sqrt{\phi^2 + (2.5h)^2}}
$$

注意测试用 `dynamic_phi`：每个 TVD-RK 子级用的是**该级的当前 $\phi$** 计算 $S$，而不是冻结 $\phi_0$。

---

### 2.6 测试标签：解析曲率

对重初始化后的场 $\phi$ 的每个界面相邻节点 $(x_n, y_n)$，Newton 投影也求对应的 $\theta^*_n$，然后用 flower 曲率的**解析公式**：

$$
\kappa(\theta) = \frac{r^2 + 2{r'}^2 - r\,r''}{(r^2 + {r'}^2)^{3/2}}
$$

其中：
$$
r = b + a\cos(p\theta), \quad r' = -ap\sin(p\theta), \quad r'' = -ap^2\cos(p\theta)
$$

量纲化目标值：

$$
h\kappa^{\text{analytic}}_n = h \cdot \kappa(\theta^*_n)
$$

**具体数值**（smooth 曲线 `a=0.05, b=0.15, p=3`）：

在平滑处（$\cos(3\theta)=1$ 最大半径）：
$$
r = 0.20, \quad r' = 0, \quad r'' = -0.05\times9 = -0.45
$$
$$
\kappa = \frac{0.04 + 0 - 0.20\times(-0.45)}{0.04^{3/2}} = \frac{0.04+0.09}{0.008} = 16.25
$$

在尖锐处（$\cos(3\theta)=-1$ 最小半径）：
$$
r = 0.10, \quad r' = 0, \quad r'' = 0.45
$$
$$
\kappa = \frac{0.01 + 0 - 0.10\times0.45}{0.001} = \frac{0.01-0.045}{0.001} = -35.0
$$

典型 $h\kappa$ 范围（smooth\_256，$h\approx3.92\times10^{-3}$）：
$$
h\kappa \in [-35\times3.92\times10^{-3},\; 16.25\times3.92\times10^{-3}] \approx [-0.137,\; 0.064]
$$

---

## 第三部分：模型训练

### 3.1 MLP 架构

**输入**：9 维向量（3×3 模板展平），**输出**：标量 $h\kappa$ 预测

网络结构（`CurvatureMLP`）：

$$
f(\mathbf{x};\theta) = W_5 \cdot \text{ReLU}(W_4 \cdot \text{ReLU}(W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1\mathbf{x} + b_1) + b_2) + b_3) + b_4) + b_5
$$

参数维度（`hidden_dim_overrides: {266: 140, 276: 140}`）：

| $\rho$ | $d_{\text{hidden}}$ | 层参数总数 |
|---|---|---|
| 256 | 128 | $10\times128 + 3\times(128^2+128) + (128+1) = 1280 + 49536 + 129 = 50945$ |
| 266 | 140 | $10\times140 + 3\times(140^2+140) + (140+1) = 1400 + 59220 + 141 = 60761$ |
| 276 | 140 | 同 266 = 60761 |

权重初始化（Xavier Uniform）：
$$
W \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{d_{\text{in}}+d_{\text{out}}}},\; \sqrt{\frac{6}{d_{\text{in}}+d_{\text{out}}}}\right], \quad b = 0
$$

---

### 3.2 特征 Z-Score 正则化

在训练集（70% 子集）上计算每个特征维度 $k$ 的均值 $\mu_k$ 和标准差 $\sigma_k$：

$$
\mu_k = \frac{1}{N_{\text{train}}} \sum_{n=1}^{N_{\text{train}}} X_{n,k}
$$
$$
\sigma_k = \sqrt{\frac{1}{N_{\text{train}}} \sum_{n=1}^{N_{\text{train}}} X_{n,k}^2 - \mu_k^2}
$$

若 $\sigma_k = 0$ 则令 $\sigma_k = 1$（零方差特征不归一化）。

网络实际输入：
$$
\tilde{X}_{n,k} = \frac{X_{n,k} - \mu_k}{\sigma_k}
$$

由于正负对称扩增，$\mu_k \approx 0$，故 $\tilde{X}_{n,k} \approx X_{n,k}/\sigma_k$。

$\mu_k$、$\sigma_k$ 存储于 `zscore_stats_{rho}.csv`，推理时使用。

---

### 3.3 数据集分割

总样本随机打乱，固定 seed=42：

$$
N_{\text{train}} = \lfloor 0.70 N \rfloor, \quad N_{\text{val}} = \lfloor 0.15 N \rfloor, \quad N_{\text{test}} = N - N_{\text{train}} - N_{\text{val}}
$$

---

### 3.4 损失函数

优化目标：**均方误差（MSE）**：

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{n=1}^{B} \left(f(\tilde{\mathbf{x}}_n;\theta) - y_n\right)^2
$$

其中 $B=256$ 为 batch size。

注：由于 $y = h\kappa$ 是量纲化量，量级 $\sim 10^{-2}$–$10^{-1}$，MSE 在此尺度定义，与原始曲率 $\kappa$ 的 MSE 相差 $h^2$ 因子。

---

### 3.5 优化器：Adam

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad \beta_1 = 0.9
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \qquad \beta_2 = 0.999
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon} \hat{m}_t
$$

超参数值（`train/config.yaml`）：
$$
\eta = 1.5\times10^{-4}, \quad \beta_1 = 0.9, \quad \beta_2 = 0.999, \quad \varepsilon = 10^{-8}, \quad \lambda_{\text{wd}} = 0
$$

---

### 3.6 Early Stopping

监控验证集 MAE：

$$
\text{MAE}_{\text{val}} = \frac{1}{N_{\text{val}}} \sum_n |f(\tilde{\mathbf{x}}_n) - y_n|
$$

若连续 `patience=30` 个 epoch 无改善，停止训练并恢复最优权重。

最大训练轮数：`max_epochs=500`。

---

### 3.7 训练中的评估指标

**MSE、RMSE、MAE、MaxAE**：

$$
\text{MSE} = \frac{1}{N}\sum_n e_n^2, \quad
\text{RMSE} = \sqrt{\text{MSE}}, \quad
\text{MAE} = \frac{1}{N}\sum_n |e_n|, \quad
\text{MaxAE} = \max_n |e_n|
$$

其中 $e_n = f(\tilde{\mathbf{x}}_n) - y_n$。

---

## 第四部分：测试与评估

### 4.1 数值基准方法（Expanded Formula）

对重初始化后的场 $\phi$，在采样节点处用中央差分计算曲率（`xy` 索引约定下）：

$$
\phi_x = \frac{\phi_{i,j+1} - \phi_{i,j-1}}{2h}, \quad
\phi_y = \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2h}
$$

$$
\phi_{xx} = \frac{\phi_{i,j+1} - 2\phi_{i,j} + \phi_{i,j-1}}{h^2}, \quad
\phi_{yy} = \frac{\phi_{i+1,j} - 2\phi_{i,j} + \phi_{i-1,j}}{h^2}
$$

$$
\phi_{xy} = \frac{\phi_{i+1,j+1} - \phi_{i+1,j-1} - \phi_{i-1,j+1} + \phi_{i-1,j-1}}{4h^2}
$$

曲率公式（mean curvature in 2D = 曲线曲率）：

$$
\kappa = \frac{\phi_x^2 \phi_{yy} - 2\phi_x\phi_y\phi_{xy} + \phi_y^2\phi_{xx}}{(\phi_x^2+\phi_y^2)^{3/2}}
$$

量纲化：

$$
h\kappa^{\text{FD}} = h \cdot \kappa
$$

此方法的精度由 Taylor 截断误差决定（中央差分，2 阶）：
$$
|\kappa^{\text{FD}} - \kappa^{\text{exact}}| = O(h^2) \approx O(10^{-5})\text{–}O(10^{-4})
$$

---

### 4.2 神经网络预测

**推理过程**：

1. 提取采样节点 3×3 模板 $\mathbf{x}$ （`training_order` 编码）
2. 正则化：$\tilde{\mathbf{x}} = (\mathbf{x} - \boldsymbol{\mu})/\boldsymbol{\sigma}$
3. 前向传播：$\hat{y} = f(\tilde{\mathbf{x}};\theta^*)$

其中 $\theta^*$ 为最优 epoch 对应的权重，$\boldsymbol{\mu},\boldsymbol{\sigma}$ 来自训练集统计。

---

### 4.3 评估指标

对 $M$ 个采样节点，计算：

$$
\text{MAE}_{h\kappa} = \frac{1}{M}\sum_{n=1}^M |y_n^{\text{pred}} - y_n^{\text{analytic}}|
$$

$$
\text{MaxAE}_{h\kappa} = \max_{n=1}^M |y_n^{\text{pred}} - y_n^{\text{analytic}}|
$$

$$
\text{MSE}_{h\kappa} = \frac{1}{M}\sum_{n=1}^M (y_n^{\text{pred}} - y_n^{\text{analytic}})^2
$$

---

### 4.4 Eikonal 质量评估

通过检验 $|\nabla\phi| \approx 1$ 评估重初始化质量，在界面相邻节点上：

$$
\text{Eikonal error}_n = \left|\sqrt{\phi_x^2 + \phi_y^2}\bigg|_n - 1\right|
$$

若重初始化收敛，界面附近 $|\nabla\phi| \to 1$，Eikonal error $\to 0$。

---

## 第五部分：完整 Pipeline 流程图

```
一、训练数据生成
────────────────────────────────────────────────────────────────────
  For ρ ∈ {256, 266, 276}:
    h = 1/(ρ−1)
    For r_k (k=0..N_r-1, N_r≈124), 5 variations:
      Blueprint: circle(r_k, c_x, c_y)
        ↓
      [SDF初值]  φ⁰ = √((x−cx)²+(y−cy)²) − r      → n_s=0
      [非SDF初值] φ⁰ = (x−cx)²+(y−cy)² − r²
        ↓ WENO5-空间 + TVD-RK3-时间
      重初始化 n_s ∈ {5,10,15,20} 步
        ↓
      界面相邻节点采样 → M≈800 个节点
        ↓
      提取3×3模板 + training_order 编码
        ↓
      标签 y = h/r （解析曲率）
        ↓
      正负扩增: (x,y) → (+x,+y) 和 (−x,−y)
        ↓
      写入 HDF5: X(N,9), Y(N,1)

二、测试数据生成
────────────────────────────────────────────────────────────────────
  For scenario ∈ {smooth_256, smooth_266, smooth_276, acute_276}:
    花形曲线 r(θ)=b+a·cos(pθ)，a=0.05,b=0.15,p=3
    网格 [-L,L]², N×N，indexing='xy'
      ↓
    初值模式（formula_phi0/exact_sdf等）
      ↓ WENO5-空间 + TVD-RK3-时间 (CFL=0.95, dynamic sign)
    重初始化 n_s ∈ {5,10,20} 步
      ↓
    界面相邻节点采样
      ↓
    对每节点 (x_n,y_n): Newton投影求θ*_n
      ↓
    解析曲率 y_n = h·κ(θ*_n)
    数值曲率 ŷ_n^{FD} = h·κ^{FD}
    原始模板  x_n (3×3)
      ↓
    存储 HDF5: indices, stencils_raw, hkappa_target, hkappa_fd

三、模型训练
────────────────────────────────────────────────────────────────────
  For ρ ∈ {256, 266, 276}:
    加载 train_rho{ρ}.h5
    Z-Score 归一化（在训练集上计算 μ, σ）
    70%/15%/15% 分割 (train/val/test)
      ↓
    MLP: 9 → d → d → d → d → 1 (d=128 or 140)
    Xavier 初始化
      ↓
    For epoch=1..500:
      Adam(lr=1.5e-4) 最小化 MSE
      监控 val MAE, early stopping (patience=30)
      保存最优权重 model_rho{ρ}.pth
      ↓
    存储 zscore_stats_{ρ}.csv

四、评估
────────────────────────────────────────────────────────────────────
  对每个测试场景:
    方法A: FD数值方法  → MAE_{hκ} ~ 10⁻³–10⁻⁴
    方法B: MLP神经网络 → MAE_{hκ} ~ 10⁻⁶
      ↓
    输出 MAE, MaxAE, MSE 对比表
```

---

## 第六部分：关键数学分析

### 6.1 为什么 MLP 能超越 FD？

有限差分曲率误差的主项（Taylor 展开，$\phi$ 足够光滑时）：

$$
\kappa^{\text{FD}} = \kappa + C_1 h^2 + O(h^4)
$$

其中 $C_1$ 依赖三阶偏导，在曲率较大处（曲率变化快）系数大，故**FD 误差正比于 $h^2\kappa'''$**。

MLP 从 3×3 模板直接学习 $h\kappa$ 的映射，绕过了 Taylor 截断。其有效精度受限于：
1. 训练数据（圆形）的解析曲率精度（精确）
2. 测试时 $\phi$ 的 Eikonal 性质（$|\nabla\phi|\approx1$，重初始化质量）
3. 网络容量与泛化能力

当界面几何与训练分布匹配时，MLP 误差可降至 $O(10^{-6})$ 量级。

### 6.2 量纲分析与尺度匹配

本流程中的核心量纲化设计：训练和测试的 $h$ 值精确匹配（同 $\rho$），使得：

$$
h\kappa_{\text{test}} \approx h\kappa_{\text{train}} \in [0, 0.624]
$$

网络输入（$\phi$ 值）与网格间距 $h$ 同量级（SDF 在界面附近 $\phi \sim O(h)$），故正则化后的输入分布在训练和测试间一致。

### 6.3 WENO5 精度阶估计

WENO5 在光滑区域退化为 5 阶精度，在间断处 $\ell^1$ 精度为 1 阶：

$$
\|D^\pm\phi - \phi'\| = O(h^5) \quad \text{光滑处}
$$

对于已是 SDF 的场（$|\nabla\phi|=1$ 精确），重初始化迭代本质上是恒等映射（每步 $\mathcal{L} = -S(|\nabla\phi|-1) = 0$），不引入误差。对非 SDF 初值，经有限步重初始化后 $|\nabla\phi| \to 1$，收敛速度约为 $O(\tau)$。

---

## 附录 A：Eikonal 验证

验证数据集 Eikonal 性质的检验（`validate_curvature_dataset`）：

对 $n_s=0$（SDF）样本，随机抽取节点，用模板中的差分估计梯度模：

$$
|\nabla\phi|_n \approx \sqrt{\left(\frac{X_{n,5}-X_{n,3}}{2h_{\text{eff}}}\right)^2 + \left(\frac{X_{n,1}-X_{n,7}}{2h_{\text{eff}}}\right)^2}
$$

（下标对应 `training_order` 编码：X[5]=右邻，X[3]=左邻相关位置；X[1]=上邻，X[7]=下邻相关位置）

验证条件：$0.90 < \overline{|\nabla\phi|} < 1.10$。

---

## 附录 B：测试域尺寸的设计逻辑

测试域 $L$ 和 $N$ 的选取满足：
$$
h_{\text{test}} = \frac{2L}{N-1} = \frac{1}{\rho-1} = h_{\text{train}}
$$

以 smooth\_256 为例：
$$
h = \frac{2\times0.207843}{107-1} = \frac{0.415686}{106} = 3.9216\times10^{-3} = \frac{1}{255}
$$

验证：$1/255 = 3.9216\times10^{-3}$ ✓

这确保了网络在推理时看到的 $\phi$ 值尺度与训练时完全一致，是 MLP 高精度的关键前提。

---

## 附录 C：数值参数汇总

| 参数 | 训练 | 测试 |
|---|---|---|
| 空间格式 | WENO5 (5阶) | WENO5 (5阶) |
| 时间格式 | TVD-RK3 | TVD-RK3 |
| Hamiltonian | Godunov (Rouy-Tourin) | 同 |
| CFL | 0.5 | 0.95 |
| $\varepsilon_s$ | 1.0 | 2.5 |
| sign_mode | frozen\_phi0 | dynamic\_phi |
| $\varepsilon_{\text{WENO}}$ | $10^{-6}$ | 同 |
| 边界填充 | edge (3层) | 同 |
| 重初始化步 | 0,5,10,15,20 | 5,10,20 |
| MLP 隐藏层 | 4 层 | — |
| 激活函数 | ReLU | — |
| 批大小 | 256 | — |
| 学习率 | $1.5\times10^{-4}$ | — |
| 正则化 | 无 weight decay | — |
| Early stopping | patience=30 | — |
