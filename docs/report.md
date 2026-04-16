# Level-Set 曲率学习数据生成的统一数学表述

## Method

### 1. Problem Formulation

我们考虑如下监督学习问题：给定界面附近的局部 level-set 模板，学习其对应的无量纲曲率
$$
\kappa_h := h \kappa .
$$
因此，数据集被表述为
$$
\mathcal{D}=\{(X_n,y_n)\}_{n=1}^{M},\qquad X_n\in\mathbb{R}^9,\qquad y_n\in\mathbb{R},
$$
其中
$$
y_n=\kappa_h=h\kappa.
$$

该数据生成过程可抽象为如下算子链：
$$
\Gamma \longmapsto \phi_0 \longmapsto \phi=\mathcal{R}(\phi_0)
\longmapsto \mathcal{I}=\mathcal{S}(\phi)
\longmapsto P_{ij}
\longmapsto X_{ij}=\mathcal{E}(P_{ij})
\longmapsto y_{ij}.
$$
这里：

- $\Gamma$ 表示连续界面（interface）；
- $\phi_0$ 表示初始 level-set field；
- $\mathcal{R}$ 表示 reinitialization operator；
- $\mathcal{S}$ 表示 interface-adjacent sampling operator；
- $\mathcal{E}$ 表示 local stencil encoding operator；
- $y_{ij}$ 表示解析定义的目标曲率。

整个方法的核心在于：以统一的 Hamilton--Jacobi reinitialization dynamics 连接几何对象与局部离散样本，再以解析曲率定义监督量，从而避免把标签定义建立在学习器自身的数值近似之上。

### 2. Geometric Representation and Initial Level-Set Fields

#### 2.1 Training Geometry: Circular Interface

训练监督采用圆界面族（circular interface family）
$$
\Gamma^{\mathrm{tr}}_{r,c}
=
\left\{
(x,y)\in\mathbb{R}^2:
(x-c_x)^2+(y-c_y)^2=r^2
\right\},
$$
其中 $c=(c_x,c_y)$ 为中心，$r>0$ 为半径。

在规则网格上，对同一几何对象考虑两类初始 field：

1. **Signed Distance Function (SDF)**
$$
\phi^{\mathrm{tr}}_{\mathrm{SDF}}(x,y)
=
\sqrt{(x-c_x)^2+(y-c_y)^2}-r.
$$

2. **Non-distance Algebraic Field (non-SDF)**
$$
\phi^{\mathrm{tr}}_{\mathrm{alg}}(x,y)
=
(x-c_x)^2+(y-c_y)^2-r^2.
$$

二者具有相同零水平集
$$
\{(x,y):\phi(x,y)=0\}=\Gamma^{\mathrm{tr}}_{r,c},
$$
但仅前者满足
$$
\|\nabla \phi^{\mathrm{tr}}_{\mathrm{SDF}}\|=1
\quad
\text{in a neighborhood of }\Gamma^{\mathrm{tr}}_{r,c}.
$$
因此，训练链路本质上同时考察了"distance-consistent"与"non-distance-consistent"两类初值在统一重初始化算子下的局部统计结构。

#### 2.2 Test Geometry: Polar Flower Interface

测试监督采用极坐标曲线（polar flower interface）
$$
r(\theta)=b+a\cos(p\theta),
\qquad \theta\in[0,2\pi),
$$
并定义参数曲线
$$
C(\theta)
=
\bigl(r(\theta)\cos\theta,\ r(\theta)\sin\theta\bigr).
$$
当 $0<a<b$ 时，该曲线为单连通闭曲线；参数 $p\in\mathbb{N}$ 控制角向振荡频率，$a$ 控制非圆度（non-circularity）。

测试链路的主初值采取径向公式场（formula-based radial field）
$$
\phi^{\mathrm{te}}_0(x,y)
=
\sqrt{x^2+y^2}
-a\cos\!\bigl(p\,\operatorname{atan2}(y,x)\bigr)-b.
$$
它与 $\Gamma^{\mathrm{te}}$ 共享零水平集，但通常并非精确 signed distance function。于是，测试链路中由重初始化诱导的局部模板变化直接决定了数值曲率参考量与解析投影标签之间的关系。

### 3. Reinitialization Dynamics

#### 3.1 Continuous Hamilton--Jacobi Form

我们对任意初始 field $\phi_0$ 考虑如下重初始化方程（reinitialization PDE）
$$
\partial_\tau \phi
+S(\cdot)\bigl(\|\nabla \phi\|-1\bigr)=0,
\qquad
\phi(\cdot,0)=\phi_0.
$$
该方程的稳态目标是逼近满足 Eikonal relation 的场
$$
\|\nabla \phi\|=1,
$$
同时保持零水平集不变，即
$$
\{x:\phi(x,\tau)=0\}\approx \{x:\phi_0(x)=0\}.
$$

#### 3.2 Smoothed Sign Function

为避免 $\operatorname{sign}(\phi)$ 在界面附近的不连续性，采用平滑符号函数（smoothed sign function）
$$
S(\psi)=\frac{\psi}{\sqrt{\psi^2+\varepsilon^2}},
\qquad
\varepsilon=\alpha h,
$$
其中 $h$ 为网格尺度，$\alpha>0$ 为无量纲平滑参数。

本文采用两种自然的符号选择：

- **Frozen sign law**
$$
S_{\mathrm{fr}}=S(\phi_0);
$$
- **Dynamic sign law**
$$
S_{\mathrm{dy}}^{(m)}=S(\phi^{(m)}).
$$

前者强调对初值符号结构的冻结保持，后者强调当前场对局部传播速度的即时调制。两者并不改变零水平集的目标，但会显著影响界面附近模板的离散几何结构。

#### 3.3 Baseline Numerical Regime

本文采用的基准重初始化参数为
$$
\Delta\tau = \mathrm{CFL}\cdot h,
\qquad
\mathrm{CFL}=0.95,
\qquad
\varepsilon=2.5h,
$$
并配合
$$
\text{5th-order WENO}
\qquad\text{and}\qquad
\text{3rd-order TVD-RK}.
$$
这一参数选择的目标不是追求一般性最优，而是获得在界面附近同时具备较低 Eikonal defect 与较好几何保持性的局部模板族。

### 4. Analytic Supervision on the Training Branch

#### 4.1 Exact Curvature of the Circle

对于圆界面 $\Gamma^{\mathrm{tr}}_{r,c}$，其几何曲率为常数
$$
\kappa^{\mathrm{tr}}=\frac{1}{r}.
$$
因此训练标签自然定义为
$$
y^{\mathrm{tr}}=\kappa_h^{\mathrm{tr}}=h\kappa^{\mathrm{tr}}=\frac{h}{r}.
$$

这一监督具有两个关键性质：

1. **analytic exactness**：标签不依赖数值近似；
2. **scale normalization**：通过 $h\kappa$ 消除网格尺度变化带来的维度差异。

#### 4.2 Unified Treatment of Distance and Non-Distance Fields

无论初值为 $\phi^{\mathrm{tr}}_{\mathrm{SDF}}$ 还是 $\phi^{\mathrm{tr}}_{\mathrm{alg}}$，都先经过统一重初始化算子
$$
\phi^{\mathrm{tr}}=\mathcal{R}(\phi_0),
$$
再在界面附近提取模板。于是，同一曲率标签 $h/r$ 可与不同的局部 field geometry 对应，从而把学习问题转写为
$$
\text{local level-set geometry}
\longmapsto
\text{dimensionless curvature}.
$$
这一步避免了把"是否为距离函数"直接硬编码为标签，而是让其作为局部模板分布的一部分进入样本空间。

### 5. Projection-Based Supervision on the Test Branch

#### 5.1 Orthogonal Projection

对于任意采样点 $x=(x_1,x_2)$ 及花形界面参数曲线 $C(\theta)$，定义其正交投影参数 $\theta^*$ 为如下方程的解：
$$
\bigl(C(\theta^*)-x\bigr)\cdot C'(\theta^*)=0.
$$
这意味着 $x$ 到曲线的最短连线与切向量正交，因此 $\theta^*$ 给出最邻近曲线点（nearest point on the interface）。

#### 5.2 Analytic Curvature of the Polar Curve

记
$$
r(\theta)=b+a\cos(p\theta),
\qquad
r'(\theta)=-ap\sin(p\theta),
\qquad
r''(\theta)=-ap^2\cos(p\theta).
$$
则极坐标曲线的曲率为
$$
\kappa^{\mathrm{te}}(\theta)
=
\frac{r(\theta)^2+2r'(\theta)^2-r(\theta)r''(\theta)}
{\bigl(r(\theta)^2+r'(\theta)^2\bigr)^{3/2}}.
$$
因此，测试标签被定义为
$$
y^{\mathrm{te}}
=
\kappa_h^{\mathrm{te}}
=
h\,\kappa^{\mathrm{te}}(\theta^*).
$$

该定义的本质是：标签由连续曲线几何唯一决定，而不是由重初始化后的数值模板反推得到。于是，测试链路真正度量的是"局部离散场对真实界面曲率的可恢复性（recoverability）"。

### 6. Local Sample Extraction and Representation

#### 6.1 Interface-Adjacent Sampling Set

记重初始化后的离散场仍记为 $\phi$。定义界面邻近采样集合（interface-adjacent set）
$$
\mathcal{I}=\mathcal{S}(\phi)
$$
为所有与变号边相邻的内点集合。等价地，若某点 $(i,j)$ 的某条相邻网格边满足
$$
\phi_{i,j}\phi_{i+1,j}\le 0
\qquad\text{or}\qquad
\phi_{i,j}\phi_{i,j+1}\le 0,
$$
则其被纳入 $\mathcal{I}$ 的邻域。

该集合的几何含义是：只在真正与零水平集局部相互作用的区域提取样本，而非在整个域上平均采样。

#### 6.2 Local Patch and Encoding

对于 $(i,j)\in\mathcal{I}$，定义局部模板
$$
P_{ij}
=
\phi[i-1:i+1,\ j-1:j+1]
\in\mathbb{R}^{3\times 3}.
$$
再通过编码算子 $\mathcal{E}$ 形成长度为 9 的特征向量
$$
X_{ij}=\mathcal{E}(P_{ij})\in\mathbb{R}^9.
$$
于是最终样本对为
$$
(X_{ij},y_{ij}).
$$

对训练链路，还施加符号对称增广（sign-symmetry augmentation）
$$
\mathfrak{A}(X_{ij},y_{ij})=(-X_{ij},-y_{ij}),
$$
其依据在于曲率符号与局部 field orientation 的反对称耦合关系。

---

## Appendix

### A. Semi-Discrete Reinitialization Operator

#### A.1 Upwind One-Sided Derivatives and Godunov Selection

设 $D_x^-\phi, D_x^+\phi, D_y^-\phi, D_y^+\phi$ 分别表示沿 $x$ 与 $y$ 方向的 backward/forward one-sided derivatives。定义 Godunov 型梯度范数
$$
G_h(\phi;S)
=
\begin{cases}
\left(
\max\{\max(D_x^-\phi,-D_x^+\phi),0\}^2
+
\max\{\max(D_y^-\phi,-D_y^+\phi),0\}^2
\right)^{1/2},
& S\ge 0,
\\
\left(
\max\{\max(-D_x^-\phi,D_x^+\phi),0\}^2
+
\max\{\max(-D_y^-\phi,D_y^+\phi),0\}^2
\right)^{1/2},
& S<0.
\end{cases}
$$
于是半离散重初始化方程写为
$$
\frac{d}{d\tau}\phi
=
\mathcal{H}_h(\phi)
:=
-S(\cdot)\bigl(G_h(\phi;S)-1\bigr).
$$
其中，Godunov selection 的作用是根据传播方向选择与 Hamiltonian 一致的上风信息，从而在界面附近保留正确的黏性解结构（viscosity-solution consistency）。

#### A.2 High-Order Spatial Reconstruction

上述 one-sided derivatives 并非直接由低阶差分给出，而是通过高阶空间重构获得。记
$$
(D_x^-\phi,\ D_x^+\phi,\ D_y^-\phi,\ D_y^+\phi)
=
\mathcal{W}_h(\phi),
$$
其中 $\mathcal{W}_h$ 表示基于五阶 WENO 的 one-sided reconstruction operator。该算子在平滑区间提供高阶精度，在梯度急剧变化区间自动降低局部插值权重，以抑制非物理解振荡。

#### A.3 Time Marching

设 $\phi^{(m)}$ 为第 $m$ 步近似，时间步长为 $\Delta\tau$。则可将一步推进表示为
$$
\phi^{(m+1)}
=
\mathcal{T}_{\Delta\tau}\!\left(\mathcal{H}_h,\phi^{(m)}\right),
$$
其中 $\mathcal{T}_{\Delta\tau}$ 表示三阶 TVD Runge--Kutta 演化算子。该写法强调：高阶时间推进并不是独立的物理模型，而是半离散 Hamilton--Jacobi operator $\mathcal{H}_h$ 的稳定时间积分机制。

### B. Eikonal Defect and Interface Preservation

重初始化的目标并非改变界面几何，而是在固定零水平集附近构造更接近 signed distance 的场。因此可引入 Eikonal defect
$$
e_{\mathrm{eik}}(x)
=
\bigl|\|\nabla\phi(x)\|-1\bigr|.
$$
若 $e_{\mathrm{eik}}$ 在界面邻近带内较小，则 $\phi$ 更适合被解释为几何距离场。与此同时，还需保持
$$
\{x:\phi(x)=0\}\approx\Gamma,
$$
否则虽然 $\|\nabla\phi\|$ 接近 1，但几何对象已发生漂移。因而，重初始化质量应同时从 **distance regularity** 与 **interface fidelity** 两个方面衡量。

### C. Orthogonal Projection on the Polar Interface

记
$$
C(\theta)=\bigl(r(\theta)\cos\theta,\ r(\theta)\sin\theta\bigr).
$$
则其导数为
$$
C'(\theta)
=
\bigl(r'(\theta)\cos\theta-r(\theta)\sin\theta,\ 
r'(\theta)\sin\theta+r(\theta)\cos\theta\bigr).
$$
最近点参数 $\theta^*$ 满足
$$
F(\theta;x):=
\bigl(C(\theta)-x\bigr)\cdot C'(\theta)=0.
$$
这一条件来自距离平方泛函
$$
d(\theta;x)^2=\|C(\theta)-x\|^2
$$
的一阶驻值条件
$$
\frac{d}{d\theta}d(\theta;x)^2
=
2\bigl(C(\theta)-x\bigr)\cdot C'(\theta)=0.
$$
因此，$\theta^*$ 的求解本质上是一个一维非线性正交投影问题。

### D. Curvature Formula for the Polar Curve

对极坐标曲线 $r=r(\theta)$，其曲率满足
$$
\kappa(\theta)
=
\frac{r^2+2(r')^2-r r''}{\bigl(r^2+(r')^2\bigr)^{3/2}}.
$$
代入
$$
r(\theta)=b+a\cos(p\theta),\qquad
r'(\theta)=-ap\sin(p\theta),\qquad
r''(\theta)=-ap^2\cos(p\theta),
$$
即可得到测试链路的解析标签
$$
y^{\mathrm{te}}=h\kappa(\theta^*).
$$
这一表达清楚表明：测试标签完全由连续曲线几何决定，而与离散模板仅通过投影点 $\theta^*$ 发生联系。

### E. Finite-Difference Reference Curvature

为构造纯数值参考量，引入局部有限差分曲率
$$
\kappa_{\mathrm{FD}}
=
\frac{\phi_x^2\phi_{yy}-2\phi_x\phi_y\phi_{xy}+\phi_y^2\phi_{xx}}
{(\phi_x^2+\phi_y^2)^{3/2}},
$$
并相应定义无量纲形式
$$
\kappa_{h,\mathrm{FD}}:=h\kappa_{\mathrm{FD}}.
$$
其中
$$
\phi_x,\ \phi_y,\ \phi_{xx},\ \phi_{yy},\ \phi_{xy}
$$
均由局部中心差分给出。

需要强调的是，
$$
\kappa_{h,\mathrm{FD}}
$$
只是一个 `reference quantity`：它刻画的是给定离散场在局部二阶差分意义下诱导出的曲率，而不是监督学习问题中的主标签。主标签始终是解析几何定义下的
$$
\kappa_h=h\kappa.
$$

### F. Unified View of the Generated Dataset

综合训练与测试两条链路，可将整个数据生成过程统一表示为
$$
\mathcal{D}
=
\mathcal{D}_{\mathrm{tr}}
\cup
\mathcal{D}_{\mathrm{te}},
$$
其中
$$
\mathcal{D}_{\mathrm{tr}}
=
\left\{
\bigl(\mathcal{E}(P_{ij}), h/r\bigr)
:\ 
(i,j)\in\mathcal{S}(\mathcal{R}(\phi_0)),
\ 
\phi_0\in\{\phi^{\mathrm{tr}}_{\mathrm{SDF}},\phi^{\mathrm{tr}}_{\mathrm{alg}}\}
\right\},
$$
而
$$
\mathcal{D}_{\mathrm{te}}
=
\left\{
\bigl(\mathcal{E}(P_{ij}), h\kappa(\theta^*)\bigr)
:\ 
(i,j)\in\mathcal{S}(\mathcal{R}(\phi^{\mathrm{te}}_0))
\right\}.
$$
因此，训练集强调 **constant-curvature analytic supervision**，测试集强调 **projection-based variable-curvature supervision**；二者共享同一局部表示、同一重初始化动力学以及同一无量纲目标尺度 $\kappa_h$。这正是统一几何--算子--目标框架的数学本质。
