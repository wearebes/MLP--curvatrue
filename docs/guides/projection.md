# SDF Projection Methods for Parametric Curves

本篇文档记录了我们在 PDE 模型测试管线中，从初始曲线方程生成全局符号距离场 (Signed Distance Field, SDF) 的数学原理演进。主要对比了“原始代数法”与“牛顿正交投影法”。

## 1. 原始方法：代数射线法 (Algebraic Radial Distance)

### 1.1 数学定义
测试域中定义的目标流形是一条基于极坐标的花瓣曲线（Flower Curve）：
$$ r_{curve}(\theta) = b + a \cos(p\theta) $$

在最初的测试数据生成实现中，管线直接使用了论文中的基础隐式公式作为全场初始 SDF $\phi_0$：
$$ \phi_0(x, y) = r - (b + a \cos(p\theta)) $$
其中，当前点在极坐标下的表示为 $r = \sqrt{x^2+y^2}, \;\theta = \arctan(y, x)$。

### 1.2 几何本质与缺陷
几何上，该方法衡量的是网格点 $\mathbf{q} = (x, y)$ **沿着从原点发出的射线（Radial Ray）** 到达边界的距离差。

- **非正交性**：全空间真正的欧氏最短距离必须是网格点到曲面的“正交投影距离”，而不是沿射线的切点插值。除了 $a=0$ 退化为正圆形外，$\phi_0(x,y)$ 的计算结果**并非欧氏最短距离**。
- **破坏 Eikonal 方程**：由于它不是真实的欧氏距离，导致产生的场 $\phi_0$ 在绝大部分区域不满足高维 Eikonal 方程，即梯度范数不为 1（$|\nabla \phi_0| \neq 1$）。
- 由于初始状态已经在欧式距离意义上严重偏离了神经常微分/偏微分方程的训练流形 (Signed-Distance Manifold)，即使用 Godunov WENO5 格式对该初始场进行 20 步 reinitialization，也无法将其完全吸收、拉回正确的特征曲面，从而使得最终用于测试评估的 $3\times3$ stencils 出现了系统性偏差。


## 2. 当前方法：牛顿正交投影法 (Newton-Raphson Orthogonal Projection)

为了得到绝对严格的、满足 $|\nabla \phi_{exact}| = 1$ 的精准 SDF (Exact SDF)，当前的 `vectorized_exact_sdf` 采取了直接寻找欧氏空间极小距离的正交投影方案。

### 2.1 目标函数
极坐标花瓣曲线可以参数化给出一个二维向量函数 $\mathbf{c}(\theta)$：
$$
\mathbf{c}(\theta) = 
\begin{bmatrix}
c_x(\theta) \\
c_y(\theta)
\end{bmatrix}
=
\begin{bmatrix}
(b + a \cos(p\theta)) \cos\theta \\
(b + a \cos(p\theta)) \sin\theta
\end{bmatrix}
$$

对于网格上的任意空间查询点 $\mathbf{q} = (x, y)$，其到曲线的精确无符号最短距离 (Unsigned Shortest Distance) 定义为：
$$ d(\mathbf{q}) = \min_{\theta \in [0, 2\pi)} \|\mathbf{c}(\theta) - \mathbf{q}\| $$

为了避免欧式范数根号求导时的解析奇异性，我们将其转化为求距离平方的局部极小值点（即寻找正交投影点）：
$$ E(\theta) = \frac{1}{2} \|\mathbf{c}(\theta) - \mathbf{q}\|^2 = \frac{1}{2} \left[ (c_x(\theta)-x)^2 + (c_y(\theta)-y)^2 \right] $$

### 2.2 牛顿 - 拉夫逊迭代 (Newton-Raphson Iteration)
最小化连续可导函数 $E(\theta)$ 的必要条件是一阶导数（梯度）为零。这在几何上等价于要求连接段向量与曲线该点切向量互相正交：
$$ f(\theta) = E'(\theta) = (c_x(\theta)-x)c'_x(\theta) + (c_y(\theta)-y)c'_y(\theta) = 0 $$

为了通过数值算法快速求解该非线性根方程 $f(\theta) = 0$，我们引入了二阶收敛的牛顿迭代法。我们需要计算其二阶导数（即 Hessian 矩阵在标量意义下的一维退化形式）：
$$ f'(\theta) = E''(\theta) = (c'_x(\theta))^2 + (c'_y(\theta))^2 + (c_x(\theta)-x)c''_x(\theta) + (c_y(\theta)-y)c''_y(\theta) $$

对于空间坐标 $(x,y)$，以极角 $\theta_0 = \arctan(y, x)$ 启动迭代，更新步长公式为：
$$ \theta_{k+1} = \theta_k - \frac{f(\theta_k)}{f'(\theta_k)} $$

### 2.3 边界判定与封闭 SDF 融合
在经过若干步收敛到最近的正交点对应参数 $\theta_{proj}$ 后，我们计算出查询点 $\mathbf{q}$ 距离该曲线的极小欧式截距 $d = \|\mathbf{c}(\theta_{proj}) - \mathbf{q}\|$。

无符号距离计算完成后，仍需为其附加流形向内（或向外）的拓扑符号信息。此时，由于该花瓣曲线拓扑上属于标准的**星形域（Star-shaped Domain）**（即从原点引出的任何射线仅与边界穿透相交一次），我们可以安全且严密地利用旧有原始代数方程作为该点处拓扑内外的“判定器 (Parity Oracle)”。

确切符号 $\text{sign}$ 由代数公式判断：
$$ \text{sign} = \text{sgn} \Big( \sqrt{x^2+y^2} - \big( b + a \cos(p\cdot \arctan(y,x)) \big) \Big) $$

最终，全空间 Exact SDF 标量场由二者组合生成：
$$ \phi_{exact}(x, y) = \text{sign} \cdot d $$

### 2.4 未来精进方向（Algorithm Safeguards）
尽管目前实现中 `theta_0 = \arctan(y, x)` 的朴素法在绝大数平滑形态中完美奏效（这也是本次 Benchmark 能对齐论文精度的原因），但从数值分析来看，面对极端长花瓣等高曲率形状，当前的实现被归类为“朴素牛顿投影（Naïve Newton Projection）”，理论上存在如下隐患，可作后续算子升级：

1. **局部极小值捕获 (Local Minima Basin)**：从 $\arctan(y, x)$ 径向启动有时会错误地收敛到次近的花瓣边缘（局部极小值）。改进方案为采用 **全局网格粗筛 (Global Grid Search)**：在 $[0, 2\pi)$ 密铺 $M$ 个点采样并计算欧式距离，选取距离下界最小对应的 $\theta_{search}$ 作为真正的牛顿启动收敛核。
2. **Hessian 凹陷折越 (Hessian Defect)**：当空间查询点陷入包围的曲率中心环内时，目标距离函数变为非凸结构，其二阶微分 $f'(\theta_k)$ 容易变为负数。此时牛顿法会错误地向“极大值（Gradient Ascent）”逃逸（把顶点越推越远）。可在迭代步中部署防御策略：检测到 $f'(\theta) < \epsilon$ 时触发阻尼阻拦（如 Levenberg-Marquardt 阻尼或切换为线搜索一阶梯度下降退坡策略）稳固收敛轨道。
