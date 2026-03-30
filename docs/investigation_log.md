# 排查总记录

## 当前目标
复现论文 irregular flower regular-grid pipeline，并定位当前实现与论文结果不一致的主因。

## 当前基线
- 数据生成：`formula_phi0 -> reinitialize -> sample -> 3x3 stencil`
- target：projection-based analytic curvature
- numerical：equation (3) / `expanded_formula`
- neural：`origin` 或指定 `run-id` 模型
- 当前参考数据集：`paper_formula_cfl05`

## 排查项总表

| ID | 假设 | 当前可能性 | 状态 | 最近证据 | 下一步实验 | 关联结果 |
|---|---|---:|---|---|---|---|
| H01 | `reinitializer` 把 `phi` 推到错误但稳定的离散流形 | 85% | supported | 相关性实验支持 local exact-gap；动态 `S(phi)` 的正式数据链复现后，numerical / origin 的 12/12 case-step 全部改善，平均 MSE 仍下降约 `69%` | 继续和论文表格逐项对照，并决定是否把 `dynamic_phi` 升成新的 irregular-test 默认实现 | E01, E03, E04, E05, E06, E07 |
| H02 | 论文原始 irregular-test 实现与当前重写版不一致 | 30% | open | 在 `dynamic_phi` 已经正式落地后，平均 paper-gap 仍有 numerical `7.6x`、origin `28x`；这更像 provenance 级差异，而不是单个小公式错误 | 继续核对 paper irregular-test 的实际生成细节，尤其 reinit 和 sample 提取是否还有未披露实现差异 | - |
| H03 | numerical formula / 坐标约定 / stencil 编码存在残余偏差 | 5% | weakened | `legacy_flat` 对 origin 平均 paper-ratio 只从 `28.17` 变到 `28.07`；`field_div_normal` 仅在 `smooth_276` 局部有利，整体远差于 `eq3` | 保持观察即可，不再作为主线 | E08, E09 |
| H04 | 分辨率缩放或 `rho_eq` 匹配问题，尤其 `276` 档 | 10% | open | 历史分析显示 `276` 更敏感，尤其 `smooth_276` 存在 input drift | 检查 case `h` 与 `trainStats_276` 的 normalized drift | - |
| H05 | frozen-interface effect 主导误差闭环 | 3% | weakened | 样本节点上的 `|S0|` 并不接近 0，reinit 后 patch 明显变化且更接近 exact-SDF | 如有必要，再做相关性分析作为补充证伪 | E02 |

## 实验记录

## 2026-03-27 E01 论文主链基线

- 假设：H01
- 目的：建立 `formula_phi0 + cfl=0.5` 的 paper-style regular-grid 基线
- 方法：使用 `formula_phi0` 生成 irregular flower 测试集；reinit 迭代固定为 `5/10/20`；评估 `numerical + origin`
- 执行命令：
  - `python -m generate test --dataset-name paper_formula_cfl05 --mode formula_phi0`
  - `python -m test numerical origin --data-split test --source paper_formula_cfl05`
- 输入数据集 / 配置：
  - dataset: `paper_formula_cfl05`
  - phi0 mode: `formula_phi0`
  - cfl: `0.5`
  - reinit steps: `5, 10, 20`
- 原始结果：[`20260327_H01_formula_phi0_cfl05_origin_vs_numerical.txt`](/e:/Research/PDE/code1/test/results/20260327_H01_formula_phi0_cfl05_origin_vs_numerical.txt)
- 关键结果：
  - `smooth_256` sample count = `528`
  - `smooth_266` sample count = `552`
  - `smooth_276` sample count = `564`
  - `acute_276` sample count = `672`
  - `origin smooth_256 step20 MSE = 5.097931e-05`
  - `origin acute_276 step20 MSE = 5.262630e-04`
  - `numerical smooth_256 step20 MSE = 6.361701e-05`
  - `numerical acute_276 step20 MSE = 6.410078e-04`
- 结论：论文主链已经复现到正确的样本规模和合理误差档位，当前主问题不是 pipeline 结构错误，而是 `phi0 -> reinit -> stencil` 质量仍不足。
- 对排查表的更新：H01 保持 `testing`

## 2026-03-27 E02 frozen-interface 假设检查

- 假设：H05
- 目的：验证 `S(phi0)` 是否让界面模板几乎冻结不动
- 方法：在 `smooth_256` 和 `acute_276` 上统计样本节点与 patch 节点的 `|S0|`、`|phi(step)-phi0|`、`|phi(step)-exact|`，步数取 `5/10/20`
- 执行命令：
  - 以 `formula_phi0` 生成 `phi0`
  - 用当前 PDE reinitializer 在 `cfl=0.5` 下推进 `5/10/20` 步
  - 对样本 patch 统计 `S0` 与 `phi` 变化量
- 输入数据集 / 配置：
  - cases: `smooth_256`, `acute_276`
  - phi0 mode: `formula_phi0`
  - cfl: `0.5`
  - reinit steps: `5, 10, 20`
- 原始结果：[`20260327_H05_frozen_interface_check.txt`](/e:/Research/PDE/code1/test/results/20260327_H05_frozen_interface_check.txt)
- 关键结果：
  - `smooth_256` 样本节点 `|S0|` 均值约 `0.445`
  - `acute_276` 样本节点 `|S0|` 均值约 `0.506`
  - `smooth_256 step20` patch 上 `|phi-step0|` 均值约 `1.03e-03`
  - `smooth_256 step20` patch 上 `|phi-exact|` 均值约 `1.86e-05`
  - `acute_276 step20` patch 上 `|phi-step0|` 均值约 `2.09e-03`
  - `acute_276 step20` patch 上 `|phi-exact|` 均值约 `5.65e-05`
- 结论：样本节点上的 `|S0|` 并不接近 0，模板也不是几乎不动；reinit 反而在 patch 上显著改变了 `phi`，并把它推近 exact-SDF。frozen-interface 假设不能作为主因。
- 对排查表的更新：H05 改为 `weakened`

## 2026-03-27 E03 H01 相关性验证

- 假设：H01
- 目的：判断曲率误差更像是由 `phi` 偏离 exact-SDF 引起，还是仅仅由“离开 `phi0` 的幅度”引起
- 方法：在 `paper_formula_cfl05` 的 `smooth_256 / smooth_266 / smooth_276 / acute_276` 上，逐个 `5/10/20` 步统计样本节点的
  - `abs_err_hk = |hkappa_fd - hkappa_target|`
  - `center_move = |phi(step) - phi0|`
  - `center_exact_gap = |phi(step) - exact_sdf|`
  - `patch_move_mean = mean_3x3 |phi(step) - phi0|`
  - `patch_exact_gap_mean = mean_3x3 |phi(step) - exact_sdf|`
  然后计算与 `abs_err_hk` 的 Pearson 相关系数。
- 执行命令：
  - `python - < inline correlation script >`
- 输入数据集 / 配置：
  - dataset: `paper_formula_cfl05`
  - phi0 mode: `formula_phi0`
  - cfl: `0.5`
  - reinit steps: `5, 10, 20`
- 原始结果：[`20260327_H01_patch_drift_exact_gap_correlation.txt`](/e:/Research/PDE/code1/test/results/20260327_H01_patch_drift_exact_gap_correlation.txt)
- 关键结果：
  - 汇总后 `corr(mean_err, mean_center_exact_gap) = 0.975224`
  - 汇总后 `corr(mean_err, mean_center_move) = 0.920199`
  - `smooth_266 step10` 上 `corr(err, center_exact) = 0.554270`，高于 `corr(err, center_move) = 0.376422`
  - `smooth_276 step20` 上 `corr(err, center_exact) = 0.452369`，高于 `corr(err, center_move) = 0.382912`
  - `acute_276 step20` 上 `corr(err, center_exact) = 0.399840`，高于 `corr(err, center_move) = 0.349360`
- 结论：曲率误差与 local exact-gap 的耦合比与“偏离初值的幅度”更强，说明当前主问题更接近“reinit 后停在了一个离 exact-SDF 仍有系统偏差的局部流形”，而不是简单的模板被冻结或模板动得太多。
- 对排查表的更新：H01 概率上调到 `60%`，继续保持 `testing`

## 2026-03-27 E04 动态符号项消融

- 假设：H01
- 目的：直接验证 `frozen S(phi0)` 是否是当前 irregular flower gap 的主因之一
- 方法：在保持 `formula_phi0 + cfl=0.5 + space_order=5 + time_order=3` 不变的前提下，把 reinitializer 中的符号项从固定的 `S(phi0)` 改为每个 RK stage 都重新计算 `S(phi)`，并分别评估 numerical 和 `origin`
- 执行命令：
  - `python - < inline dynamic-sign ablation script >`
- 输入数据集 / 配置：
  - dataset: `paper_formula_cfl05`
  - phi0 mode: `formula_phi0`
  - cfl: `0.5`
  - reinit steps: `5, 10, 20`
  - compared methods: `numerical`, `origin`
- 原始结果：[`20260327_H01_dynamic_sign_ablation.txt`](/e:/Research/PDE/code1/test/results/20260327_H01_dynamic_sign_ablation.txt)
- 关键结果：
  - numerical 平均 MSE 从 `3.954637237e-04` 降到 `1.233044114e-04`，相对下降 `68.82%`
  - origin 平均 MSE 从 `2.978602545e-04` 降到 `9.099263520e-05`，相对下降 `69.45%`
  - numerical 的 12/12 个 case-step 全部改善，单行 MSE 降幅范围 `52.55% ~ 76.27%`
  - origin 的 12/12 个 case-step 全部改善，单行 MSE 降幅范围 `56.12% ~ 75.21%`
- 结论：当前 `frozen S(phi0)` 不是一个次要细节，而是现有 gap 的高影响源。虽然论文正文写的是 `S(phi0)`，但就当前实现而言，固定符号项会把 irregular flower 的局部模板推向明显更差的离散流形。
- 对排查表的更新：H01 状态提升为 `supported`，概率上调到 `75%`

## 2026-03-27 E05 动态符号项的局部误差形状

- 假设：H01
- 目的：确认动态 `S(phi)` 主要修复 `3x3` 模板的哪些位置
- 方法：对 `paper_formula_cfl05` 的 4 个 case、`5/10/20` 步，分别统计样本 patch 相对 exact-SDF 的 `center / cross / diag` 平均绝对误差，并比较 frozen vs dynamic 两种 reinitializer
- 执行命令：
  - `python - < inline patch-group script >`
- 输入数据集 / 配置：
  - dataset: `paper_formula_cfl05`
  - phi0 mode: `formula_phi0`
  - cfl: `0.5`
  - reinit steps: `5, 10, 20`
- 原始结果：[`20260327_H01_dynamic_sign_patch_groups.txt`](/e:/Research/PDE/code1/test/results/20260327_H01_dynamic_sign_patch_groups.txt)
- 关键结果：
  - aggregate `center` exact-gap 从 `3.348110273e-05` 降到 `2.436089784e-05`
  - aggregate `cross` exact-gap 从 `4.014133588e-05` 降到 `3.609525214e-05`
  - aggregate `diag` exact-gap 几乎不变，只下降 `2.47e-07`
  - `step=5` 的 smooth cases 是例外：exact-gap 略升，但 curvature error 仍显著下降
- 结论：动态 `S(phi)` 主要在 patch 中心和十字邻点上修复当前流形偏差，这和之前 “center/cross 主导 gap” 的诊断一致。对角项不是主问题。
- 对排查表的更新：H01 的局部机制证据增强

## 2026-03-27 E06 动态符号项的空间阶数扫描

- 假设：H01
- 目的：排除“其实只要改用低阶 Godunov 就行”的解释
- 方法：固定动态 `S(phi)`，扫描 `space_order=3/4/5`，比较加权的 numerical / origin 误差
- 执行命令：
  - `python - < inline dynamic-sign space-order sweep >`
- 输入数据集 / 配置：
  - dataset: `paper_formula_cfl05`
  - compared space_order: `3, 4, 5`
  - cfl: `0.5`
  - reinit steps: `5, 10, 20`
- 原始结果：[`20260327_H01_dynamic_sign_space_order_sweep.txt`](/e:/Research/PDE/code1/test/results/20260327_H01_dynamic_sign_space_order_sweep.txt)
- 关键结果：
  - `space_order=3` 数值直接爆炸，weighted numerical MSE = `1.125900620e+28`
  - `space_order=4` 同样爆炸，weighted numerical MSE = `1.244755430e+30`
  - 只有 `space_order=5` 在动态 `S(phi)` 下保持稳定，weighted numerical MSE = `1.362494506e-04`
- 结论：改进点不是“换成低阶 Godunov”，而是“在当前高阶显式框架里，固定符号项本身就有问题”。
- 对排查表的更新：进一步收缩 H01 的解释空间

## 2026-03-27 E07 动态符号项的正式代码链复现

- 假设：H01
- 目的：验证 `dynamic_phi` 改动不是 inline 原型偶然结果，而是能通过正式数据生成与正式 `test` 评估链稳定复现
- 方法：把 `sign_mode` 接进 `LevelSetReinitializer`、`generate/config.py`、`generate/test_data.py`、`generate/__main__.py`，新增 [`config.paper_formula_dynamic.yaml`](/e:/Research/PDE/code1/generate/config.paper_formula_dynamic.yaml)，然后正式生成 `paper_formula_cfl05_dynsign` 数据集并用 `test` 读取 HDF5 评估 `numerical + origin`
- 执行命令：
  - `python -m generate --config generate/config.paper_formula_dynamic.yaml test --dataset-name paper_formula_cfl05_dynsign`
  - `python -m test numerical origin --data-split test --source paper_formula_cfl05_dynsign`
- 输入数据集 / 配置：
  - dataset: `paper_formula_cfl05_dynsign`
  - phi0 mode: `formula_phi0`
  - cfl: `0.5`
  - sign_mode: `dynamic_phi`
  - time_order: `3`
  - space_order: `5`
- 原始结果：[`20260327_H01_dynamic_sign_formal_eval.txt`](/e:/Research/PDE/code1/test/results/20260327_H01_dynamic_sign_formal_eval.txt)
- 关键结果：
  - `numerical smooth_256 step20 MSE = 2.785738e-05`，明显优于基线 `6.361701e-05`
  - `origin smooth_256 step20 MSE = 1.959254e-05`，明显优于基线 `5.097931e-05`
  - `numerical acute_276 step20 MSE = 2.590851e-04`，明显优于基线 `6.410078e-04`
  - `origin acute_276 step20 MSE = 1.891777e-04`，明显优于基线 `5.262630e-04`
  - 所有 24 行（12 个 numerical + 12 个 origin）全部改善
- 结论：`dynamic_phi` 不是分析期原型偶然收益，而是通过正式数据链也能稳定复现的系统性改进。到这一步，H01 已经从“主嫌疑”上升为“当前最可信的主结果”。
- 对排查表的更新：H01 概率上调到 `85%`

## 2026-03-27 E08 origin 输入编码残余检查

- 假设：H03
- 目的：验证 `origin` 的剩余 paper-gap 是否主要来自 `training_order` vs `legacy_flat` 的编码错位
- 方法：在 `paper_formula_cfl05_dynsign` 上重建同一批 patch，分别用 `training_order` 和 `legacy_flat` 喂给 `origin`，并与论文 model MSE 直接做比值
- 执行命令：
  - `python - < inline origin encoding check >`
- 原始结果：[`20260327_H03_origin_encoding_dynsign.txt`](/e:/Research/PDE/code1/test/results/20260327_H03_origin_encoding_dynsign.txt)
- 关键结果：
  - 平均 `paper_ratio`：`training_order = 28.17`，`legacy_flat = 28.07`
  - 虽然 `legacy_flat` 在 9/12 行略胜，但收益只在边角位，远不足以解释主 gap
- 结论：编码顺序是残余问题，不是主问题。
- 对排查表的更新：H03 下调为 `weakened`

## 2026-03-27 E09 numerical 公式残余检查

- 假设：H03
- 目的：验证 `eq3`、`div(normal)` 或坐标翻转是否是 dynamic-sign 之后的主剩余误差源
- 方法：在 `paper_formula_cfl05_dynsign` 上，固定同一 `phi` 和 target，比较 `field_eq3`、`field_div_normal`、`axis_flip_eq3`
- 执行命令：
  - `python - < inline numerical formula check >`
- 原始结果：[`20260327_H03_numerical_formula_dynsign.txt`](/e:/Research/PDE/code1/test/results/20260327_H03_numerical_formula_dynsign.txt)
- 关键结果：
  - 平均 `paper_ratio`：`field_eq3 = 7.61`，`field_div_normal = 51.20`，`axis_flip_eq3 = 7.61`
  - `field_div_normal` 只在 `smooth_276` 显著更好，但在其余 case 明显更差
  - `axis_flip_eq3` 和 `field_eq3` 基本相同，没有暴露出主轴翻转错误
- 结论：数值公式与坐标约定仍有局部敏感性，但不是当前 dynamic-sign 管线下的主剩余来源。
- 对排查表的更新：H03 继续维持 `weakened`

## 记录规范

- 以后每次实验都必须先生成一个独立原始结果文件，落在 `test/results/`
- 文件命名规则：`YYYYMMDD_Hxx_slug.txt`
- 每条实验记录必须绑定一个假设 ID
- 主记录只写摘要与结论，不重复塞整份终端输出
- 排查总表中的状态只使用：`open`、`testing`、`weakened`、`supported`、`closed`

## 2026-03-27 E10 reinit hyperparameter sweep after dynamic-sign

- 假设：H01
- 目的：确认 `dynamic_phi` 之后的剩余 gap 是不是还主要来自 reinitializer 的隐藏数值细节，而不是 pipeline 结构本身
- 方法：固定 `formula_phi0 -> reinitialize -> current_interface_nodes -> projection target -> numerical / origin` 主链，只扫 `cfl`、`eps_sign_factor`、`time_order`
- 原始结果：[`20260327_H06_reinit_hyperparameter_sweep.txt`](/e:/Research/PDE/code1/test/results/20260327_H06_reinit_hyperparameter_sweep.txt)
- 关键结果：
  - `frozen_eps1_cfl05_t3`: numerical ratio `20.134`，origin ratio `76.159`
  - `frozen_eps2_cfl05_t3`: numerical ratio `1.527`，origin ratio `3.957`
  - `dynamic_eps1_cfl05_t3`: numerical ratio `7.613`，origin ratio `28.174`
  - `dynamic_eps2_cfl05_t3`: numerical ratio `0.952`，origin ratio `2.093`
  - `dynamic_phi + cfl=0.95 + eps=2.50 + RK3`: numerical ratio `0.832`，origin ratio `1.021`
- 结论：之前的 paper-gap 不是单一 provenance 问题，而是 reinitializer 至少还有两个高影响细节没对齐：
  1. `sign_mode` 不能停留在 `frozen_phi0`
  2. `eps_sign_factor` 之前明显过小
  在当前实现里，`cfl` 也需要比之前试过的 `0.5~0.6` 更高，接近 `0.9~1.0` 才能逼近论文结果。

## 2026-03-27 E11 near-paper config check

- 假设：H01，H04
- 目的：验证最优 reinit 组合是否在逐 case-step 上也接近论文，而不是只在平均值上看起来好
- 方法：用 `dynamic_phi + cfl=0.95 + eps_sign_factor=2.5 + RK3 + WENO5` 直接重建 4 个 regular-grid irregular cases，逐行对比 paper MSE；额外检查 `model_h / case_h` 缩放对 origin 的残余影响
- 原始结果：[`20260327_H06_best_near_paper_config.txt`](/e:/Research/PDE/code1/test/results/20260327_H06_best_near_paper_config.txt)
- 关键结果：
  - 12 行 sample counts 全部与 paper 完全一致
  - average numerical ratio = `0.832`
  - average origin raw ratio = `1.021`
- average origin_to_model_h ratio = `0.991`
- 同一高 `cfl/high-eps` 设定下，`frozen_phi0` 仍然只有 numerical `0.952`、origin `1.357`，明显差于 `dynamic_phi`
- 结论：现在已经可以说，当前 rebuilt pipeline 在不改 sampling rule 和 target rule 的前提下，靠 reinitializer 细节修正就能达到“对齐论文或更好”的水平。剩余的 `origin` 微小偏差主要落在 `276` 档的局部尺度匹配上，H04 仍有少量残余，但已经不是主线。

## 2026-03-27 E12 small-CFL method-preserving sweep

- 假设：H01
- 目的：响应“尽量不大改原始参数、优先看方法”的约束，专门检查 `cfl=0.5~0.6` 区间内，能否靠更标准的符号函数写法把结果拉近 paper
- 方法：比较两类 sign smoothing：
  - 当前仓库写法：`phi / sqrt(phi^2 + h^2)`
  - `phi0` 梯度感知写法：`phi0 / sqrt(phi0^2 + (eps*h*|grad(phi0)|)^2)`
  然后只在 `cfl ∈ {0.5, 0.6}`、`eps_sign_factor ≤ 1.5` 范围内扫 `frozen_phi0 / dynamic_phi`
- 原始结果：[`20260327_H07_small_cfl_method_preserving_sweep.txt`](/e:/Research/PDE/code1/test/results/20260327_H07_small_cfl_method_preserving_sweep.txt)
- 关键结果：
  - `dynamic_phi + plain-h sign + cfl=0.5 + eps=1.0`: numerical ratio `7.613`，origin ratio `28.174`
  - `dynamic_phi + phi0-grad sign + cfl=0.6 + eps=1.0`: numerical ratio `3.101`，origin ratio `9.906`
  - `dynamic_phi + phi0-grad sign + cfl=0.6 + eps=1.5`: numerical ratio `1.131`，origin ratio `2.230`
- 结论：如果坚持把 `cfl` 压在 `0.5~0.6`、并且不愿意把 `eps_sign_factor` 往 `2+` 提，那么“方法修正”仍然是有效的，但达不到 fully near-paper 的水平。也就是说，小改法能显著改善，但不足以完全取代较大的 reinit 细节修正。
