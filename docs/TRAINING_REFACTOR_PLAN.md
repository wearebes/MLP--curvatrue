# 训练并发与资源调度实施计划

## 1. 目标

本计划用于把当前训练系统收敛为一套可逐步实施、可逐步验证的规范化方案，适配以下真实运行场景：

- 开发服务器主要是 CPU 环境，用于开发、静态检查、轻量验证和产物检查。
- HPC 环境提供 GPU，用于正式训练、资源调度和性能验证。
- 训练系统不能只依赖 `job.sh` 兜底，Python 代码本身也必须具备明确的调度语义、资源约束和失败处理能力。

最终目标：

- Python 成为训练生命周期的主调度层。
- `job.sh` 收敛为 HPC 的薄包装入口。
- 训练在 CPU 开发机和 GPU HPC 上都有清晰、可验证的运行边界。
- 并发、内存、日志、SwanLab 和失败恢复行为全部具备明确规范。

## 2. 当前问题概览

当前系统已经完成了第一轮基础收敛，但仍然存在结构性问题：

- 根目录 `job.sh` 仍然承担了较多调度职责。
- Python 侧虽然已有 `prepare-run`、`worker-mode`、`aggregate-run`，但还没有形成正式的 dataset 级主入口。
- CPU 开发机与 GPU HPC 的环境边界还没有完全产品化，缺少独立的环境说明和自检入口。
- 训练稳定性仍然高度依赖运行时环境，特别是 CUDA 可用性、共享 GPU 并发和 HDF5 数据加载行为。
- 目前缺少一套分层验证链路，不能把“环境问题”“调度问题”“训练问题”“产物问题”清晰拆开。

## 3. 总体实施原则

后续实施遵循以下原则：

1. 不做一次性大改，按阶段逐步推进。
2. 每个阶段必须有明确产物、明确边界、明确验证方式。
3. 优先收敛环境和可观测性，再收敛调度，再优化性能。
4. Python 负责训练语义，Shell 负责作业包装。
5. 训练默认以 GPU 为正式执行目标，CPU 仅用于开发验证与显式允许的回退路径。

## 4. 目标架构

### 4.1 角色划分

- Python 主入口
  - 负责 dataset 生命周期调度。
  - 负责 rho worker 启动、等待、失败收集、降级策略和聚合。
  - 负责训练期资源策略与日志规范。
- rho worker
  - 负责单个 rho 的独立训练。
  - 只写本 rho 独占产物。
  - 不写共享 summary，不直接初始化共享 SwanLab run。
- `job.sh`
  - 负责 Slurm 提交入口。
  - 负责环境激活、GPU preflight 和调用 Python 主入口。
  - 不再承载核心训练调度语义。

### 4.2 生命周期单元

- dataset 是主生命周期单元。
- rho 是 dataset 内的并发 worker 单元。
- 一个 dataset 目录下聚合所有 rho 的训练结果与摘要。

### 4.3 并发拓扑

默认目标拓扑：

- 同一 Slurm task 内，dataset 串行推进。
- 进入某个 dataset 后，最多 3 个 rho 并发启动。
- 这 3 个 rho 共享当前 task 的 1 张 GPU。
- 出现 OOM 或资源异常后，后续 dataset 自动降级为单 rho 串行。

## 5. 分阶段实施计划

## 阶段 0：冻结架构和验收口径

### 目标

把后续改造的边界一次性说清楚，避免边改边变目标。

### 实施内容

- 确认 Python 主调度是最终方向。
- 确认 `job.sh` 只保留 HPC 薄包装职责。
- 确认 dataset 是主生命周期单元，rho 是内部 worker 单元。
- 确认 CPU 服务器与 GPU HPC 的职责边界。
- 确认 SwanLab 粒度为每个 dataset 一个 offline run。

### 交付物

- 本计划文档。

### 验收标准

- 后续改动都能对应到本计划中的某个阶段。

## 阶段 1：环境拆分与环境自检

### 目标

把基础环境和 GPU 训练环境显式拆开，先消除“到底跑在什么环境里”的不确定性。

### 实施内容

- 梳理依赖，拆分为通用依赖和 GPU 依赖。
- 设计环境文件方案，二选一或并行支持：
  - `environment.yaml`
- 新增环境自检入口，例如：
  - `python -m train env-check`
- 自检输出至少包括：
  - Python 版本
  - torch 版本
  - torch CUDA 版本
  - `torch.cuda.is_available()`
  - GPU 数量
  - `CUDA_VISIBLE_DEVICES`
  - Slurm GPU 相关变量
  - `nvidia-smi` 摘要

### 交付物

- 明确的基础/GPU 环境文件。
- 统一的 `env-check` CLI。
- 对应文档说明。

### 验收标准

- CPU 开发机可以运行 `env-check` 并明确显示无 GPU 的预期状态。
- HPC GPU 节点可以运行 `env-check` 并明确显示 GPU 可用状态。
- 不进入真实训练，也能先确认环境是否正确。

### 风险

- HPC 的模块系统和本地 Conda 环境可能冲突。
- GPU 环境依赖版本需要与集群驱动兼容。

## 阶段 2：Python dataset 主入口落地

### 目标

把当前分散在 `job.sh` 里的 dataset 生命周期调度正式迁移到 Python 中。

### 实施内容

- 新增正式主入口，例如：
  - `python -m train dataset --dataset <name> --batch-size <n>`
- 主入口内部负责：
  - prepare run
  - 生成 run 目录
  - 启动 rho worker
  - 等待所有 worker
  - 收集退出码
  - 聚合本 dataset summary
  - 更新 dataset 级 SwanLab
- 保留兼容接口：
  - `--prepare-run`
  - `--worker-mode`
  - `--aggregate-run`
- 把它们收敛为内部兼容层，而不是主要用户入口。

### 交付物

- 正式 dataset 主命令。
- 兼容旧接口的过渡实现。

### 验收标准

- 不依赖 `job.sh`，直接调用 Python 主命令，也能完整跑完一个 dataset 生命周期。
- 产物结构与当前 run 目录结构兼容或可平滑迁移。

### 风险

- CLI 兼容期内可能出现参数语义重叠。
- 现有测试和脚本需要同步更新调用方式。

## 阶段 3：资源策略与失败降级内建到 Python

### 目标

把“并发度、失败判定、降级逻辑、是否允许 CPU 回退”从 Shell 逻辑转移到 Python 训练系统。

### 实施内容

- Python 主调度内部维护 rho 并发上限。
- 默认值为 3，可配置，可在异常后自动降级为 1。
- 定义统一失败类型：
  - worker 退出码非 0
  - 缺失关键产物
  - OOM
  - CUDA 初始化失败
  - 数据文件损坏或维度错误
- 明确设备策略：
  - 真实训练默认要求 CUDA。
  - 仅显式指定 `--allow-cpu` 时允许 CPU 路径。
  - `prepare-run`、`aggregate-run`、`env-check` 不要求 CUDA。
- 把资源上下文统一写入结构化结果和文本日志。

### 交付物

- Python 内建的资源策略模块。
- 统一失败分类与降级逻辑。

### 验收标准

- 不依赖 `job.sh` 的额外判断，Python 主命令也能自主执行并发与降级。
- 人为制造单个 rho 失败后，后续调度行为符合预期。

### 风险

- 多进程下日志和状态同步容易出现时序问题。
- 共享 GPU 并发需要更稳的异常识别逻辑，避免误判。

## 阶段 4：数据加载与内存模型重构

### 目标

让训练数据层天然适配共享 GPU 并发和 CPU 开发验证，避免每个进程重复全量加载 HDF5。

### 实施内容

- 把数据加载策略显式化，至少提供：
  - 低内存流式模式
  - 小规模调试缓存模式
- 默认保持低内存模式。
- 明确 `num_workers=0` 是稳定默认值。
- 增加数据前置校验：
  - 文件存在
  - HDF5 可打开
  - X/Y key 存在
  - 样本数大于 0
  - shape 匹配
  - feature_dim 合法
- 评估是否需要 dataset 级共享缓存或更细粒度的复用策略。

### 交付物

- 显式数据加载配置。
- 数据校验入口或自动校验日志。

### 验收标准

- 三个 rho 并发时，RAM 压力明显小于旧版整表复制方式。
- 训练日志中能明确看出当前数据模式。

### 风险

- 流式读取会增加文件系统压力，需要结合 HPC 文件系统观察。
- 不同 HDF5 文件格式可能暴露兼容性问题。

## 阶段 5：日志、产物和 SwanLab 规范化

### 目标

让每个 run 的产物结构、状态文件和 SwanLab 日志形成稳定规范。

### 实施内容

- 固定 rho 独占产物命名：
  - `model_rho{rho}.pth`
  - `zscore_stats_{rho}.csv`
  - `training_rho{rho}.log`
  - `metrics_rho{rho}.json`
- 固定 dataset 聚合产物：
  - `dataset_metrics_summary.json`
  - `training_summary.txt`
  - `training_summary_partial.txt`
  - `training_failure_summary.json`
  - `training_failure_summary.txt`
  - `stop_summary.txt`
- 固定状态文件：
  - dataset 级 job state 文件
  - 运行时间与退出码记录
- SwanLab 收敛为：
  - 每个 dataset 一个 offline run
  - rho worker 不直接初始化共享 run
  - dataset 聚合阶段统一写入
- 增加 run 完整性验证命令，例如：
  - `python -m train verify-run --run-id <id>`

### 交付物

- 统一 run 目录结构。
- `verify-run` 或等价的产物检查入口。

### 验收标准

- 任意一个 run 都可以通过统一命令检查完整性。
- SwanLab 目录结构稳定，不再依赖被删除的 `tools/` 脚本。

### 风险

- 历史 run 目录格式与新格式可能并存，需要兼容策略。

## 阶段 6：验证体系建设

### 目标

把验证拆成 CPU 开发验证链路和 HPC GPU 训练验证链路，减少调试噪声。

### 实施内容

- CPU 开发验证链路：
  - 语法检查
  - CLI 参数检查
  - `env-check`
  - `prepare-run`
  - 假 metrics 的 `aggregate-run`
  - smoke test
- HPC GPU 验证链路：
  - GPU preflight
  - 单 dataset、单 rho
  - 单 dataset、三 rho
  - OOM / CUDA unavailable / worker fail 的降级测试
  - 完整 run 验证
- 重新补一套正式 smoke 测试，避免只靠手工命令验证。

### 交付物

- CPU smoke 测试脚本或测试用例。
- HPC 验证清单和最小训练用例。

### 验收标准

- CPU 环境能在不依赖 GPU 的情况下验证主要控制流。
- HPC 环境能在真实 GPU 上验证完整 dataset 生命周期。

### 风险

- GPU 真实训练耗时长，需要设计更小的测试集或更短的 smoke 配置。

## 阶段 7：稳定化 HPC 入口层

### 目标

在 Python 主调度成熟后，把 `job.sh` 收敛为稳定的 HPC 入口层。重点不是减少代码量，而是让大多数场景下的运行边界、preflight 和失败行为更可预期。

### 实施内容

- 保留：
  - Slurm 头
  - 环境激活
  - GPU preflight
  - 调用 Python 主调度入口
- 增加稳定性能力：
  - job 级 Python preflight
  - 输入数据与配置预校验
  - 结构化 preflight 日志
  - 明确的退出码和结束日志
- 移除或简化：
  - Shell 级 worker 池调度
  - Shell 级失败分类
  - Shell 级聚合逻辑

### 交付物

- 稳定化后的根目录 `job.sh`
- job 级 preflight 入口

### 验收标准

- `job.sh` 不再承担核心业务逻辑。
- 作业开始前即可发现绝大多数环境/输入问题。
- shell 和 Python 的职责边界清晰，失败日志可直接定位。

## 6. 实施顺序建议

推荐按以下顺序推进：

1. 阶段 1：环境拆分与 `env-check`
2. 阶段 2：Python dataset 主入口
3. 阶段 3：资源策略与失败降级
4. 阶段 4：数据加载与内存模型
5. 阶段 5：日志、产物和 SwanLab 规范
6. 阶段 6：验证体系
7. 阶段 7：稳定化 HPC 入口层

原因：

- 先把环境和可观测性收紧，才能降低后续改造噪声。
- 先把调度归属理顺，后续的数据、日志、失败处理才有稳定落点。
- 最后再稳定化 `job.sh`，风险最低。

## 7. 第一阶段建议的具体落点

建议马上开始实施阶段 1，目标如下：

- 明确 CPU 开发环境和 GPU 训练环境文件。
- 新增 `python -m train env-check`。
- 让 CPU 开发机和 GPU HPC 都能先跑通环境自检。
- 将“CUDA unavailable”从训练期故障前移为环境期故障。

阶段 1 完成之后，再进入 Python dataset 主入口的实现。

## 8. 每阶段验收记录建议

建议后续每完成一个阶段，就在本文件底部补一段简短记录：

- 完成时间
- 修改文件
- 验证命令
- 风险与遗留问题

这样可以把整个重构过程沉淀成完整的工程记录。

## 9. 当前状态

截至本计划编写时，项目已经具备以下基础：

- 根目录已有正式 `job.sh`
- 已有 GPU preflight
- 已有 `prepare-run` / `worker-mode` / `aggregate-run`
- 已有低内存 HDF5 默认模式
- 已有 dataset 级聚合与失败摘要基础能力

后续工作重点不再是“从零开始搭架子”，而是把已有基础能力继续上移到更规范的 Python 主调度体系中，并补齐环境、验证和长期维护能力。

## 10. 阶段验收记录

### 阶段 1：环境拆分与环境自检

完成时间：

- 2026-04-16

修改文件：

- [train/__main__.py](/e:/Research/PDE/code1/train/__main__.py:1)
- [train/utils.py](/e:/Research/PDE/code1/train/utils.py:1)
- [environment.yaml](/e:/Research/PDE/code1/environment.yaml:1)
- [docs/ENVIRONMENTS.md](/e:/Research/PDE/code1/docs/ENVIRONMENTS.md:1)

完成内容：

- 新增 `python -m train --env-check`
- 新增 `python -m train --env-check --require-cuda`
- 新增 CPU / GPU 分层依赖文件
- 新增 CPU / GPU 环境说明文档

验证命令：

```bash
python -m py_compile train/__main__.py train/utils.py
python -m train --env-check
CUDA_VISIBLE_DEVICES='' python -m train --env-check
CUDA_VISIBLE_DEVICES='' python -m train --env-check --require-cuda
```

验证结果：

- 语法检查通过
- `env-check` 可正常输出环境报告
- 在 GPU 可见时，环境报告显示 `cuda_available: true`
- 在隐藏 GPU 后，环境报告显示 `cuda_available: false`
- `--require-cuda` 在 CUDA 不可用时正确返回非零退出码

遗留问题：

- 当前仅完成环境拆分与环境自检，Python dataset 主调度尚未落地
- `swanlab` 版本已纳入新依赖文件，但仍需在真实目标环境中完成一次安装验证

### 阶段 2：Python dataset 主入口

完成时间：

- 2026-04-16

修改文件：

- [train/__main__.py](/e:/Research/PDE/code1/train/__main__.py:1)
- [job.sh](/e:/Research/PDE/code1/job.sh:1)

完成内容：

- 新增 `python -m train --dataset-run`
- 新增 `--rho-max-concurrent`
- 将 dataset 生命周期的以下职责迁入 Python：
  - prepare run
  - 启动 rho worker
  - 等待 worker 结束
  - 写 dataset 级状态文件
  - 调用聚合逻辑生成 summary
- 保留 `--prepare-run` / `--worker-mode` / `--aggregate-run` 作为兼容层
- 将根目录 `job.sh` 收敛为更薄的包装层：
  - 负责环境激活
  - 负责 GPU preflight
  - 负责按 dataset 调用 `--dataset-run`
  - 支持 `TRAIN_CONFIG_PATH` 环境变量

验证命令：

```bash
python -m py_compile train/__main__.py train/utils.py train/trainer.py
bash -n job.sh
python -m train --help
python -m train --dataset tiny_ds --config test/fixtures/phase2_smoke/train_config.yaml --run-id phase2_smoke_run --rho 256 266 --dataset-run --rho-max-concurrent 2 --allow-cpu
TRAIN_CONFIG_PATH=test/fixtures/phase2_smoke/train_config.yaml DATASETS=tiny_ds RHOS='256 266' BATCH_SIZES='8' CONDA_ENV_NAME=jq RHO_MAX_CONCURRENT=2 bash job.sh
```

验证结果：

- Python 静态语法检查通过
- `job.sh` shell 语法检查通过
- CLI 帮助已显示 `--dataset-run` 和 `--rho-max-concurrent`
- 通过 synthetic HDF5 数据集在 CPU 允许模式下完成了完整 dataset 生命周期
- 通过根目录 `job.sh` 在 GPU 可用环境下完成了同一 smoke 流程
- 产物已正确生成：
  - `dataset_job_state.txt`
  - `dataset_metrics_summary.json`
  - `training_summary.txt`
  - `metrics_rho*.json`
  - `model_rho*.pth`

遗留问题：

- 失败降级和资源保护策略虽然已有基础行为，但还没有完全以内建资源策略模块的形式整理
- `job.sh` 仍保留 batch/dataset 外层循环，尚未收敛到极薄最终形态

### 阶段 3：资源策略与失败降级

完成时间：

- 2026-04-16

修改文件：

- [train/__main__.py](/e:/Research/PDE/code1/train/__main__.py:1)
- [train/trainer.py](/e:/Research/PDE/code1/train/trainer.py:1)
- [job.sh](/e:/Research/PDE/code1/job.sh:1)

完成内容：

- 新增 Python 级总调度入口：
  - `python -m train --schedule-run`
- 新增跨 dataset / batch size 的 Python 调度能力：
  - 多 dataset
  - 多 batch size
  - 内建 rho 并发度降级
- 新增统一失败分类：
  - `input_data`
  - `missing_metrics`
  - `missing_artifact`
  - `resource_oom`
  - `cuda_unavailable`
  - `cuda_runtime`
  - `worker_failure`
- 资源保护触发逻辑不再只依赖 shell，改为优先基于 worker payload 分类判断
- 新增 Python 级 schedule summary：
  - `model/_schedules/schedule_summary_*.json`
- 将根目录 `job.sh` 进一步收敛为单次调用 Python：
  - 只做环境激活
  - 只做 GPU preflight
  - 只调用一次 `--schedule-run`

验证命令：

```bash
python -m py_compile train/__main__.py train/trainer.py train/utils.py
bash -n job.sh
python -m train --help
python -m train --config test/fixtures/phase3_schedule/train_config.yaml --datasets tiny_fail tiny_ok --batch-sizes 8 --rho 256 266 --schedule-run --rho-max-concurrent 2 --allow-cpu
TRAIN_CONFIG_PATH=test/fixtures/phase2_smoke/train_config.yaml DATASETS=tiny_ds RHOS='256 266' BATCH_SIZES='8' CONDA_ENV_NAME=jq RHO_MAX_CONCURRENT=2 bash job.sh
```

验证结果：

- Python 静态语法检查通过
- `job.sh` shell 语法检查通过
- CLI 帮助已显示 `--schedule-run`、`--datasets`、`--batch-sizes`
- 通过 phase3 fixture 验证了 Python 级降级逻辑：
  - 第一个 dataset 因缺失 HDF5 文件失败
  - 后续 dataset 自动降级为 `rho_max_concurrent=1`
  - 第二个 dataset 成功完成训练
- `schedule_summary_*.json` 正确记录了：
  - 失败 dataset 的失败分类
  - 后续成功 dataset 的实际并发度
- 根目录 `job.sh` 已通过 phase2 成功路径 smoke，说明 shell 已经退化为薄包装且不再持有核心调度循环

遗留问题：

- CPU 线程策略目前仍以“记录和遵守环境变量”为主，尚未抽成独立资源策略模块
- 还没有单独的 `verify-run` / `verify-schedule` CLI

### 阶段 4：数据加载与内存模型

完成时间：

- 2026-04-16

修改文件：

- [train/config.py](/e:/Research/PDE/code1/train/config.py:1)
- [train/config.yaml](/e:/Research/PDE/code1/train/config.yaml:1)
- [train/data.py](/e:/Research/PDE/code1/train/data.py:1)
- [train/trainer.py](/e:/Research/PDE/code1/train/trainer.py:1)
- [train/__main__.py](/e:/Research/PDE/code1/train/__main__.py:1)
- [test/fixtures/phase2_smoke/train_config.yaml](/e:/Research/PDE/code1/test/fixtures/phase2_smoke/train_config.yaml:1)
- [test/fixtures/phase2_smoke/train_config_in_memory.yaml](/e:/Research/PDE/code1/test/fixtures/phase2_smoke/train_config_in_memory.yaml:1)
- [test/fixtures/phase3_schedule/train_config.yaml](/e:/Research/PDE/code1/test/fixtures/phase3_schedule/train_config.yaml:1)

完成内容：

- 新增显式 `data_loading` 配置段：
  - `mode`
  - `num_workers`
  - `stats_chunk_size`
  - `pin_memory`
- 将数据加载模式收敛为显式二选一：
  - `stream`
  - `in_memory`
- `build_dataloaders(...)` 不再直接接受裸 `in_memory` 布尔值，而是通过显式模式和参数驱动
- 训练日志现在会记录：
  - 数据加载模式
  - `num_workers`
  - `stats_chunk_size`
  - `pin_memory`
- 新增数据校验 CLI：
  - `python -m train --validate-data`
- 训练开始前会把 dataset 级校验报告写入 `dataset_job_state.txt`

验证命令：

```bash
python -m py_compile train/__main__.py train/config.py train/data.py train/trainer.py
python -m train --dataset tiny_ds --config test/fixtures/phase2_smoke/train_config.yaml --rho 256 266 --validate-data
python -m train --dataset tiny_fail --config test/fixtures/phase3_schedule/train_config.yaml --rho 256 266 --validate-data
CUDA_VISIBLE_DEVICES='' python -m train --dataset tiny_ds --config test/fixtures/phase2_smoke/train_config_in_memory.yaml --run-id phase4_in_memory_run --rho 256 266 --dataset-run --rho-max-concurrent 2 --allow-cpu
```

验证结果：

- Python 静态语法检查通过
- `--validate-data` 成功路径可输出结构化校验报告
- `--validate-data` 失败路径在缺失 HDF5 文件时正确返回非零
- `in_memory` 模式已通过小数据集完整训练验证
- 训练日志和 `dataset_job_state.txt` 已记录显式数据模式与校验报告

遗留问题：

- 目前数据加载模式只有 `stream` 和 `in_memory`，还没有更细粒度的共享缓存层
- 还没有单独的数据校验批量入口，例如 `--validate-datasets`

### 阶段 5：日志、产物和 SwanLab 规范

完成时间：

- 2026-04-16

修改文件：

- [train/__main__.py](/e:/Research/PDE/code1/train/__main__.py:1)

完成内容：

- 扩展 `train_config.yaml` 快照，新增：
  - `data_loading`
  - `tracking`
- 新增 run 级产物检查入口：
  - `python -m train --verify-run --run-id <id>`
- 新增 schedule 级产物检查入口：
  - `python -m train --verify-schedule-path <path>`
- `verify-run` 现在统一检查：
  - `train_config.yaml`
  - `dataset_job_state.txt`
  - `dataset_metrics_summary.json`
  - `training_summary*.txt`
  - `training_failure_summary.*`
  - `metrics_rho*.json`
  - `model_rho*.pth`
  - `zscore_stats_*.csv`
  - `training_rho*.log`
  - SwanLab 目录状态
- `verify-schedule` 现在统一检查：
  - schedule summary JSON 是否存在
  - 每个 entry 的 state file / summary file 是否存在
  - 成功 entry 是否具备对应 summary

验证命令：

```bash
python -m py_compile train/__main__.py train/utils.py train/trainer.py train/data.py train/config.py
python -m train --verify-run --config test/fixtures/phase2_smoke/train_config_in_memory.yaml --run-id phase4_in_memory_run
python -m train --verify-schedule-path test/fixtures/phase3_schedule/model/_schedules/schedule_summary_20260416_161618.json
python -m train --dataset tiny_ds --config test/fixtures/phase2_smoke/train_config.yaml --run-id phase5_verify_run --rho 256 266 --dataset-run --rho-max-concurrent 2 --allow-cpu --max-epochs 1 --patience 1
python -m train --verify-run --config test/fixtures/phase2_smoke/train_config.yaml --run-id phase5_verify_run
```

验证结果：

- Python 静态语法检查通过
- `verify-run` 可正确检查已有成功 run
- `verify-schedule` 可正确检查已有 schedule summary
- 新生成的 `phase5_verify_run/train_config.yaml` 已包含：
  - `data_loading`
  - `tracking`
- `verify-run` 对新 run 返回 `status: ok`

遗留问题：

- 当前还没有单独的 `verify-dataset-state` 或 `verify-swanlab` 入口
- 历史 run 的 `train_config.yaml` 可能不含新字段，因此检查逻辑需要继续保持向后兼容

### 阶段 6：验证体系

完成时间：

- 2026-04-16

修改文件：

- [test/smoke_train_pipeline.py](/e:/Research/PDE/code1/test/smoke_train_pipeline.py:1)
- [VALIDATION.md](/e:/Research/PDE/code1/VALIDATION.md:1)

完成内容：

- 新增 CPU 集成 smoke 脚本：
  - `python test/smoke_train_pipeline.py`
- smoke 脚本现在覆盖：
  - `env-check`
  - `validate-data` 成功路径
  - `validate-data` 失败路径
  - `dataset-run` CPU 允许模式
  - `verify-run`
  - `schedule-run` 降级路径
  - `verify-schedule`
- 新增验证说明文档：
  - CPU 验证链
  - HPC GPU 验证链
  - `verify-run`
  - `verify-schedule`
  - 结果解释

验证命令：

```bash
python -m py_compile test/smoke_train_pipeline.py train/__main__.py
python test/smoke_train_pipeline.py
```

验证结果：

- Python 静态语法检查通过
- `test/smoke_train_pipeline.py` 整体通过
- CPU 验证链条已经形成单命令闭环

遗留问题：

- HPC GPU 验证链目前以文档和命令清单为主，还没有单独的集群自动化脚本
- 未来可以补一份更严格的 regression smoke，对 schedule summary 做更细的断言

### 阶段 7：稳定化 HPC 入口层

完成时间：

- 2026-04-16

修改文件：

- [train/__main__.py](/e:/Research/PDE/code1/train/__main__.py:1)
- [job.sh](/e:/Research/PDE/code1/job.sh:1)

完成内容：

- 新增 Python 级 job preflight：
  - `python -m train --preflight-schedule`
- preflight 现在统一检查：
  - 配置是否可加载
  - 数据集列表和 batch size 列表是否有效
  - 每个 dataset / rho 的 HDF5 输入是否齐全
  - 环境报告和 CUDA 状态
- 根目录 `job.sh` 现在采用稳定包装层模式：
  - shell 级 GPU preflight
  - Python 级 schedule preflight
  - 通过后才执行 `--schedule-run`
  - 通过 EXIT trap 记录结束时间和退出码
- 新增 job 级 preflight 日志：
  - `logs/schedule_preflight_<jobid>.json`

验证命令：

```bash
python -m py_compile train/__main__.py
bash -n job.sh
python -m train --config test/fixtures/phase2_smoke/train_config.yaml --datasets tiny_ds --batch-sizes 8 --rho 256 266 --preflight-schedule
CUDA_VISIBLE_DEVICES='' python -m train --config test/fixtures/phase3_schedule/train_config.yaml --datasets tiny_fail tiny_ok --batch-sizes 8 --rho 256 266 --preflight-schedule --require-cuda
TRAIN_CONFIG_PATH=test/fixtures/phase2_smoke/train_config.yaml DATASETS=tiny_ds RHOS='256 266' BATCH_SIZES='8' CONDA_ENV_NAME=jq RHO_MAX_CONCURRENT=2 bash job.sh
```

验证结果：

- Python 静态语法检查通过
- `job.sh` shell 语法检查通过
- `--preflight-schedule` 成功路径可输出完整结构化报告
- `--preflight-schedule --require-cuda` 失败路径会同时暴露 CUDA 不可用和数据问题
- `job.sh` 成功路径已通过：
  - GPU preflight 通过
  - schedule preflight 通过
  - `--schedule-run` 正常执行
  - `logs/job_summary_local.log` 记录结束时间和退出码

遗留问题：

- 当前 `job.sh` 仍保留一份 shell 级 GPU preflight；后续如果 Python preflight进一步成熟，可以评估是否合并
- HPC 集群侧仍需要一次真实 `sbatch` 验证，确认 Slurm 环境变量和日志路径符合预期
