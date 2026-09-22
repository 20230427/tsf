# DD-Mamba / TSP

面向多变量时间序列长期预测的 PyTorch 研究仓库。模型将同一段历史窗口并行送入时域分支和频域分支，再通过可学习融合模块得到多步预测；仓库同时包含主结果、统一基线、单因素消融、累积删减、超参数搜索、鲁棒性与效率实验脚本。

> 本 README 依据当前仓库的源码、配置和实验脚本整理。实际可复现性以配置文件、结果 JSON 中的 provenance，以及当前数据/依赖环境为准。

## 1. 模型概览

输入、输出张量均采用批次优先格式：

- 输入历史窗口：B × L × C
- 输出预测窗口：B × H × C
- B 为批大小，L 为输入长度，H 为预测长度，C 为变量数

整体执行链如下：

~~~text
原始序列
  │
  ├─ 仅用训练集统计量做全局 StandardScaler
  ├─ 按时间顺序切分并构造滑动窗口
  │
  └─ 可选 RevIN / α-RevIN
        │
        ├──────────── 时域分支 ────────────┐
        │  移动平均分解                     │
        │  DLinear 式 L→H 线性骨干          │
        │  Mamba / BiMamba / Attention / MLP│
        │  可选跨变量 mixer                 │
        │                                   ├─ 融合 ─ 可选 dispersion ─ RevIN 还原 ─ 预测
        └──────────── 频域分支 ────────────┘
           rFFT + 高频稀疏化
           复线性滤波或频率轴 BiMamba
           可选 FITS 频谱骨干
           可选跨变量 mixer
~~~

### 1.1 时域分支

src/models/time_branch.py 包含三部分：

1. 移动平均分解，将输入拆成趋势项与季节项。
2. DLinear 风格的线性预测骨干，直接完成 L→H 映射。
3. 可选的时序依赖编码器，通过零初始化修正头叠加到线性预测上。

model.time_encoder 支持：

- mamba / unimamba：单向选择性状态空间模型；
- bimamba：双向 Mamba；
- attention：自注意力；
- mlp：沿时间维的 MLP；
- none：移除时序编码器，仅保留线性骨干。

### 1.2 频域分支

src/models/freq_branch.py 对输入执行实数 FFT，并可按 model.freq_sparsity 丢弃高频分量。频谱编码器有两种：

- linear：复数线性滤波；
- mamba：沿频率 bin 运行双向 Mamba。

model.freq_backbone=fits 时，额外使用 FITS 风格的频域线性外推骨干。最终结果经 irFFT 或预测头映射回 H × C。

### 1.3 跨变量 mixer

model.channel_mixer_layers 控制跨变量依赖建模的层数，model.mixer_placement 控制放置位置：

- both：时域、频域各自使用独立 mixer；
- shared：两分支共享同一 mixer；
- time：仅时域分支；
- freq：仅频域分支。

model.mixer_kind 支持 bimamba、unimamba 和 attention。

### 1.4 融合

src/models/fusion.py 实现：

- gated、sum、concat：凸组合类融合；
- residual、affine、doubly_residual：允许两分支做加性叠加；
- time_only、freq_only：用于分支消融。

注意：time_only 和 freq_only 在当前 DualDomainForecaster.forward 中仍会计算两个分支，只是最终输出及有效梯度路径采用其中一个。因此 active parameter count 可用于描述概念上的活跃参数，但不能直接代替实测吞吐、显存或延迟。

### 1.5 归一化与尺度

仓库存在两层尺度处理：

1. 数据集级 StandardScaler：只在训练区间拟合；
2. 窗口级 RevIN / α-RevIN：在模型内部归一化并还原。

当前训练和 src.evaluate 均不会把输出再通过数据集 StandardScaler.inverse_transform 还原到原始物理单位。因此默认 MSE、MAE 和预测图是在“训练集标准化后的数据空间”中计算，不应误写成原始单位误差。

## 2. 快速开始

### 2.1 安装

项目声明 Python >= 3.10。PyTorch 应先按本机 CUDA/CPU 环境单独安装，再安装仓库依赖：

~~~powershell
python -m venv .venv
..venvScriptsActivate.ps1

# 按本机 CUDA 版本选择官方 PyTorch wheel；下行仅为仓库验证过的一个示例
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"
~~~

可选依赖：

~~~powershell
pip install -e ".[hpo]"       # Optuna
pip install -e ".[tables]"    # openpyxl，生成 xlsx
~~~

官方 Mamba CUDA 内核是可选项：

~~~powershell
pip install "causal-conv1d>=1.4.0"
pip install "mamba-ssm>=2.2.0"
~~~

如果未安装，DD-Mamba 的 MambaLayer 会使用纯 PyTorch 扫描实现。smamba 官方基线没有纯 PyTorch 回退，必须使用可用的官方内核和 CUDA。

### 2.2 无数据冒烟测试

~~~powershell
python scripts/run_synthetic.py
~~~

也可以直接运行默认合成数据配置：

~~~powershell
python -m src.train --config configs/default.yaml --train.epochs 2
~~~

### 2.3 训练与评估

~~~powershell
python -m src.train --config configs/ETTh1.yaml --data.pred_len 336
python -m src.evaluate --checkpoint checkpoints/ETTh1_best.pt --plot forecast.png --channel 0
~~~

实际 checkpoint 名由 experiment.name 决定，不一定等于上例。训练开始后请以日志打印和配置为准。

CPU 运行时建议显式禁用官方 Mamba：

~~~powershell
python -m src.train --config configs/default.yaml --train.device cpu --model.use_official_mamba false
~~~

## 3. 数据与切分协议

### 3.1 支持的数据源

src/data/dataset.py 支持：

- synthetic：运行时生成合成序列；
- csv：宽表 CSV，通常首列为时间戳、其余数值列为变量；
- 无表头 txt / txt.gz；
- npz：PEMS 数据，读取约定的交通流特征；
- pt：缓存后的张量数据。

配置虽然可能写 data/ETTh1.csv，但加载器会静默优先查找同名 .pt 缓存，包括 data 根目录及 ETT-small、weather、electricity、traffic、exchange_rate、Solar、PEMS、illness 等已知子目录。日志出现下列信息时，以实际打印路径为准：

~~~text
[data] using pt cache: ...
~~~

### 3.2 数据集配置

| 数据组 | 配置 | 输入长度 | 推荐预测长度 |
|---|---|---:|---|
| ETT hourly | ETTh1、ETTh2 | 96 | 96 / 192 / 336 / 720 |
| ETT minute | ETTm1、ETTm2 | 96 | 96 / 192 / 336 / 720 |
| 常规长序列 | weather、electricity、solar、exchange_rate、traffic | 96 | 96 / 192 / 336 / 720 |
| PEMS | PEMS03、PEMS04、PEMS07、PEMS08 | 96 | 12 / 24 / 48 / 96 |
| Illness | illness | 36 | 24 / 36 / 48 / 60 |

ETT 必须保留配置中的 data.split_protocol=ETTh 或 ETTm，以使用固定的 12/4/4 月边界。改成比例切分后，结果不能与常见文献协议直接比较。

PEMS 使用 source=npz；Illness 是周粒度数据。scripts/run_benchmarks.sh 的默认数据集集合不包含全部 PEMS，也没有注册 Illness，运行这些数据集时应直接使用对应 YAML 或显式使用支持它们的矩阵脚本。

### 3.3 自定义数据

最小配置示例：

~~~yaml
data:
  source: csv
  csv_path: data/my_series.csv
  target_columns: null
  seq_len: 96
  pred_len: 96
  train_ratio: 0.7
  val_ratio: 0.1
  split_protocol: ratio
  scale: true
~~~

切分严格按时间顺序完成。StandardScaler 只在训练段拟合，验证和测试共享训练统计量。

## 4. 项目结构

~~~text
.
├─ configs/                       数据集配置、HPO/基线搜索空间
├─ src/
│  ├─ train.py                    训练、早停、checkpoint、最终测试
│  ├─ evaluate.py                 checkpoint 重载、指标与绘图
│  ├─ data/
│  │  ├─ dataset.py               数据解析、缓存解析、滑动窗口、缩放
│  │  └─ data_loader.py           train/val/test DataLoader
│  ├─ models/
│  │  ├─ dual_domain_model.py     DD-Mamba 顶层模型与 RevIN
│  │  ├─ time_branch.py           时域分解、线性骨干、时序编码器
│  │  ├─ freq_branch.py           FFT、稀疏化、频谱编码
│  │  ├─ fusion.py                凸组合与加性融合
│  │  ├─ mamba_block.py           官方内核选择与纯 PyTorch 回退
│  │  ├─ dispersion.py            fixed / learned 尺度预测
│  │  ├─ baselines.py             统一基线注册表
│  │  └─ smamba_official.py       官方 S-Mamba 适配
│  ├─ hpo/                        Optuna 目标、搜索空间、GPU 预算、study 管理
│  └─ utils/                      配置、CLI、指标、provenance、原子写文件
├─ scripts/
│  ├─ run_ablation.py             单数据集单因素消融
│  ├─ run_ablation_matrix.py      多数据集 × 多 horizon × 多 seed 消融
│  ├─ analyze_ablation_matrix.py  配对统计、置信区间、BH 校正
│  ├─ run_branch_matrix.py        full / time-only / freq-only
│  ├─ run_selection_protocol.py   验证集筛选后冻结候选并最终测试
│  ├─ run_unified_baselines.py    统一基线复现
│  ├─ run_main_results.py         主结果矩阵
│  ├─ run_channel_order_ablation.py 变量顺序敏感性
│  ├─ run_input_length_sensitivity.py 输入长度敏感性
│  ├─ robustness/                 测试时噪声鲁棒性
│  ├─ hpo/                        Optuna CLI、汇总、清理、表格更新
│  └─ ablation/                   累积删减注册表、启动与汇总
├─ tests/                         模型、协议、统计、provenance 测试
├─ results/                       已整理、受版本控制的结果 JSON
├─ checkpoints/                   运行时 checkpoint/结果，通常被忽略
├─ output/                        批量实验与累积消融产物
├─ documents/                     已有实验和问题分析
├─ assets/                        规范、图表说明、Optuna 文档、工作簿
└─ efficiency_experiment/         独立效率实验包及一份 vendored src 副本
~~~

efficiency_experiment/src 是独立快照。修改它不会修改根目录 src，反之亦然；比较效率实验时应先确认两份实现是否仍一致。

## 5. 配置与命令行覆盖

训练入口首先读取 YAML，再应用未知命令行参数作为覆盖项：

~~~powershell
python -m src.train --config configs/weather.yaml --data.pred_len 192 --model.d_model 256 --train.lr 0.0005
~~~

使用时注意：

- 最稳妥的格式是 --section.key value；
- 裸字段只有在 YAML 中已存在唯一同名键时才会生效；
- 无法识别的裸字段可能被放入 cfg.cli 而不影响训练；
- 点号只按第一个点拆分，配置层级实际只支持 section.key；
- 通用解析器会把 none 和 null 转为 Python None。

最后一点会影响需要字符串值 "none" 的枚举，例如 model.time_encoder、model.freq_backbone 和 model.dispersion。要把这些字段设为字符串 none，请编辑/物化 YAML，或使用在 Python 内直接修改配置字典的专用实验脚本。直接基线切换应使用 --model.arch，而不是 --arch。

## 6. 训练、输出与 provenance

训练流程为：

1. 固定随机种子；
2. 构造 train/validation，按需构造 test；
3. 用验证损失早停并保存最佳 checkpoint；
4. 普通最终运行会在最佳 checkpoint 上评估测试集；
5. 原子写入结果 JSON。

默认输出：

~~~text
checkpoints/<experiment.name>_best.pt
checkpoints/<experiment.name>_results.json
~~~

文件名不自动包含 horizon 或 seed。手工循环实验时若复用 experiment.name，会覆盖前一次结果；批量 runner 会主动生成唯一名字。

新结果 JSON 包含解析后的配置、配置 SHA-256、seed、代码提交信息等 provenance。可验证：

~~~powershell
python scripts/validate_provenance.py checkpoints/my_results.json
~~~

旧格式结果会被标记为 legacy_unverifiable，不应混入新的可恢复实验集合。当前目录如果不是 Git checkout，提交字段会缺失；这不影响训练，但削弱代码版本可追溯性。

## 7. 基线与其他实验

### 7.1 统一基线

统一注册表包含：

~~~text
dlinear, nlinear, rlinear, patchtst, itransformer, smamba, msmamba,
crossformer, tide, fedformer, autoformer, tf4tf
~~~

示例：

~~~powershell
python scripts/run_unified_baselines.py --config configs/ETTh1.yaml --horizons 96 192 --seeds 3
~~~

该入口一次接收一个数据集配置，多数据集实验需分别调用。smamba 依赖官方 CUDA Mamba；tf4tf 是外部适配器，使用前需检查额外依赖和资源是否存在。

### 7.2 主结果

~~~powershell
python scripts/run_main_results.py --config configs/ETTh1.yaml --horizons 96 192 336 720 --seeds 5
~~~

批量 shell 脚本依赖 bash，部分启动器还依赖 nohup、flock 或调度器。在原生 Windows PowerShell 中应使用 Python 入口，或改在 WSL / Linux / Git Bash 中运行。

### 7.3 HPO

~~~powershell
python scripts/hpo/run_hpo.py --config configs/weather.yaml --pred_len 96 --n_trials 50 --metric val_loss
python scripts/hpo/hpo_dump.py --list
python scripts/hpo/hpo_clean.py
~~~

重要约束：

- 默认 HPO 目标遵循仓库中的 Pamba 协议，会按 trial 查看 test MSE；论文级无泄漏选择应显式使用 --metric val_loss；
- --pred_len all 固定展开为 96/192/336/720，不适用于 PEMS 和 Illness；
- src/hpo/gpu_budget.py 使用 fcntl，原生 Windows 下不可用，HPO 建议在 Linux/WSL 运行；
- --bg 只重定向日志，不负责让进程真正脱离当前终端；
- 清理脚本默认 dry-run，且不会删除 JSON 结果。

## 8. 消融实验设计

本仓库实际上实现了两种含义不同的消融：独立单因素消融和顺序相关的累积删减。论文中应分别报告。

### 8.1 独立单因素消融：推荐用于组件贡献

入口：

~~~powershell
python scripts/run_ablation.py --config configs/weather.yaml --chain core --seeds 5 --validation-only
~~~

设计原则：

- 每个 variant 都从同一个数据集基准配置深拷贝；
- 只修改目标组件，其余数据切分、训练日程和 seed 保持一致；
- 相同 seed 在 full 与各 variant 之间配对；
- 结果中的 ΔMSE 定义为 variant MSE - full MSE，正值表示移除/替换后变差；
- resolve_chain 会根据数据集当前配置解析方向并移除 no-op。

内置 chain：

| chain | 研究问题 |
|---|---|
| core | mixer、时序编码器、双分支、频域编码、融合、线性骨干、RevIN |
| smamba | 按 S-Mamba 风格替换/移除 VC 与 TD 模块 |
| td | Mamba、BiMamba、Attention、MLP、无 TD 编码器 |
| placement | mixer 位于 both/shared/time/freq 或完全移除 |
| init | 时域头、频域头、FITS 骨干、融合初始化 |
| spectral | FITS、复线性/频率 Mamba、频率稀疏率 |
| fusion | sum 与三种加性融合规则 |
| revin_alpha | RevIN 开关、全局/逐通道可学习 α |
| dispersion | 无归一化、RevIN、fixed、learned dispersion |

也可显式选择变体：

~~~powershell
python scripts/run_ablation.py --config configs/ETTh1.yaml --variants full time_only freq_only no_linear_backbone --seeds 5 --out checkpoints/ablation_ETTh1_h96.json
~~~

默认模式会评估测试集，适合已经预注册的最终对照，不适合从许多候选中挑最优方案。探索阶段应添加 --validation-only；该模式不会构造 test split。

一个实现层面的混杂项是 with_fits：当 freq_zero_init_head=auto 时，启用 FITS 还可能改变频域预测头是否零初始化。因此：

- 研究“是否需要 FITS”时可使用 spectral chain；
- 研究“初始化是否有效”时应单独使用 init chain；
- 不要把 with_fits 的差异完全归因于频谱映射本身。

### 8.2 多数据集、多 horizon、多 seed 矩阵

~~~powershell
python scripts/run_ablation_matrix.py --configs configs/ETTh1.yaml configs/weather.yaml --horizons 96 192 336 720 --chain core --seeds 6 --out-dir checkpoints/ablation_matrix
python scripts/analyze_ablation_matrix.py --input-dir checkpoints/ablation_matrix --pattern "*_core.json"
~~~

矩阵 runner 为每个 dataset × horizon × variant × seed 生成唯一实验名和结果。分析器会：

- 校验 horizon 完整性、seed 配对、重复记录和 evaluation scope；
- 以“每个 seed 内先对 horizon 求平均”作为主汇总；
- 同时保留逐 horizon 的次级结果；
- 计算配对均值差、置信区间、原始 p 值；
- 按 family/scope 执行 Benjamini-Hochberg 多重比较校正；
- 在小样本条件下提供精确符号翻转检验。

这条流程比单次 run_ablation.py 输出的总体均值/总体标准差更适合形成论文中的显著性结论。

### 8.3 双分支消融

~~~powershell
python scripts/run_branch_matrix.py --datasets ETTh1 weather electricity --horizons 96 192 336 720 --seeds 5 --out checkpoints/branch_matrix.json
~~~

该脚本固定比较 full、time_only、freq_only。其内置数据集注册表目前主要覆盖九个常规长序列数据集；PEMS 和 Illness 请使用 run_ablation_matrix.py 或直接使用对应配置。

### 8.4 累积删减：用于展示退化路径

累积链定义在 scripts/ablation/cumulative_registry.py。后一步继承前面所有修改：

~~~text
Full DD-Mamba
  → fusion=sum
  → freq_backbone=none 且 freq_sparsity=0
  → channel_mixer_layers=0
  → time_linear_backbone=false
  → use_revin=false
  → fusion=time_only
~~~

这类表适合回答“模型逐步简化时性能如何变化”，但每一步的差值依赖删减顺序，不能解释为该组件独立的因果贡献。另有两个注意点：

- no_freq_enh 同时修改 freq_backbone 和 freq_sparsity，不是严格单因素；
- 最后一行显示名是 Time-only Mamba，但注册表只设置 fusion=time_only，没有强制 time_encoder=mamba。对于以 MLP 为基础配置的 Electricity、Traffic、PEMS，实际末端是 time-only MLP。

运行与汇总：

~~~bash
CUM_GEN_EXTRA="--datasets ETTh1 weather" CUM_COLLECT_EXTRA="--datasets ETTh1 weather" bash scripts/ablation/run_all_cumulative.sh
~~~

这些启动脚本面向 bash/集群环境。当前默认数据集波次是 ETTh2、ETTm1、PEMS08、illness；为了让实验清单可复核，建议总是显式传 --datasets。已有汇总位于 output/cumulative_ablation/。

### 8.5 变量顺序与输入长度敏感性

变量顺序压力测试会对输入和目标施加同一个变量置换，并比较 BiMamba、Attention、无 mixer：

~~~powershell
python scripts/run_channel_order_ablation.py --help
~~~

输入长度敏感性：

~~~powershell
python scripts/run_input_length_sensitivity.py --help
~~~

### 8.6 推荐的论文级消融协议

1. 预先声明主数据集、horizon、variant family、seed 数和主指标。
2. 探索与候选筛选只使用 validation-only。
3. 使用 scripts/run_selection_protocol.py 冻结候选后，再单独进行一次最终测试。
4. full 与 variant 使用相同 seed；建议至少 5 个，资源允许时使用 10 个。
5. 主分析在每个 seed 内先平均 horizon，再做配对差；逐 horizon 结果作为次级分析。
6. 同一 family 内对 p 值做 BH 校正，并同时报告效应量和置信区间。
7. 同时报 stored/trainable params、active params 和实测延迟/显存；不要用 active params 推断速度。
8. 保存完整 provenance。单 seed 下小于约 0.005 MSE 的差异应优先视为噪声信号，再通过多 seed 验证。

一个泄漏安全的开关选择示例：

~~~powershell
python scripts/run_selection_protocol.py --config configs/solar.yaml --horizons 96 192 336 720 --seeds 5 --switch model.use_revin=true,false --switch model.channel_mixer_layers=0,1 --fixed model.use_revin=true,model.channel_mixer_layers=0
~~~

## 9. 测试

~~~powershell
pytest -q
pytest tests/test_model.py -k fusion
python -m compileall -q src scripts tests
~~~

测试结果受 Mamba 环境影响：

- 安装官方 mamba-ssm 后，MambaLayer 只按“能否 import”选择官方实现，不检查张量设备；CPU forward 会报 Expected x.is_cuda()；
- 未安装官方内核时，大部分 DD-Mamba 测试可走纯 PyTorch 回退，但 smamba 基线形状测试无法运行；
- 纯 PyTorch 的大规模 dispersion GPU 效率测试可能显存不足；
- 因此“pytest -q 必然是 CPU-only 且全绿”不是本仓库的有效假设。

如果验证损失出现非有限值，可先降低学习率或关闭 train.amp。早停对 NaN 的比较行为是有意设计，不应通过修改比较逻辑掩盖数值问题。

## 10. 已知边界与复现实验前检查

- src.evaluate.collect_predictions 当前只把 x 移到 device，没有把 stats 移动到同一设备。dispersion=fixed/learned 的 GPU checkpoint 独立重评可能触发设备不一致；默认 dispersion=none 不受影响。
- 训练阶段最终测试支持 train.noise_level，但 src.evaluate 不会重新施加该噪声。鲁棒性结果应由同一 runner 产出的结果 JSON 解释。
- weather 数据中若存在 -9999 等缺失哨兵，加载器不会自动替换，需在预处理阶段处理。
- HPO 的默认 test-MSE objective 属于复现实验协议，不应被描述成无泄漏调参。
- results/ 中的期望值、旧结果或标注 requires rerun 的配置不是新实验事实；准确性声明前应重新运行并验证 provenance。
- scripts/audit_submission.py 默认 fail-closed；允许 legacy 结果仅用于检查，不会使其变成可验证结果。
- data/、checkpoints/ 和 references/ 可能被忽略或不存在。运行前检查数据、上游仓库副本及外部适配器是否齐全。

## 11. 延伸文档

- documents/EXPERIMENTS.md：实验入口和结果约定
- documents/BOTTLENECK_ANALYSIS.md：瓶颈分析
- documents/DESIGN_LIMITATIONS_LITERATURE.md：设计限制与相关工作
- documents/PEMS_RESULTS_AND_ABLATION.md：PEMS 结果与消融
- documents/ELECTRICITY_CONFIG_DECISION.md：Electricity 配置决策
- assets/optuna.md：Optuna 工作流
- efficiency_experiment/EFFICIENCY_EXPERIMENT.md：独立效率实验协议

## 12. 引用与许可

仓库当前未提供 LICENSE 或 CITATION.cff。公开分发、复用代码或正式引用前，请由项目维护者补充许可证、作者信息和推荐引用格式。
