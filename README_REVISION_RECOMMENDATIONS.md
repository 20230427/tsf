# DD-Mamba 三项核心修订实施方案

> 文档性质：实验与论文修订建议（尚未执行、尚未验证）  
> 审计对象：仓库 `main` 分支提交 `4210121d1893628a6156da8fccc7c4c33479409d` 对应的 v14 论文  
> 生成日期：2026-09-20  
> 方法来源：academic-research-suite / experiment-agent（plan mode）

## 1. 修订目标与优先级

当前论文围绕 prediction-level late fusion 提出主要创新，但现有证据还不能回答三个关键问题：

1. 性能提升究竟来自“晚期融合位置”，还是来自参数量、分支容量或训练预算差异？
2. learned gate 是否真正利用了两条分支的互补性，还是基本保持初始化时的时间分支偏置？
3. 论文中的数据集特定配置能否从表格追溯到配置文件、运行记录、模型输出和最终表格单元格？

建议按以下顺序修订：

| 优先级 | 必须新增的证据 | 对论文主张的作用 |
|---|---|---|
| P0 | 参数匹配的 early / intermediate / late fusion 对照 | 直接检验核心架构主张 |
| P0 | gate、分支预测与误差互补性分析 | 解释 learned gate 是否必要、如何工作 |
| P0 | 完整 recipe 表与结果工件映射 | 使实验可复核、可复现、可审计 |
| P1 | horizon-dependent gate 探索实验 | 检验当前 horizon-shared gate 的表达限制 |
| P1 | 参数匹配之外的 FLOPs、显存和时延匹配 | 排除计算预算混杂因素 |

在这些实验完成前，不建议在摘要或结论中断言 late fusion “优于” early/intermediate fusion，也不建议将 learned gate 描述为已经被证实的“自适应互补性建模”。更稳妥的暂定表述是：模型采用可分解的 prediction-level fusion，并允许直接观测两个分支及其融合权重；其相对优势需要由新增对照验证。

---

## 2. 参数匹配的 early / intermediate / late fusion 对照

### 2.1 研究问题与预注册假设

**RQ1：** 在相同数据处理、训练预算和近似相同的可训练参数量下，late fusion 是否优于 early fusion 与 intermediate fusion？

建议在运行测试集实验前固定以下假设：

- **H1（融合位置）**：Late-Gate 的跨 horizon 平均 MSE 低于 Early-Fusion 和 Intermediate-Fusion。
- **H2（learned gate 必要性）**：Late-Gate 优于 Late-Mean；否则只能支持“双分支集成”，不能支持“学习式门控带来收益”。
- **H3（稳定性）**：上述差异在多个随机种子下方向稳定，且置信区间不依赖单一数据集或单一 horizon。
- **H4（成本约束）**：若 Late-Gate 的参数量或推理成本更高，其精度收益在同时报告资源差异后仍然成立。

### 2.2 对照模型定义

所有模型使用相同输入窗口、预测长度、归一化、数据划分、损失函数和训练协议。推荐至少比较以下六个模型：

| 变体 | 融合位置 | 定义 | 回答的问题 |
|---|---|---|---|
| T-only | 无 | 仅保留 temporal branch | 时间分支的独立能力 |
| F-only | 无 | 仅保留 frequency branch | 频率分支的独立能力 |
| Early-Fusion | 编码器前 | 将时间表示与经过投影、对齐后的频域表示拼接，再送入共享编码器与预测头 | 输入级联合建模是否足够 |
| Intermediate-Fusion | 编码器后、预测头前 | 两个分支分别编码，将同尺度 latent 表示拼接或线性融合，再使用共享预测头 | 表征级融合是否优于预测级融合 |
| Late-Mean | 两个预测头后 | `0.5 * Y_time + 0.5 * Y_freq` | 双分支集成本身的收益 |
| Late-Gate | 两个预测头后 | `g * Y_time + (1-g) * Y_freq` | learned gate 的增量价值 |

实现 Early-Fusion 时，应明确频域表示如何恢复或投影到与时间 token 可拼接的形状；实现 Intermediate-Fusion 时，应明确融合发生在哪一层、latent 的维度和预测头结构。不能只给出“early/intermediate”名称而不提供计算图或张量形状。

为避免只挑选有利的 early/intermediate 实现，建议在验证集上为两者使用同样大小的预先声明候选集，例如：

- Early-Fusion：`concat + linear projection`、`concat + two-layer MLP`；
- Intermediate-Fusion：`concat + linear projection`、`concat + two-layer MLP`；
- 每个模型的候选数、训练轮数与早停规则相同；
- 只按验证集指标选定一个最终 recipe，选定后再运行测试集；
- 将所有候选及其验证结果写入 selection ledger，不能只保留胜出配置。

### 2.3 参数与训练预算匹配规则

Early-Fusion、Intermediate-Fusion、Late-Mean 与 Late-Gate 四个融合位置主对照采用**有效可训练参数匹配**，而不是加入不参与计算的 dummy 参数。T-only 与 F-only 主要用于机制诊断，应保留各自自然规模并如实报告参数量；可在附录另做预算匹配的单分支版本，但不能与自然规模版本混名。

| 项目 | 固定内容 | 允许调整内容 | 验收阈值 |
|---|---|---|---|
| 数据 | split、窗口、horizon、采样与归一化 | 不允许 | 完全一致 |
| 优化 | loss、optimizer、scheduler、epoch 上限、patience、梯度裁剪 | 不允许 | 完全一致 |
| 随机性 | 各模型使用相同 seed 集合 | 不允许 | 成对运行 |
| 调参预算 | 候选配置数、每个候选的训练预算 | 不允许 | 完全一致 |
| 参数量 | 总 trainable parameters | 仅调整 fusion projection、共享编码器宽度或 bottleneck 宽度 | 相对 Late-Gate 在 ±1%；确实无法满足时放宽到 ±2%并解释 |
| 计算量 | MACs/FLOPs、峰值显存、训练与推理时延 | 不作为隐藏调节项 | 必须实测并报告；建议控制在 ±5%或提供成本—精度曲线 |

除参数匹配版本外，附录可再报告一个“结构保持版本”：尽量保持原分支宽度不变，仅移动融合点。两类结果应分开命名，防止把“同结构”和“同预算”混为一谈。

参数统计必须覆盖所有参与训练的模块，包括归一化层、投影层、FITS、variate mixer、gate 与 prediction head。推理时延应在同一设备、相同 batch size、相同 warm-up 和重复次数下测量，并报告均值与标准差。

### 2.4 数据集、horizon 与随机种子

推荐的确认性实验至少覆盖四类结构设置：

| 数据集 | 选择理由 | 建议 horizon |
|---|---|---|
| ETTh2 | 低变量数；当前 core ablation 中 Mean Fusion 多个 horizon 不差于 full model | 96、192、336、720 |
| Weather | 中等变量数、周期性较强，且启用特定频域/混合配置 | 96、192、336、720 |
| Solar | 高变量数、单独 mixer、RevIN 设置具有差异 | 96、192、336、720 |
| ECL | 高变量数、共享 mixer，可检验扩展性 | 96、192、336、720 |

若算力允许，Traffic 或 PEMS04 可作为第五个高维数据集。不能只在 late fusion 已领先的任务上增加对照。

- **推荐确认性设置：** 4 数据集 × 4 horizons × 6 变体 × 10 seeds，共 960 次训练，其中已有且满足相同协议的运行可复用。
- **预算受限的筛查设置：** 4 数据集 × `{96, 720}` × 6 变体 × 5 seeds，共 240 次训练；该版本只能作为描述性证据，不能替代完整确认实验。
- 若从筛查设置决定扩展哪些任务，选择规则必须预先写明，不能依据测试集输赢临时选择。

之所以推荐 10 个 seeds，是因为每个数据集上将四个 horizons 聚合后，可进行 seed-paired 检验；两侧 exact sign-flip test 在 10 个配对下的最小非零 `p` 值约为 `2/2^10 = 0.00195`。6 个 seeds 的最小值约为 `0.03125`，在多个比较校正后通常不足以支持强结论。

### 2.5 统计分析与报告

**主要终点：** 对每个数据集、每个 seed，先计算四个 horizons 的平均 MSE，再比较 Late-Gate 与 Early-Fusion、Intermediate-Fusion、Late-Mean。由于同一数据集内量纲一致，可以直接平均；不同数据集分别检验，不把同一数据集的四个 horizons 错当成独立重复。

建议报告：

1. 每个模型的 `mean ± std`，同时保留逐 seed 原始值；
2. 配对差值 `ΔMSE = MSE(Late-Gate) - MSE(control)`；负值代表 Late-Gate 更好；
3. 配对差值的 95% bootstrap CI；bootstrap 以 seed 为单位，不拆分高度重叠的测试窗口；
4. exact paired sign-flip/permutation test；
5. 对 4 个数据集 × 3 个预注册比较共 12 个检验进行 Benjamini–Hochberg 校正；
6. horizon 级结果作为次要/探索性结果，不能把单个显著 horizon 代替整体结论；
7. 同时报告 MSE、MAE、参数量、MACs/FLOPs、峰值显存和时延。

推荐主表：

| Dataset | Horizon | T-only | F-only | Early-Fusion | Intermediate-Fusion | Late-Mean | Late-Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<dataset>` | `<H>` | `<MSE/MAE>` | `<MSE/MAE>` | `<MSE/MAE>` | `<MSE/MAE>` | `<MSE/MAE>` | `<MSE/MAE>` |

推荐统计附表：

| Dataset | Contrast | ΔMSE | 95% CI | raw p | BH q | Param ratio | Latency ratio |
|---|---|---:|---|---:|---:|---:|---:|
| `<dataset>` | Late-Gate − Early | `<value>` | `<low, high>` | `<value>` | `<value>` | `<value>` | `<value>` |

### 2.6 结论边界

| 观察结果 | 可以支持的表述 | 不应继续使用的表述 |
|---|---|---|
| Late-Gate 稳定优于 Early、Intermediate 和 Mean，且预算相当 | late fusion 与 learned gate 在所测任务上具有增量价值 | “普遍最优”或超出数据集范围的因果结论 |
| Late-Gate 优于 Early/Intermediate，但与 Mean 接近 | 预测级双分支集成有效；gate 的额外价值有限 | “自适应门控是性能来源” |
| Late-Mean 优于或等于 Late-Gate | late fusion 架构可能有价值，但 learned gate 未被证明必要 | “动态门控显著提升精度” |
| Early/Intermediate 与 Late-Gate 相当 | 融合位置不是当前性能的决定性因素 | “late fusion 优于特征融合” |
| 优势只出现在个别数据集/horizon | 数据集依赖的收益 | 无条件的总体优势 |

---

## 3. Gate、分支预测与误差互补性分析

### 3.1 必须保存的逐样本输出

当前模型的 gate 是样本级、变量级权重，并在预测 horizon 上共享。对每个 Late-Gate 测试运行，至少保存以下字段：

| 字段 | 含义 |
|---|---|
| `experiment_id`、`seed`、`checkpoint_hash` | 将预测绑定到唯一运行与模型 |
| `dataset`、`split`、`horizon`、`sample_id` | 定位测试样本 |
| `variable_id` | 定位变量 |
| `y_true` | 真实序列 |
| `y_time` | temporal branch 的独立预测 |
| `y_freq` | frequency branch 的独立预测 |
| `y_final` | 融合后预测 |
| `gate` | 未扩展前的 `B × C` gate；若实现不同，记录实际形状 |
| `normalization_state` | 确认各输出是否已处在同一反归一化尺度 |

建议保存为压缩的 NPZ、Parquet 或 HDF5，并在 manifest 中记录文件哈希。禁止只保存聚合后的 MSE，因为那样无法复核 gate 行为和误差互补性。

### 3.2 Gate 分布

对每个数据集、horizon 和 seed 报告：

- `mean`、`std`、`min`、`max`、P05、P25、P50、P75、P95；
- 饱和比例：`Pr(g < 0.05)` 与 `Pr(g > 0.95)`；
- 二元熵：`-g log(g) - (1-g) log(1-g)`，用于衡量 gate 是否接近硬选择；
- 样本间方差、变量间方差，以及不同 seeds 间的一致性；
- gate 与输入统计量（趋势强度、谱熵、周期强度、变量尺度）的相关性；相关性分析应标为探索性。

推荐图形：

1. 每个数据集/horizon 的 gate violin/box plot；
2. 变量维度较高时，画按变量排序的平均 gate heatmap；
3. gate 熵与饱和率的汇总图；
4. 不同 seeds 的 gate 排名一致性图。

解释时要特别检查 gate 是否仍集中在初始化对应的时间分支权重约 0.90 附近。如果训练后仍高度集中，论文应说明 gate 主要保持了时间分支偏置，而不能据此声称发生了强烈的动态路由。

### 3.3 分支预测质量

在完全相同的测试样本和反归一化尺度上，分别计算：

- `MSE_time`、`MAE_time`；
- `MSE_freq`、`MAE_freq`；
- `MSE_mean`、`MAE_mean`；
- `MSE_gate`、`MAE_gate`；
- oracle branch selector 的误差，仅作为不可部署的上界并明确标注 `Oracle`。

Oracle 可以定义为每个 `sample × variable` 选择平均 horizon 误差更小的分支，以与当前 horizon-shared gate 的决策粒度保持一致。若再报告逐时间点 oracle，必须单独命名，因为它使用更细粒度的真实标签，不能与可部署模型直接比较。

推荐表格：

| Dataset | Horizon | Time | Frequency | Mean Fusion | Learned Gate | Oracle | Gate gain vs best branch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<dataset>` | `<H>` | `<MSE>` | `<MSE>` | `<MSE>` | `<MSE>` | `<MSE>` | `<value>` |

其中：

`Gate gain vs best branch = min(MSE_time, MSE_freq) - MSE_gate`

正值表示融合优于全局最佳单分支，负值表示 learned gate 没有超过最佳分支。

### 3.4 误差互补性与 gate 校准

建议计算以下指标：

1. **残差相关性**：时间分支与频率分支的 signed residual correlation 及 absolute-error correlation；相关性越高，互补性通常越弱。
2. **预测分歧度**：`D = mean(abs(Y_time - Y_freq))`；同时报告分歧较高样本上融合是否更有收益。
3. **分支优势量**：`A = abs(e_freq) - abs(e_time)`；`A > 0` 表示时间分支更准。若论文中的公式为 `Y_final = g*Y_time + (1-g)*Y_freq`，则合理 gate 应与 `A` 呈正相关。
4. **胜率**：`Pr(abs(e_time) < abs(e_freq))`，按数据集、变量和 horizon 汇总。
5. **互补上界**：best-branch 与 oracle selector 之间的差距；该差距大说明分支具有可利用的局部互补性。
6. **已利用互补性比例**：`(MSE_best_branch - MSE_gate) / (MSE_best_branch - MSE_oracle)`；仅在分母为正且足够大时报告。

Gate 校准图建议将 `g` 分为十个等频 bins，并在每个 bin 中画：

- 时间分支获胜概率；
- 平均分支优势量 `A`；
- 样本数量及 95% bootstrap CI。

如果 `g` 越大但时间分支并未更常获胜，则 gate 不能被解释为可靠的“分支选择概率”。此时应把它描述为混合系数，而不是概率或可校准置信度。

### 3.5 必做 sanity checks 与反事实检验

1. 从保存的 `y_time`、`y_freq` 和 `g` 重算 `y_final`，检查最大绝对误差；建议 FP32 阈值 `1e-6`，混合精度阈值 `1e-4`。
2. 确认三种预测在计算误差前使用同一反归一化流程和同一 mask。
3. 打乱样本间或变量间 gate，比较原 gate 与 shuffled gate；若无明显差异，说明 gate 的样本/变量特异性贡献有限。
4. 固定 `g=0.5`、固定为训练集均值、使用 learned gate，三者必须在同一 checkpoint 输出上比较，以隔离 gate 本身的作用。
5. 检查 gate 方向与论文公式一致，避免 `g` 的解释与实际代码相反。
6. Oracle 只用于诊断，不得参与模型选择或测试集调参。

可选的 P1 实验是加入 horizon-dependent gate `g ∈ (0,1)^(H×C)`，并与当前 `g ∈ (0,1)^C` 进行参数匹配。若该变体显著更好，说明当前 horizon-shared gate 是表达瓶颈；若无改善，则支持更简单的共享设计。该实验应标为扩展分析，不应替代 early/intermediate 对照。

### 3.6 结果解释矩阵

| 诊断结果 | 更可能的解释 | 论文应如何修改 |
|---|---|---|
| gate 长期接近 0.9 且熵低 | 初始化偏置主导，模型接近时间分支 | 弱化动态路由主张，讨论初始化敏感性 |
| 两分支残差高度相关，oracle 改善很小 | 分支缺少实质互补性 | 将“双分支”描述为冗余增强，而非互补专家 |
| oracle 改善大，但 learned gate 改善小 | 分支有互补性，但 gate 未能识别 | 将问题定位为 gate 学习不足，增加校准或监督信号研究 |
| learned gate 优于两分支和 mean，且校准关系正确 | gate 成功利用局部互补性 | 可保留自适应融合主张，但限定在已测任务 |
| Mean Fusion 与 learned gate 相当或更好 | 主要收益来自集成而非门控 | 将 gate 从核心创新降为可选组件 |

---

## 4. 完整 recipe 表与结果工件映射

### 4.1 单一事实来源

建议将每个最终实验的解析后配置作为单一事实来源，并使用如下目录约定。以下是建议结构，不表示当前仓库已经存在这些文件：

```text
configs/
  final/<dataset>/<horizon>/<variant>.yaml
artifacts/
  <experiment_id>/
    manifest.json
    resolved_config.yaml
    metrics_seed_<seed>.json
    predictions_seed_<seed>.npz
    checkpoint_seed_<seed>.pt
    train_seed_<seed>.log
results/
  fusion_controls.csv
  gate_diagnostics.csv
  manuscript_cell_map.csv
```

`resolved_config.yaml` 必须是应用默认值、继承和命令行覆盖后的最终配置，而不只是原始 YAML。每个运行应使用唯一且稳定的 `experiment_id`，建议编码：

```text
<dataset>__H<horizon>__<variant>__seed<seed>__<config8>
```

### 4.2 论文必须提供的 recipe 表

主文可放数据集级结构配置，附录或补充材料提供完整的 dataset × horizon × variant 表。完整表至少包含：

| 类别 | 必填字段 |
|---|---|
| 数据 | dataset、数据版本/下载校验值、split 边界、输入长度、horizon、变量数、采样频率 |
| 预处理 | scaler/RevIN 开关与参数、缺失值策略、时间特征、detrending/moving-average window |
| Temporal branch | encoder 类型、层数、hidden/state size、kernel、dropout、prediction head |
| Frequency branch | FFT 保留比例或 bins、复数映射维度、FITS 开关及参数、频域 head |
| Mixer | 是否启用、separate/shared、层数、hidden/state size、插入位置 |
| Fusion | early/intermediate/late 类型、融合算子、projection/bottleneck 维度、gate bias、是否 horizon-shared |
| 训练 | batch size、optimizer、learning rate、weight decay、scheduler、epoch、patience、gradient clipping、AMP |
| 随机性 | seeds、deterministic 设置、重复次数 |
| 成本 | trainable parameters、MACs/FLOPs、峰值显存、训练时间、推理时延 |
| 追溯 | config path、config SHA-256、code commit、environment lockfile、experiment IDs |

建议主表采用以下格式，避免只写一个宽泛的超参数范围：

| Dataset | H | Variant | L | TD encoder | FD bins/ratio | FITS | Mixer/type | RevIN | Gate | Batch | LR | Params | Config hash |
|---|---:|---|---:|---|---|---|---|---|---|---:|---:|---:|---|
| `<dataset>` | `<H>` | `<variant>` | `<value>` | `<value>` | `<value>` | `<on/off>` | `<value>` | `<on/off>` | `<value>` | `<value>` | `<value>` | `<value>` | `<hash>` |

如果同一数据集的不同 horizon 使用不同配置，必须逐 horizon 展开；不能用单行“dataset recipe”掩盖差异。

### 4.3 Manifest 最低规范

每个 experiment manifest 至少记录：

```json
{
  "schema_version": "1.0",
  "experiment_id": "<dataset__H__variant__seed__config8>",
  "status": "success|failed|incomplete",
  "dataset": "<name>",
  "data_version": "<version-or-sha256>",
  "split_hash": "<sha256>",
  "horizon": "<integer>",
  "variant": "<name>",
  "seed": "<integer>",
  "git_commit": "<sha>",
  "config_path": "<path>",
  "config_sha256": "<sha256>",
  "environment_lock": "<path-and-sha256>",
  "checkpoint_path": "<path>",
  "checkpoint_sha256": "<sha256>",
  "metrics_path": "<path>",
  "predictions_path": "<path>",
  "command": "<exact-invocation>",
  "started_at_utc": "<timestamp>",
  "finished_at_utc": "<timestamp>"
}
```

上例中的尖括号是待运行后填写的占位符，不能原样作为“已完成记录”提交。

### 4.4 论文单元格到工件的映射

新增 `results/manuscript_cell_map.csv`，使每个表格单元格和图片都能追溯到原始运行：

| manuscript_item | cell_or_panel | metric | experiment_ids | seeds | aggregation_script | output_artifact | code_commit | config_hashes | verification |
|---|---|---|---|---|---|---|---|---|---|
| `Table_X` | `<row/column>` | `MSE` | `<IDs>` | `<seed list>` | `<path@sha>` | `<csv/json>` | `<sha>` | `<hashes>` | `verified/pending` |
| `Figure_Y` | `<panel>` | `gate distribution` | `<IDs>` | `<seed list>` | `<path@sha>` | `<svg/pdf>` | `<sha>` | `<hashes>` | `verified/pending` |

聚合程序必须自动检查：

- 预期 seed 是否完整，是否有重复运行；
- 比较模型是否使用完全相同的 seed 集合；
- 所有运行是否为 `success`，失败运行不得静默忽略；
- 配置哈希、代码提交和数据 split 是否一致；
- 表格中的均值和标准差能否由原始 JSON/NPZ 重算；
- LaTeX 表格是否由 CSV/JSON 自动生成，而不是手工复制四舍五入后的数字。

对于外部 baseline，还应增加 `provenance` 字段：`local_rerun` 或 `published`。若来自论文，记录论文版本、表号、行列位置和评价协议；若本地复现，按同样的 manifest 规范保存。两种来源不能混在同一列而不标注。

### 4.5 Recipe 选择记录

每个数据集/horizon/variant 应保留 selection ledger：

| candidate_id | search_space_version | validation_metric | validation_score | selected | rejection_reason | test_evaluated |
|---|---|---|---:|---|---|---|
| `<id>` | `<version>` | `<metric>` | `<value>` | `yes/no` | `<reason>` | `yes/no` |

最终配置只能依据验证集选择。ledger 应能证明未根据测试集结果回改 recipe。若历史实验无法满足这一要求，应明确标为 retrospective，而不是补写成预注册过程。

---

## 5. 推荐执行顺序

### 阶段 A：实现与单元检查

1. 实现 Early-Fusion、Intermediate-Fusion，并为全部变体增加统一的参数/计算量统计。
2. 为 Late-Gate 增加 `y_time`、`y_freq`、`y_final`、`gate` 导出。
3. 实现 resolved config、manifest、文件哈希和 experiment ID。
4. 用一个小 batch 验证张量形状、反归一化和融合重算误差。

**通过条件：** 预测重算一致；工件字段完整；所有模型参数统计覆盖相同模块边界。

### 阶段 B：小规模 pilot

在 ETTh2-H96 与 Weather-H96 上运行全部六个变体，每个 2 个 seeds。pilot 只用于排错和估算成本，不进入最终统计结论。

**通过条件：** 训练稳定；各变体参数差在阈值内；运行时间可控；gate 输出没有 NaN/shape 错误。

### 阶段 C：冻结协议

在完整测试前提交并冻结：候选配置集、选择规则、最终 seed 列表、主要终点、12 个主要比较和 BH 校正规则。记录冻结时的 code commit 与配置哈希。

### 阶段 D：确认性实验

完成 4 数据集 × 4 horizons × 6 变体 × 10 seeds；已有实验只有在代码、数据、配置和 seeds 与冻结协议一致时才能复用。Gate 分析复用 Late-Gate 的预测工件，不需要额外训练。

### 阶段 E：自动聚合与论文改写

1. 从 manifests 自动生成融合对照表、统计附表与 gate 图；
2. 随机抽查表格单元格能否回溯到逐 seed metrics；
3. 根据第 2.6 与第 3.6 节的结果矩阵调整主张；
4. 最后才更新摘要、贡献列表和结论，禁止在结果产生前预写有利结论。

---

## 6. 论文结构修改建议

| 位置 | 建议新增/修改内容 |
|---|---|
| Introduction | 将“early fusion 丢失结构、late fusion 更优”改为待验证假设；贡献中加入参数匹配对照和可审计诊断 |
| Method | 增加 Fusion Control Variants，小图标明 early/intermediate/late 的精确位置和张量形状 |
| Experimental Setup | 给出调参预算、参数匹配阈值、seed 数、主要终点和多重比较规则 |
| Results | 增加 Fusion Location under Matched Budgets；主表同时报告精度与成本 |
| Analysis | 增加 Branch Quality, Gate Distribution, and Error Complementarity |
| Reproducibility | 增加 recipe 表、manifest schema、结果工件链接和 manuscript cell map |
| Limitations | 说明 gate 在 horizon 上共享、数据集特定 recipe 的外部有效性，以及 oracle 只代表诊断上界 |
| Abstract/Conclusion | 严格依据新增对照结果决定是否保留“late fusion 优势”和“自适应互补性”表述 |

可使用以下不预设结果的写作模板：

> To isolate the effect of fusion location, we compare parameter-matched early-, intermediate-, and prediction-level fusion under identical data splits, optimization budgets, and random seeds. We further report the standalone branch predictions, gate distributions, and residual complementarity. All dataset–horizon recipes and manuscript results are linked to immutable experiment manifests.

完成实验后，再将占位性的中性描述替换为带有差值、置信区间、校正后 `q` 值和成本比的具体结论。

---

## 7. 提交前检查清单

- [ ] Early-Fusion 与 Intermediate-Fusion 有明确计算图、张量形状和实现说明。
- [ ] 六个对照使用相同 split、预处理、训练预算和 seed 集合。
- [ ] 参数差在 ±1%（或有理由的 ±2%）以内，并报告 FLOPs、显存和时延。
- [ ] 主要终点、主要比较和 BH 校正规则在查看最终测试结果前冻结。
- [ ] 保存逐样本 `y_true`、两分支预测、最终预测和 gate。
- [ ] 报告 gate 分布、熵、饱和率、校准和 seed 稳定性。
- [ ] 报告两分支误差相关性、胜率、oracle 上界与已利用互补性比例。
- [ ] 验证 `y_final` 能由分支输出与 gate 精确重算。
- [ ] 每个 dataset × horizon × variant 有完整 resolved recipe 与 config hash。
- [ ] 每个运行有 manifest、状态、代码提交、数据版本和工件哈希。
- [ ] 每个论文表格单元格和图片 panel 可映射到 experiment IDs 与聚合脚本。
- [ ] 外部 baseline 明确区分本地复现与论文转载。
- [ ] 最终论述遵循证据边界；若 Mean Fusion 或 early/intermediate 不差，则同步收缩核心创新主张。

## 8. 预期修订收益

这三组修改分别解决**因果归因、机制解释和结果追溯**问题：参数匹配对照决定 late fusion 是否真有独立贡献；gate 与分支诊断决定“自适应互补性”是否被数据支持；recipe—artifact 映射决定审稿人能否从论文数字追溯到唯一实验运行。三者应作为一个整体完成，单独增加更多 baseline 或更多平均结果不能替代这些证据。
