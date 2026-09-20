# 论文库全量阅读与 DD-Mamba 审计报告

> 审计目标：从本仓库论文中提炼时序预测论文的写作与实验范式，并以 Neurocomputing 审稿人的视角审计当前稿件 **DD-Mamba: A Forecast-Decomposable Dual-Domain State-Space Framework for Multivariate Time-Series Forecasting**。
>
> 审计日期：2026-09-20  
> 仓库快照：`b1e3e3438ff2e333cfecec7b844f75aadbf1d865`  
> DD-Mamba PDF SHA-256：`79559d42b7f6fdb5ea3211abf29584962629321f4f015a88991e8faab6e2726a`

## 1. 结论先行

当前稿件的研究问题、整体结构和写作质量已经达到可送审水平，但**证据链尚不足以支撑接收**。建议在投稿前完成一次 Major Revision；若按现状直接投稿，模拟决定倾向于 **Major Revision / Reject and Resubmit**，而不是 Accept。

决定性问题不是“采用了传统时序预测结果表”。本仓库样本显示，描述性的均值排名、复用已发表基线结果、主表不报告显著性检验，都是常见做法。DD-Mamba 的主结果表只要明确标注数据来源并限制为描述性比较，就是可以审计的。

真正的高优先级问题有三个：

1. 论文额外声称“四个差距经 Benjamini-Hochberg 校正后仍显著”，但正文没有给出可复核该结论的 inferential table、配对差值、`p`/`q` 值、完整检验族定义和种子级输入。
2. 核心创新是“两条完整预测路径 + prediction-level convex gate”，但现有消融没有证明 late fusion 相对 early feature fusion 的优势，且 mean fusion、移除频域分支等变体在若干数据集上持平或更优。
3. 高维数据上的双向 variate-Mamba 对变量顺序敏感，而论文没有 channel-order stress test；仓库中的同刊 S-Mamba 论文恰好做了这类测试。

只补一张统计表还不够。优先级最高的是补强**核心机制的直接证据**，然后再决定采用“传统描述性路线”还是“正式推断路线”。

## 2. 阅读范围与证据边界

### 2.1 全量处理情况

- 仓库包含 5 个 ZIP、37 个 PDF，共 503 页。
- 37/37 个 PDF 均通过结构完整性检查，所有页面均完成全文文本抽取和章节索引。
- 对每篇论文均检查了首页身份、摘要/引言、方法、实验设置、主结果、消融或分析章节、结论；对 DD-Mamba 的 14 页成稿进行了逐页视觉检查。
- 另对 S-Mamba、PeriodPatch、DFIMformer 的首页、方法页、主结果页、消融页和结论页进行了渲染检查，用于比较版式与图表表达。
- 这里的“全量读取”指所有文件和页面均被机器读取、索引并纳入检索；写作与实验范式的人工对照聚焦于 27 篇内容与文件名一致的论文。10 篇错配文件不用于推导时序预测领域惯例。

### 2.2 语料代表性限制

- `INS.zip` 中的论文实际来自 **Information Systems**，不是 Information Sciences。
- 仓库中只有 1 篇明确的 **Neurocomputing** 论文：*Is Mamba Effective for Time Series Forecasting?*
- `NeuNet.zip` 中的 5 篇论文来自 **Neural Networks**。
- 因此，本报告可以可靠提炼“时序预测/相关 AI 论文的常见写法”，但不能把该仓库表述为大规模 Neurocomputing 接收论文样本。

### 2.3 全文特征统计

以下计数基于 27 篇内容匹配论文的全文关键词索引，并对统计检验用途做了人工复核；它们表示“是否出现”，不是论文质量评分。

| 项目 | 论文数 | 比例 | 解释 |
|---|---:|---:|---|
| 明确描述数据集 | 27/27 | 100.0% | 数据集规模、采样频率或任务设置几乎是必备项 |
| 明确描述基线 | 25/27 | 92.6% | 两篇例外含综述/非标准预测任务 |
| 出现 ablation | 20/27 | 74.1% | 模块型论文通常需要组件级证据 |
| 明确报告多次运行或不同 seed | 8/27 | 29.6% | 并非所有已发表论文都报告随机性；部分放在附录 |
| 对模型排名做显式 inferential test | 2/27 | 7.4% | Hformer 使用 Friedman/Wilcoxon；ODE-RGRU 使用 Wilcoxon/rank |

这个结果很重要：**传统时序预测论文普遍采用描述性排名，而不是统计推断。** 因而，不能因为主结果表没有 `p` 值就判定论文不可接受；但一旦摘要和结论主动使用“BH-corrected significant”这类推断性表述，作者就承担了更高的报告义务。

## 3. 从论文库提炼出的写作方法

### 3.1 摘要：五段逻辑压缩成一段

高质量摘要通常依次完成：

1. 说明任务价值和应用场景；
2. 指出现有方法的具体机制缺口；
3. 用一句话给出方法总览；
4. 列出 2-3 个真正有区别度的模块；
5. 用可核查的实验范围和结果收尾，并避免把局部胜出写成普适 SOTA。

DD-Mamba 的摘要基本符合这一结构。需要修改的是最后一句：如果不补齐推断证据，应删除“四个差距经 BH 校正后仍显著”，保留“五个数据集上的描述性最低平均 MSE”即可。

### 3.2 引言：从任务矛盾收束到单一研究问题

仓库中较成熟的引言通常采用以下漏斗：

1. 应用需求；
2. 方法演进；
3. 现有方法仍未解决的单一矛盾；
4. 设计直觉；
5. 方法概览；
6. 3 条左右贡献。

DD-Mamba 的“latent entanglement → forecast-level decomposition”主线清楚，是稿件最强的写作部分之一。贡献列表也较克制，没有直接宣称所有组件都普遍提升精度。建议保持这条主线，不要再增加过多模块型贡献。

### 3.3 方法：问题定义—总览—模块—输出语义

主流写法是先定义输入输出张量，再给总图，然后按数据流顺序解释模块，最后给损失、复杂度或可解释量。DD-Mamba 已经做到：

- 明确给出 `L × C → H × C` 的问题定义；
- 先总览 temporal/frequency 两条路径；
- 给出 DLinear anchor、zero-initialized correction、complex-linear encoder、variate mixer 和 gate；
- 明确最终预测是两条完整预测的凸组合。

需要收紧的术语是 **phase-preserving**。任意复线性矩阵会改变各频率分量的相位；当前公式能说明的是“不丢弃复数相位信息/保留实部-虚部耦合”，不能证明“相位保持不变”。除非增加数学约束和证明，建议改为 `phase-aware complex-linear encoder` 或 `complex-valued encoder retaining phase information`。

### 3.4 实验：证据阶梯，而不是表格堆叠

成熟论文通常按下列顺序组织：

1. 数据集、切分、指标和硬件；
2. 基线与公平性；
3. 主结果；
4. 核心组件消融；
5. 参数/鲁棒性/效率分析；
6. 可视化与失败案例；
7. 结论边界。

DD-Mamba 覆盖了主结果、核心消融、累计 step-down、噪声鲁棒性、容量分析、案例研究和效率分析，广度充分。当前问题不是“实验少”，而是**最关键的实验没有正面击中核心主张**。

## 4. DD-Mamba 做得好的地方

| 维度 | 审计判断 | 依据 |
|---|---|---|
| 研究问题 | Meets | 将 dual-domain 的问题从 feature fusion 改写为 forecast semantics，可理解且有价值 |
| 方法表达 | Meets | 公式、张量语义、数据流和模块边界总体清楚 |
| 实验广度 | Meets | 13 个数据集、11 类基线，并有消融、效率、鲁棒性、容量和案例分析 |
| 结果克制 | Mostly meets | 多处主动说明 descriptive、single-seed 和不能作因果解释 |
| 混合来源披露 | Meets | 表注区分已发表数值和本地 matched rerun，正文也限制为描述性比较 |
| 负结果披露 | Meets | 明确承认 frequency/gate 并非在所有数据集上获益，效率也非普遍占优 |
| 讨论与局限 | Meets | 对 tuning、provenance、case study、硬件范围等限制披露充分 |

这些优点应保留。尤其不要为了“看起来更强”而删除负结果；应当通过更直接的实验解释为什么仍需要 forecast-decomposable 设计。

## 5. 必须修改的问题

### P0-1：BH 显著性结论无法从当前稿件复核

出现位置：

- `main.tex` Abstract：四个差距经 BH 校正后仍显著；
- `5_Experiments.tex` Main Results：再次声称 4 个差距通过 BH；
- `7_Conclusion.tex` Conclusion：第三次重复该结论；
- `7_Conclusion.tex` Threats to validity：提到一个“compact inferential table”，但当前源码和成稿中没有这张表。

当前缺失：

- 六个 matched dataset 分别对应哪个 comparator；
- 检验单位是每个 seed 的 horizon-average、逐 horizon，还是其他聚合；
- 完整 comparison family 到底是 6 个比较、`6 × 7` 个比较，还是另一集合；
- 每个比较的 paired difference、标准误/置信区间、原始 `p` 和 BH-adjusted `q`；
- 五个 seed 的配对输入及 seed ID；
- 可重放的配置、代码 commit 和日志。

`n=5` 时，双侧 exact sign-flip test 的最小可达 `p` 为 `2/2^5 = 0.0625`；`n=6` 时为 `0.03125`。这不意味着 paired t-test “非法”，但说明其结论高度依赖正态差值假设，且无法由 exact sensitivity analysis 支持。论文自己已经承认 exact corrected test 不通过，因此摘要中的强显著性表述会成为审稿焦点。

可选修复路线：

**路线 A：采用传统描述性范式（成本最低）**

- 删除摘要、Main Results 和 Conclusion 中的 “four margins remain significant after BH correction”；
- 保留均值排名，并明确 `descriptive comparison`；
- 自有模型至少报告 mean ± std；
- 将现有小样本参数检验放入补充材料，并标为 exploratory/sensitivity analysis。

**路线 B：保留正式推断（证据更强）**

- 预先固定 matched datasets、comparators、seed、horizon aggregation 和 comparison family；
- 建议至少 10 个 matched seeds；
- 增加表格：`dataset | comparator | DD-Mamba mean | baseline mean | paired Δ | 95% CI | p | BH-q`；
- 补充完整 seed-level 表和配置快照；
- 同时报告 parametric 与 exact/permutation sensitivity，不只报告通过的检验。

### P0-2：核心创新没有被直接隔离验证

论文的主要新意不是 DLinear、FFT、Mamba 或 sigmoid gate 本身，而是**两个具备完整预测语义的分支在 prediction level 才融合**。现有实验没有回答最关键的问题：

> 在相同参数量和训练预算下，prediction-level late fusion 是否优于 early/intermediate feature fusion？

当前证据的局限：

- mean fusion 在部分 ETTh2/Weather 条件下与 learned gate 持平或更好；
- 移除 FD branch 或替换 FD encoder 在部分数据集/预测长度上持平或更好；
- 累计 step-down 的 `Mean fusion` 不是 `Early fusion`，不能作为 early-vs-late fusion 证据；
- `7_Conclusion.tex` 将 `Mean fusion` 写成 “Early fusion”，属于事实性表述错误；
- 没有参数量匹配的 early-fusion 对照；
- 没有展示两条 branch forecast 和 gate 的实际行为。

必须增加的最小实验：

1. full late gate；
2. prediction-level mean；
3. time-only；
4. frequency-only；
5. parameter-matched early feature fusion；
6. parameter-matched intermediate fusion。

建议至少覆盖 ETTh2、Weather、Solar-Energy 和 ECL，以代表低通道/高通道、强周期/弱周期和有/无 mixer 的情形。应报告 10 个 matched seeds 或至少足以支持预先声明的推断方案。

### P0-3：缺少 gate 与 branch 的行为证据

“forecast-decomposable”不仅是结构命名，还暗示每条预测路径和融合权重具有可解释语义。当前论文没有展示：

- 不同数据集/变量/样本的 gate 分布；
- gate 是否饱和到接近 0 或 1；
- gate 随 horizon 是否应变化，而当前设计对每个变量使用跨 horizon 共享的权重；
- gate 权重是否与两个分支的局部误差、周期强度或频谱能量有关；
- temporal/frequency 两个 branch forecast 是否真正形成互补误差。

建议增加：

- gate 的 dataset/variable 分布、熵和饱和率；
- `|e_time| - |e_freq|` 与 gate 的相关/校准图；
- 两分支误差相关性与 oracle-gate 上界；
- 典型成功/失败窗口中同时画出 `Y_time`、`Y_freq`、gate 和最终预测。

若 gate 没有表现出可解释的选择行为，应把贡献收缩为“prediction-level convex ensemble”，不要使用过强的机制解释。

### P0-4：variate-Mamba 的顺序敏感性尚未解决

变量轴没有自然序，而双向 Mamba 仍然不是 permutation-invariant。当前论文在 Threats to validity 中承认这一点，但没有实验。

同仓库的 Neurocomputing S-Mamba 论文专门重排 periodic/aperiodic variates，并比较重排前后表现。DD-Mamba 以 variate mixer 作为高通道配置的重要组成，更需要进行测试。

建议区分两类实验：

1. **推理时置换**：固定已训练模型，同时置换输入/输出通道，检查架构是否依赖绝对顺序；
2. **重新训练置换**：用 5-10 个固定随机通道排列分别训练，报告均值和方差；
3. **结构化置换**：将高周期/低周期或高相关/低相关变量放在前、中、后位置。

如果顺序显著影响结果，应引入 permutation-equivariant mixer、集合式聚合、排序规则，或将该问题降为明确的适用性限制。

### P0-5：稿件描述的是“数据集特定配方族”，但缺少完整可复现配方表

当前配置会随数据集改变：

- temporal correction 在低通道数据上用 causal Mamba，高通道数据上用 MLP；
- FITS-style anchor 只在 ETTh1、ETTh2、Weather 启用；
- variate mixer 有时关闭、有时分支独立、有时跨分支共享；
- RevIN 在 Solar-Energy 上关闭；
- `d`、batch size、learning rate、weight decay、epoch 只给范围。

这不必然是缺陷，但它意味着论文评估的是一个 validation-selected recipe family，而不是单一固定模型。必须增加逐数据集配置表，并说明：

- 候选搜索空间；
- 选择准则；
- 每个数据集最终配置；
- 是否对 baseline 提供了可比 tuning budget；
- test set 是否完全没有参与架构选择；
- recipe 选择规则能否迁移到新数据集。

最好增加“统一配置 vs validation-selected 配方”的对照，以区分架构贡献与调参收益。

## 6. 重要但可在一轮修订中解决的问题

### P1-1：核心消融表只报告均值

`table_core_ablation_main.tex` 声称 6 个 matched seeds，但只给 MSE/MAE 均值；正文却进一步说存在 parametric evidence。至少应为预先指定的核心 contrast 报告：

- mean ± std；
- paired difference；
- 95% CI；
- 原始 `p` 与多重校正后的 `q`；
- exact sensitivity 结果。

无需给每个单元格都做检验。应预先限定少量 confirmatory contrasts，其余保留 descriptive。

### P1-2：效率实验的测量协议不够完整

已有优点是统一硬件、统一 batch、报告 optimizer-loop time 与 peak GPU memory，并承认 DD-Mamba 非普遍高效。仍需补充：

- 是否使用 AMP、TF32、`torch.compile`；
- CUDA warm-up 次数和计时重复次数；
- 计时前后是否 `torch.cuda.synchronize()`；
- dataloader worker、pin memory 和预取设置；
- 参数量、FLOPs/MACs、推理 latency、throughput；
- time/memory 的标准差或分位数。

### P1-3：baseline currency 与模块化研究背景不足

`refs.bib` 已经包含 TimeRecipe 和 CombinationTS，但 Related Work 没有引用它们。两者与本文“模块级、配方级、可拆解验证”的定位直接相关，应明确说明区别。

仓库中的较新论文还包括 PeriodPatch、Repetitive Contrastive Learning for Mamba、DFIMformer。至少应在 Related Work 中讨论；若投稿时已经公开且任务/设置可比，建议选择 1-2 个最接近的方法做匹配复现，而不是无限扩展基线列表。

### P1-4：数据集清单内部不一致

`Illness` 出现在累计 step-down 表中，但未出现在 Experimental details 的 13 个数据集清单和 dataset characteristics 表中。应选择：

- 把 Illness 正式加入数据集介绍、切分和主实验范围；或
- 从累计表删除；或
- 明确标为 supplementary diagnostic dataset。

### P1-5：文中存在“表不存在/术语不对应”

- Threats to validity 引用了 “compact inferential table”，当前成稿中不存在；
- Discussion 中的 “Early fusion” 实际对应累计表的 “Mean fusion”；
- `phase-preserving` 与当前任意 complex-linear 映射不严格对应。

这三处会直接降低审稿人对其余细节的信任，应在投稿前做一次 claim-to-table 全文审计。

### P1-6：复现资产不足

当前 package 没有：

- 代码仓库链接；
- immutable commit；
- 完整 resolved configs；
- 全部 seed-level logs；
- 可重建主表的脚本/数据；
- 明确的数据下载和预处理说明。

`Data will be made available on request` 不适合这些公开 benchmark。建议改为：列出公开数据来源、预处理脚本、代码与结果工件链接；不能公开的内容要说明具体原因。

## 7. 投稿前的编辑与版式问题

### P1-7：投稿占位符尚未清理

- `First Author`、示例邮箱、示例单位仍存在；
- CRediT 使用 `[Author 1 full name]` 等占位符；
- Acknowledgements 仍是提示文本；
- 需要确认 AI assistance 声明符合投稿时 Elsevier 的最新要求。

### P2-1：架构图影响第一印象

当前 Figure 1 存在较大的空白、明显的 L 形边界/裁切感和偏小文字。对比仓库内近期 Elsevier 论文，常见架构图会让模块边界、数据流、张量方向和颜色含义在双栏打印下仍可读。建议重新裁切并进行灰度打印检查。

### P2-2：Discussion 过于像内部审计记录

完整披露限制是优点，但 `repository-native`、`legacy provenance schema`、`inferential family` 等内部流程语言较多。建议：

- 正文保留 3-5 个决定性限制；
- 将完整 provenance/audit 细节移入 Reproducibility Appendix；
- 结论集中回答“发现了什么、在哪些条件下成立、下一步验证什么”。

## 8. 推荐的最小新增实验矩阵

| 优先级 | 研究问题 | 最小对照 | 推荐数据集 | 输出 |
|---|---|---|---|---|
| 1 | late forecast fusion 是否必要 | late gate / mean / early / intermediate / two single branches | ETTh2, Weather, Solar, ECL | matched seeds、Δ、CI、p/q 或纯描述性均值±std |
| 2 | gate 是否真的选择互补分支 | branch errors、gate、oracle gate | 同上 | gate 分布、校准、饱和率、误差相关性 |
| 3 | variate mixer 是否依赖通道顺序 | 原始/随机/结构化排列 | Weather, Solar, ECL, PEMS04 | 重训和推理置换两套结果 |
| 4 | 配方选择贡献有多大 | uniform recipe vs selected recipe | 全部主数据集 | 配方表、选择轨迹、测试盲化说明 |
| 5 | 资源代价是否可接受 | 相同硬件/精度/batch 的 matched baselines | PEMS04, Weather, ETTm1 | train/infer time、memory、params、FLOPs、方差 |

如果资源有限，至少完成优先级 1-3。容量 staircase、额外 case study 或更多单 seed 图不能替代这三项。

## 9. 推荐的论文重构顺序

1. 先决定统计路线 A（描述性）或 B（正式推断）。
2. 完成 late-vs-early fusion、gate behavior 和 channel-order 三项核心实验。
3. 冻结逐数据集配置，并生成完整 recipe table。
4. 用同一数据源自动生成主表、消融表和 inferential appendix，消除手工不一致。
5. 修改摘要、Main Results、Discussion、Conclusion，使 claim strength 与最终证据一致。
6. 补齐代码、配置、seed logs 和数据说明。
7. 最后处理作者信息、声明、图形裁切和语言润色。

## 10. 模拟编辑决定

| 维度 | 判断 |
|---|---|
| Neurocomputing 主题契合度 | Meets |
| 问题重要性 | Meets |
| 方法新颖性 | Partly meets：late forecast fusion 有辨识度，但尚缺直接对照 |
| 技术表述 | Partly meets：整体清楚，`phase-preserving` 需收紧 |
| 实证充分性 | Does not yet meet：核心机制和顺序敏感性证据不足 |
| 统计报告 | Partly meets：描述性表可接受，BH 强结论不可由当前稿件复核 |
| 可复现性 | Does not yet meet：缺 exact configs、logs、commit 和代码入口 |
| 写作与版式 | Mostly meets：结构成熟，Figure 1 和内部审计式语言需调整 |

**当前建议：Major Revision / Reject and Resubmit。**

达到可接收状态的关键不是堆更多 benchmark，而是让三条主张形成闭环：

1. 为什么一定要在 forecast level 融合；
2. gate 实际学到了什么；
3. 无序变量轴上的 Mamba 是否稳健。

## 11. 仓库文件身份问题

以下 10 个 PDF 的实际首页标题与文件名不一致。它们已经完成结构检查和全文抽取，但未用于时序预测范式统计。

| 文件名 | 实际论文 |
|---|---|
| `paper1/CIKM-huang2019dsanet.pdf` | *Topological nodal lines and hybrid Weyl nodes in YCoC2* |
| `paper1/ICLR-chen2021tamp.pdf` | *Fast Convergence on Perfect Classification for Functional Data* |
| `paper1/ICLR-liu2021pyraformer.pdf` | *The role of leptonic CPV phases in cLFV observables* |
| `paper1/ICLR-zhang2023crossformer.pdf` | *6G Wireless Communications in 7-24 GHz Band: Opportunities, Techniques, and Challenges* |
| `paper1/ICML-chen2021z.pdf` | *A Framework to Counteract Suboptimal User-Behaviors in Exploratory Learning Environments: an Application to MOOCs* |
| `paper1/ICML-lan2022dstagnn.pdf` | *Large Scale Mask Optimization Via Convolutional Fourier Neural Operator and Litho-Guided Self Training* |
| `paper2/TMLR-das2023long.pdf` | *Elimination and Factorization* |
| `paper2/WWW-wang2020traffic.pdf` | *Sparse Hop Spanners for Unit Disk Graphs* |
| `paper3/IJCAI-ST-SHN.pdf` | *On the impact of non-local gravity on compact stars* |
| `paper3/KDD-pan2019urban.pdf` | *Learning Configuration Space Belief Model from Collision Checks for Motion Planning* |

建议后续更换为正确论文，并为每个 PDF 增加一个 manifest：`filename, actual_title, venue, year, DOI/arXiv, sha256`。否则文件名错配会污染后续文献综述和自动化审计。

## 12. 最终检查清单

- [ ] 选择描述性或正式推断路线，并统一摘要/正文/结论措辞
- [ ] 补 late-vs-early/intermediate fusion 对照
- [ ] 补 gate 与 branch behavior 分析
- [ ] 补 channel-order stress test
- [ ] 增加逐数据集 exact recipe table
- [ ] 修正 `Mean fusion` / `Early fusion` 术语错误
- [ ] 修正或降级 `phase-preserving` 表述
- [ ] 处理 Illness 数据集清单不一致
- [ ] 补 inferential table，或删除 BH 显著性强结论
- [ ] 补效率测量协议和离散性
- [ ] 引用并讨论 TimeRecipe、CombinationTS 及近期相近方法
- [ ] 发布代码、配置、seed logs、表格构建脚本和数据来源
- [ ] 清除作者、CRediT、Acknowledgements 占位符
- [ ] 重画/重裁 Figure 1
- [ ] 修复仓库中 10 个错配 PDF
