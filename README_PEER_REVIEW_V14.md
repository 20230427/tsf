# DD-Mamba v14 模拟同行评审报告

> **结论先行：当前版本不建议直接投稿或接收。** 本轮模拟编辑决定为 **Reject and Resubmit / Major Revision before submission**。论文有可发表潜力，但必须先补齐可复现证据、重新界定并验证核心创新、完善统计报告，并解决稿件内部不一致与投稿占位符问题。

## 1. 审阅对象与范围

- 仓库：<https://github.com/20230427/tsf>
- 审阅快照：提交 [`6e8340350d5b047c18374805e43d5d793e696546`](https://github.com/20230427/tsf/commit/6e8340350d5b047c18374805e43d5d793e696546)，提交时间 2026-09-21，说明为 `Update DD-Mamba manuscript result descriptions`
- 最新论文包：`ddmamba (14).zip`
- 主稿：`main.pdf`，14 页，题为 *DD-Mamba: A Forecast-Decomposable Dual-Domain State-Space Framework for Multivariate Time-Series Forecasting*
- 审阅材料：完整 PDF、LaTeX 源文件、表格、图和参考文献；并检查了仓库公开内容是否足以重建实验
- 目标期刊假设：Neurocomputing。主题与该刊关于机器学习、神经计算和相关应用的范围总体匹配，参见[期刊官方页面](https://shop.elsevier.com/journals/neurocomputing/0925-2312)。期刊匹配不是当前拒绝直接投稿的原因。

本报告采用五个独立角色进行模拟评审：编辑/领域综合、方法与统计、时序预测领域、信号处理与部署、反方论证。五个角色均由同一模型家族生成，因此结论可能存在相关盲点，不能替代真实期刊同行评审。

## 2. 总体评价

论文提出两条能够各自输出完整预测的时间域和频率域分支，再通过逐变量凸门控进行预测级融合。稿件的优点是结构清楚、公式和信息流较易追踪，覆盖 13 个常用数据集，也主动披露了负结果、混合来源基线、单种子诊断、变量顺序敏感性和资源成本。这种克制值得保留。

但目前最强证据只能支持：**这是一个具有竞争力、模块边界清楚的双路预测框架。** 现有实验尚不能充分支持更强的核心叙事，即两个分支具有可识别的时间/频率语义、门控权重能够可靠追踪贡献，或性能提升主要来自预测级融合本身。主要结果也无法仅凭当前公开材料独立复现和审计。

## 3. 值得保留的优点

1. **方法表达清晰。** temporal anchor/correction、complex-valued frequency mapping、可选 FITS-style anchor、variable-axis mixer 和 prediction-level gate 的信息流明确。
2. **证据边界较诚实。** 主表被明确称为描述性比较，没有把混合来源数值写成统计显著的全面领先；无显著组件对比和若干不利结果也被保留。
3. **实验覆盖面较广。** 13 个数据集、多预测长度，并包含组件、累计消融、鲁棒性、容量、案例和效率分析。
4. **资源代价有披露。** PEMS04、Weather 等任务上的训练时间和显存开销没有被隐藏。
5. **PDF 基本可读。** 14 页均可正常渲染，没有发现明显的文字截断、缺字或损坏页面。

## 4. 阻断发表的问题（P0）

### P0-1：主要结果目前不可独立复现或审计

Section 4.1 只给出 batch size、learning rate、weight decay 和 epoch 的跨任务范围，缺少足以重建 Tables 2–7 的精确信息：

- 每个数据集的 train/validation/test 时间边界、窗口生成规则，以及标准化器是否仅在训练集拟合；
- 每个 dataset–horizon 的完整 resolved configuration，包括窗口长度、频率保留率、宽度、层数、state size、dropout、frequency bins、RevIN/FITS/variable mixer 开关、patience 等；
- seed IDs、逐种子结果、checkpoint 选择规则和运行 provenance；
- 可执行代码、环境锁定、配置、原始结果或 checkpoints。

当前公开仓库主要是论文 PDF/源码和压缩包，Data availability 仍为 “available on request”。这不仅是开放材料不充分，也使审稿人无法排除 split、preprocessing 或 recipe selection 对结果的影响。

**最低验收标准：** 提供可归档的代码与环境、数据清单、全部解析后配置、固定 split、seed-level JSON/CSV 和一条命令重建每个主表；至少由干净环境完成一次独立复跑。

### P0-2：核心创新的边界没有与最接近工作充分区分

论文讨论了多种 dual-domain 方法，但未充分处理与本稿最接近的近期工作，也没有在统一协议下比较它们：

- [Learning to route in time and frequency domains for multivariate time series forecasting](https://doi.org/10.1038/s41598-026-50232-8) 同样使用并行时间/频率预测及自适应加权；这直接压缩了“forecast-level fusion”的新颖性空间。
- [CDTF-Mamba](https://doi.org/10.1016/j.knosys.2026.115341) 是时间—频率双流 Mamba，并在相近的大规模基准上评估。
- [DTAF](https://arxiv.org/abs/2511.08229) 采用双分支时间/频率建模；即使其融合仍发生在预测头之前，也应被用来精确定义本文的差异。
- [GPS-Mamba](https://doi.org/10.1016/j.eswa.2026.131373) 直接涉及 Mamba 变量扫描的排列问题，与本文的 variable mixer 限制相关。

此外，Related Work 对 DualNet 的归类需要复核；其 global/local adaptive-compensation 路径不应被宽泛地等同于 temporal/spectral dual-domain 设计。

**最低验收标准：** 增加“最接近工作—结构位置—监督方式—融合层级—计算复杂度”的对照表；在同一代码、split、lookback、调参预算和 seeds 下重跑最接近的可复现基线，或显著收窄创新声明。

### P0-3：“forecast-decomposable”只被结构定义，尚未被行为证据验证

两个分支由最终 fused MSE 联合训练，没有 branch-specific supervision、贡献识别约束或校准机制。能够观察分支输出和 gate，并不意味着分支语义可识别，也不意味着 gate 表示可信的时间域/频率域贡献。当前缺少：

- 两个单独分支的误差、校准和互补性；
- gate 在 seeds、horizons、变量、季节和异常区间上的分布与稳定性；
- gate 与可验证频谱/时域结构或相对分支误差之间的关系；
- 相同容量的 early feature fusion、late fixed fusion 和普通 ensemble 对照；
- 对 branch/gate 不可识别性的理论或经验讨论。

Table 5 中 mean fusion、移除分支或替换编码器的差异较小且方向混合，没有组件对比在 exact corrected sensitivity analysis 后保持显著。这更适合支持“architecturally inspectable outputs”，尚不足以支持“可解释或可追踪的贡献”。

**最低验收标准：** 增加同容量融合基线、分支级诊断和 gate 稳定性分析；如果仍无稳定证据，将“可解释贡献”降格为“结构上可观察的预测分支”。

### P0-4：统计证据不足以支撑微小优势和机制归因

主表将本地三种子 DD-Mamba 均值与其他论文的单点数值混合。虽然正文已称其为描述性比较，但若干优势很小，例如 Electricity 平均 MSE 约为 0.169 对 0.170，ETTh1 约为 0.439 对 0.440；在没有相同实现、相同调参预算和不确定性区间时，不能解释为可靠领先。

组件实验和效率实验也只给均值，缺少 SD/CI、effect size 和逐种子值。稿件提及 paired t-tests、BH correction 和 exact sign-flip tests，却没有给出统计量、自由度、原始/校正后 p 值、校正 family 和完整 exact-test 结果。

**最低验收标准：** 发布逐种子结果；报告 mean±SD、95% CI、配对效应量和完整多重比较过程；将“描述性排名”与“确认性比较”分表或清晰分层。

## 5. 主要修订问题（P1）

### P1-1：模型/组件选择协议存在表述冲突

实验部分称架构设置由 validation set 选择，但 threats/future work 又把“validation-only component and hyperparameter selection”写成未来需要完成的事项。必须说明现有 recipe 是否完全未查看 test 结果，并提供选择日志或不可变配置。这里不能在无证据时直接判定 test leakage，但当前表述会使读者无法排除它。

### P1-2：多处分析或结论没有对应证据

- Eq. (12) 定义了 mean absolute pairwise Pearson correlation，但正文、表格和图中没有报告结果。
- Discussion 使用 channel correlation 和 time-varying scale 描述数据集，而 Table 1 并无这些统计量。
- Discussion 提到 ECL 上“neutral FITS-style result”，但方法称 FITS 只用于 ETTh1、ETTh2 和 Weather；Table 6 的 ECL “No freq. enh.” 因而可能只是结构性 no-op。
- Discussion 提到“ineffective longer training schedule”，但稿件没有呈现训练时长实验。
- Conclusion 称框架 “compact”，但效率表显示部分任务的训练时间和显存最高；若仅指概念结构，应明确限定。

这些陈述需要补上证据，或删除/改写。

### P1-3：变量顺序敏感性没有实测

bidirectional Mamba 沿本质上无序的变量轴扫描，并不具备 permutation invariance。稿件已承认这一点，但它是高维 Electricity、Traffic、PEMS 配方中的关键组件，不能只留作限制。至少应进行多次随机重排、固定语义顺序与置换恢复测试，并报告性能和 gate 的变化。

### P1-4：鲁棒性和部署证据过窄

鲁棒性仅为 ETTh2、单种子、自身对照的 i.i.d. Gaussian 输入噪声。现实系统更常见的是连续缺失、sensor dropout、离群点、level/seasonal drift、通道重排和延迟观测。效率实验也没有 inference latency、throughput、energy、warm-up/CUDA synchronization、重复次数和方差。

建议至少在两个不同数据特征的任务上，用 matched baselines 和多 seeds 检查几种现实损坏；同时报告训练与推理资源。对于能源、交通或天气任务，可增加 peak/extreme-event 指标，而不仅是平均 MSE/MAE。

### P1-5：案例研究协议不透明

案例图包含 H=384，但主实验声明的预测长度是 96/192/336/720；该模型的配置、训练和来源没有说明。图注称窗口为 “selected”，却没有选择规则，而且只展示 sample 0。cubic-spline 插值虽声明不参与量化，但可能放大或平滑视觉差异。

建议预先定义窗口选择规则，展示原始离散点、对应误差和多个随机/代表性窗口，并解释 H=384 的来源或删除该面板。

### P1-6：术语和方法细节仍需收紧

- moving-average 后的余项未必具有周期性，除非有证据，否则宜称为 residual 而不是 seasonal component；
- 补齐 embedding、MLP/Mamba 层、矩阵形状、loss 和频率处理的可实现定义；
- 明确不同数据集启用不同组件后，论文提出的是“可按验证集配置的框架”，而不是无需选择即可迁移的单一统一模型。

## 6. 次要与投稿就绪问题（P2）

1. 首页仍有 `First Author`、`Second Author`、示例邮箱和通用 affiliation。
2. 主作者列表与 CRediT 占位作者数量不一致，Acknowledgements 仍为占位内容。
3. 补充代码与数据可用性声明；公共数据集应给出来源、版本和预处理脚本，不能只写 “available on request”。Elsevier 的通用[研究数据指南](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines)可作为整理依据。
4. 架构图内部留白较大、标签偏小；Tables 2–5 和 Figure 5 信息密度高；第 9 页存在较明显的浮动体留白。建议优化，但这些不是科学性阻断项。

## 7. 五角色评分与规则化编辑决定

评分含义：`pass` = 无实质问题；`warn` = 需要重要修订；`block` = 阻断当前接收。D1–D3 为 mandatory，D4 为 high priority。

| 审稿角色 | D1 方法严谨性 | D2 领域准确性 | D3 论证一致性 | D4 跨领域相关性 | D5 写作结构 |
|---|---|---|---|---|---|
| 编辑/领域综合 | warn | warn | pass | pass | warn |
| 方法与统计 | **block** | pass | pass | pass | warn |
| 时序预测领域 | warn | warn | warn | pass | warn |
| 信号处理/部署 | warn | pass | warn | warn | warn |
| 反方审稿 | warn | warn | warn | pass | warn |

冻结规则的结果：

- **F1 触发：** 任一 mandatory dimension 出现 `block`。方法与统计审稿人的 D1 为 `block`。
- **F2 触发：** 多数审稿人在至少两个 mandatory dimensions 上为 `warn` 或更差。
- **F3 未触发：** high-priority D4 没有 `block`。
- **F0 未触发：** mandatory dimensions 并非全部 `pass`。

`fired_conditions: [F1, F2]`

`editorial_decision=reject_or_major_revision`

这里的含义不是“研究方向没有价值”，而是**当前证据包达不到可接收状态**。若这是投稿前内审，建议先大修再投稿；若这是正式首轮审稿，建议 Reject and Resubmit，允许在完整补证后作为新稿评估。

## 8. 建议的修订顺序与验收清单

### 第一阶段：先解决可审计性

- [ ] 公开或归档代码、环境、数据 manifest、精确 splits 和全部 resolved configs。
- [ ] 发布 seed-level outputs、统计脚本和表格重建命令。
- [ ] 证明所有配方选择只使用训练/验证信息，说明 checkpoint selection。

### 第二阶段：重新建立创新性与核心机制证据

- [ ] 正面讨论并比较最接近的时间—频率预测/融合方法。
- [ ] 增加相同容量的 early-feature、fixed-late 和普通 ensemble 基线。
- [ ] 报告 branch errors、互补性、gate distributions 和跨 seed/horizon/regime 稳定性。
- [ ] 若无法证明语义可识别性，收窄“forecast-decomposable/traceable”措辞。

### 第三阶段：完成统计与稳健性证据

- [ ] 主张性能优势的比较全部采用 matched protocol 和多 seeds。
- [ ] 报告 SD/CI、effect size、完整 p 值与多重比较校正。
- [ ] 增加 channel permutation、missing blocks/outliers/drift 等测试。
- [ ] 补充 inference latency、throughput、测量方差和计时细节。

### 第四阶段：清理稿件和提交材料

- [ ] 修正 Eq. (12)、ECL/FITS、longer-training、selection-protocol 和 H=384 等内部不一致。
- [ ] 完成作者、单位、CRediT、Acknowledgements、code/data availability。
- [ ] 优化架构图、密集表格和浮动体版面。

## 9. 最终判断

**当前能否发表：不能。**

**未来能否发表：有可能。** 方法叙事清晰、结果覆盖广、限制披露较诚实，具备形成可发表论文的基础。但接收的前提不是文字润色，而是完成可复现材料、最接近工作对比、核心机制验证和完整统计报告。若这些 P0 项没有实质性新证据，仅修改措辞，不足以改变编辑决定。
