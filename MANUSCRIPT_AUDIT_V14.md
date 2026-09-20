# DD-Mamba 最新稿件审计报告（v14）

> 审计对象：`ddmamba (14).zip` 与 `DD-Mamba_revised_v10_descriptive_claims_cleanup.pdf`  
> 仓库提交：`629de0ba141203b7ab11fdd87b13994546c2892b`  
> PDF SHA-256：`d01705bf8e2218a7fad362dff9812971018fd236744b469de7db76a1f730d77f`  
> 审计日期：2026-09-20  
> 审计范围：论文正文、表格、Highlights、参考文献、PDF 版面及源文件包。按要求，本报告不讨论 Figure 1 的空白问题，也不重复列出作者、单位、邮箱、CRediT 和致谢占位符。

## 1. 审稿结论

以 Neurocomputing 时序预测审稿人的标准判断，当前版本较上一版明显更克制：强 BH 显著性结论已经删除，`phase-preserving` 已改成更准确的 `retains real--imaginary coupling`，`Early fusion` 也已改为 `prediction-level mean fusion`。这些修改是有效的。

但是，当前稿件仍不适合直接接收，建议为 **Major Revision**。决定性原因不是主表采用传统均值排名，而是论文中最醒目的五种子结论没有对应到可见表格，且核心创新“两个完整预测分支在预测层融合”尚未被直接隔离验证。若这些问题在外审中被发现，审稿人很可能质疑结果链条与主要贡献是否一致。

## 2. 阻碍接收的主要问题

### P0-1：五种子主结论与论文展示的主结果表不是同一条证据流

相关位置：

- `main.tex:70`：摘要声称六个 matched five-seed benchmarks 中五个最低或并列最低，PatchTST 在 ETTm1 更好；
- `highlights.tex:11`：再次声称 five-seed results 在六个数据集中五个有利；
- `5_Experiments.tex:18,51`：说明主表逐 horizon 结果为三种子均值，但正文结论来自五种子 matched summaries；
- `7_Conclusion.tex:23`：结论再次复述五胜一负；
- `tables/table_horizons_ett.tex` 与 `tables/table_horizons_general.tex`：表注明确说明基线主要来自已发表论文，而不是五种子 matched rerun 表。

当前仓库没有展示六个数据集的五种子 matched baseline 表，也没有给出对应的 seed-level 结果。因此，读者无法从当前稿件复核摘要中的核心数字结论。

更严重的是，可见主表与正文表述出现直接冲突：

| 数据集 | 当前主表 DD-Mamba Avg MSE | 当前主表 PatchTST/S-Mamba Avg MSE | 正文说法 |
|---|---:|---:|---|
| ETTm1 | 0.386 | PatchTST 0.387 | PatchTST 更好 |
| Weather | 0.247 | S-Mamba 0.251 | 视为描述性并列 |

主表中 DD-Mamba 在 ETTm1 的显示值优于 PatchTST，Weather 的显示值也不是数值并列。这并不证明隐藏的五种子结果错误，但证明当前稿件把两套结果混在了一起。

**必须修改：**

1. 增加独立的 matched five-seed 主结果表，明确六个数据集、比较对象、五个 seed、mean±std 和每个配置来源；或
2. 删除摘要、Highlights、Main Results 和 Conclusion 中的 five-seed 五胜一负结论，完全按照当前展示的混合来源主表进行描述；
3. 不应让“三种子/已发表基线表”和“五种子 matched rerun 结论”共用同一段排名叙述。

### P0-2：核心创新缺少 early/intermediate fusion 的直接对照

论文把主要新意定义为：时间域与频率域分别产生完整预测，只在 prediction level 通过 convex gate 融合。然而现有消融只比较 learned gate 与 prediction-level mean fusion，不能回答 late fusion 是否优于现有工作的 early/intermediate feature fusion。

当前结果还显示 gate 本身并非稳定获益：

- ETTh2 上 Mean fusion 的四个 horizon MSE 均低于 Full model；
- Solar-Energy 上 Mean fusion 在两个 horizon 更低；
- Weather 上两者多处相同或非常接近；
- 频域分支或时间域分支的移除在部分 ETTh2/Solar 设置中也未使均值变差。

这些负结果已经在正文中被诚实披露，但它们意味着论文目前只能证明“可以进行 prediction-level 分解”，还不能证明这种组织方式比特征级融合更有效。

**必须修改：**至少增加参数量和训练预算匹配的以下对照：

1. learned prediction-level gate；
2. prediction-level mean fusion；
3. early feature fusion；
4. intermediate feature fusion；
5. time-only 与 frequency-only。

如果暂时不能补实验，应将贡献收缩为一种可检查的架构组织方式，不再暗示 late fusion 已获得比较优势。

### P0-3：`forecast-decomposable` 的诊断证据仍然缺失

论文反复强调两个分支预测和 gate 具有显式语义，但没有展示：

- gate 在数据集、变量和样本上的分布；
- gate 是否长期饱和在接近 0 或 1；
- horizon 共享的单一变量权重是否限制远期预测；
- temporal/frequency 两个 branch forecast 的单独误差；
- 两分支误差是否互补，以及 gate 是否选择了局部更优的分支；
- oracle gate 相对当前 learned gate 的上界。

没有这些结果，“forecast-decomposable”目前主要是结构定义，而不是被实验验证的诊断能力。

**建议的最小证据：**gate 分布与饱和率、两分支单独 MSE、两分支误差相关性、`|e_time|-|e_freq|` 与 gate 的关系，以及若干成功/失败窗口中的两分支预测与最终输出。

### P0-4：论文实际上评估的是数据集特定 recipe family，但复现配方表被删除

当前模型会随数据集改变：

- temporal correction 使用 causal Mamba 或 temporal MLP；
- FITS-style anchor 仅在 ETTh1、ETTh2、Weather 开启；
- variate mixer 可关闭、分支独立或跨分支共享；
- RevIN 在 Solar-Energy 上关闭；
- moving-average width $w$、frequency-retention ratio $\rho$、latent width $d$、层数等未在当前稿件中给出最终取值。

`5_Experiments.tex:6` 只报告 batch size、learning rate、weight decay 和 epoch 的范围，并未给出逐数据集最终配置、候选搜索空间和选择准则。`REVISION_NOTES.md:11,27` 还明确记录逐数据集 recipe 表曾被加入，后来被删除。删除该表后，读者无法重建各结果对应的模型。

**必须修改：**在正文附录、补充材料或匿名代码仓库中恢复逐数据集配置矩阵，至少包含：TD encoder、FITS anchor、RevIN、mixer placement、共享方式、$w$、$\rho$、$d$、层数、batch size、learning rate、weight decay、epoch/early stopping、数据划分协议与配置选择规则。若正文不希望列出细节，至少必须给出稳定的配置文件链接和版本/commit。

### P0-5：多种子与探索性统计仍不可复核

当前版本已经正确地撤回 confirmatory significance claim，但仍写明使用 paired $t$ test、Benjamini--Hochberg correction 和 exact sign-flip sensitivity check（`5_Experiments.tex:18,75`），并在结论中称没有 contrast 通过 exact corrected analysis（`7_Conclusion.tex:23`）。稿件没有提供：

- 配对统计单位是 seed-level horizon average、逐 horizon 还是其他聚合；
- comparison family 的完整定义；
- 原始 $p$、BH-adjusted $q$、配对差值和置信区间；
- 六个 seed 的逐次结果；
- 表 5 和表 7 的标准差。

因此，描述性均值可以阅读，但探索性统计结论仍不能被独立核查。表 5 的许多差异只有 0.001--0.005，仅凭三位小数均值无法区分真实差异与 run-to-run variation。

**必须修改：**多种子结果至少提供 mean±std；对正文提到的统计检验另给补充表，列出 contrast、配对单位、seed、effect/paired difference、95% CI、$p$、$q$ 和 exact sensitivity 结果。若不准备公开这些结果，应删除统计检验的具体叙述，只保留 descriptive comparison。

### P0-6：variate-Mamba 的变量顺序风险仍未通过实验解决

`4_Method.tex:34,120--130` 将无自然顺序的变量轴输入双向 Mamba。双向扫描消除了单向因果方向，但没有产生 permutation invariance。稿件在 `7_Conclusion.tex:16` 已承认没有 channel-order stress test；对 ECL、Traffic 和 PEMS 等高维数据，这不是边缘问题，而是核心 mixer 的稳定性问题。

**必须修改：**至少在一个高维数据集上进行：

1. 固定模型的推理时通道置换；
2. 多个随机通道排列下重新训练；
3. 结构化排列（高/低相关或高/低周期变量位置交换）。

如果性能明显依赖排列，应给出固定排序规则、改用 permutation-equivariant mixer，或把适用边界写入贡献而不仅放在 limitations 中。

## 3. 重要的论证与实验问题

### P1-1：Discussion 中仍有三处没有对应证据的结论

`7_Conclusion.tex:11` 存在以下不受当前论文图表支持的表述：

1. **Channel correlation and time-varying scale**：实验节定义了平均绝对 Pearson correlation（`5_Experiments.tex:20--26`），但没有报告任何数据集的计算结果；`time-varying scale` 也没有定义或图表。
2. **Neutral FITS-style result on ECL**：方法节明确说 FITS-style anchor 仅用于 ETTh1、ETTh2、Weather（`4_Method.tex:116`）。累计表中 ECL 的 `No freq. enh.` 是结构性 no-op，不能解释为 ECL 上 FITS 结果为 neutral。
3. **Ineffective longer training schedule**：当前论文没有训练时长/调度对照表或图，无法支持这一结论。

**修改方式：**补上对应证据，或删除这些句子。尤其应删除“ECL 上 FITS neutral”的错误解释。

### P1-2：共享 mixer 的“parameter-efficient”贡献没有被量化

贡献 (3) 声称高通道数据集跨分支共享 variate-Mamba 是 parameter-efficient，但全文没有逐数据集参数量表，也没有 `shared mixer` 与 `two independent mixers` 的精度—参数—显存对照。当前效率实验显示 DD-Mamba 在 PEMS04 和 Weather 的峰值显存均约为相近精度方法的 8.8 倍，这使“parameter-efficient”更需要定量限定。

**建议：**报告 full/shared/separate/no-mixer 的参数量、FLOPs 或 MACs、训练时间、推理延迟、峰值显存和 MSE；若只表示“比两个独立 mixer 少参数”，应把贡献文字限定到这一局部比较。

### P1-3：效率实验协议不足以复现细小差异

`5_Experiments.tex:138--149` 报告 optimizer-loop time 和 peak allocated memory，但没有说明：

- 每个面板的统一 batch size；
- AMP/FP32、TF32、`torch.compile` 状态；
- CUDA warm-up 和正式计时迭代数；
- 是否在计时边界调用 `torch.cuda.synchronize()`；
- dataloader workers、pin memory 和预取设置；
- 时间与显存的标准差。

Weather 和 ETTm1 的精度差距很小，缺少波动范围时，图中的 Pareto 关系容易被过度解读。建议同时补 inference latency 和 throughput；如果不补，应继续把结论限制为该单一训练环境中的描述性测量。

### P1-4：近期直接相关方法只出现在 Related Work，没有进入主要数值比较

Related Work 已引用 DecMamba、TF4TF、Waveformer、TFKAN、TimeMachine、TCM 等更直接的近期方法，但主要结果表仍以 S-Mamba、iTransformer、PatchTST 和较早 Transformer/线性模型为主。TimeRecipe 与 CombinationTS 只是模块评估框架，也没有作为实验基线。

**建议：**至少选择两到三种与“dual-domain / decomposition / Mamba”最直接、协议可对齐的方法做 matched rerun，或明确解释无法比较的原因。否则“相对当前相关工作的增量”仍主要依赖结构叙述。

### P1-5：案例研究的可视化处理削弱证据价值

Figure 5 使用 sample 0 的四个 selected windows，但没有给出选择规则；两个面板使用 $H=384$，不属于正文标准 horizon 集合；曲线又经过 cubic-spline interpolation。即使插值不进入定量计算，它仍会改变视觉上的峰值、相位和贴合程度。

**建议：**使用原始离散预测点或普通折线，说明窗口选择规则；删除 $H=384$ 辅助面板或完整解释其训练与用途；给出每个窗口的数值误差。图例中的 `DDMamba` 也应统一为 `DD-Mamba`，不应在 caption 中用解释来保留不一致命名。

### P1-6：鲁棒性结果不能支持机制解释

Figure 3 仅在 ETTh2 上对 DD-Mamba 自身进行单种子高斯噪声测试。`5_Experiments.tex:104` 已承认没有 matched baseline 或 branch-level 分解，但前文仍把平滑退化与 decomposition anchor、temporal correction 和 spectral forecast 联系起来。该结果只能说明该 checkpoint 对这一种噪声的响应，不能说明为什么平滑退化，也不能证明双域结构提高鲁棒性。

**建议：**加入 time-only/frequency-only、无 anchor 和至少一个匹配基线；否则删除机制归因，只保留现象描述。

### P1-7：数据与代码可用性声明过弱

论文使用的主体数据集均为公开 benchmark，但 Data availability 仅写 `Data will be made available on request.`。同时，稿件把结果 provenance 不完整列为主要限制，却没有给出代码、配置、seed-level 工件或匿名仓库地址。

**建议：**区分公开原始数据、处理后的划分、训练代码、最终配置和结果工件；提供匿名仓库或补充材料，并将每张表映射到配置 hash、代码 commit 和 seed 列表。

## 4. 内部一致性与版面问题

### P2-1：`confirmatory` 与全文的 exploratory 定位冲突

`5_Experiments.tex:57` 称 ETTh2、Weather 和 Solar-Energy 为 `confirmatory ... recipes`，但 `5_Experiments.tex:18,75` 与 Discussion 又明确将统计分析定位为 exploratory，并说明 exact corrected analysis 不通过。建议改为 `evaluated`、`selected` 或 `controlled-ablation recipes`，避免重新引入确认性语义。

### P2-2：Figure 2 与 Table 7 的浮动位置破坏阅读顺序

训练效率 Figure 2 的源码环境位于 Component analysis 内部（`5_Experiments.tex:65--71`），而真正的 Matched training efficiency 小节到 `5_Experiments.tex:138` 才出现。最终 PDF 中 Figure 2 提前出现，Table 7 又漂移到 Discussion 已开始之后。建议把 Figure 2 环境移到效率小节，并在 Discussion 前增加适当的 float barrier。

### P2-3：部分图的可读性仍需提高

- Figure 2 在 PEMS04 和 Weather 面板中，模型文字、显存标注和气泡明显重叠；
- Figure 4(a) 的 lookback ticks 在短 lookback 区域过密；
- Figure 5 图例字号较小，且模型名不一致。

这些不影响数值正确性，但会降低双栏成稿的可读性。

### P2-4：关键词存在学科歧义

关键词 `Multivariate analysis` 通常指统计学中的多变量分析，不能准确表示本文任务。建议改为 `Multivariate time-series forecasting` 或 `Long-term time-series forecasting`。

### P2-5：参考文献与源文件包存在可清理项

- `refs.bib` 有 42 个条目，实际引用 39 个；未使用的 key 为 `lstnet`、`mamba2`、`rlinear`；
- `figures/sensitivity.pdf` 未被正文引用；
- `neurocom.bib` 是未使用的旧文献库；
- `AGENTS.md`、`REVISION_NOTES.md`、`assets/edition.md`、`assets/revision.txt`、期刊指南摘录等属于内部工作文件。

如果 `ddmamba (14).zip` 将作为投稿源文件包，上述内部记录、未使用资源和本地路径信息不应一并提交。建议另建干净的 submission ZIP，只保留可编译源码、被引用的图表、`refs.bib`、class/bst 和期刊要求材料。

## 5. 已修复且不应继续沿用的旧审稿意见

以下问题在 v14 中已经处理，不应在新一轮审稿中继续按旧版本表述：

- 摘要、Main Results 和 Conclusion 已删除“四个差距经 BH 校正显著”的强结论；
- `phase-preserving` 已改成 `retains real--imaginary coupling`；
- `Early fusion` 已改成准确的 `prediction-level mean fusion`；
- Illness 已明确为辅助诊断数据集，不计入主 benchmark；
- Dataset table 已补充 PEMS03/04/07/08；
- 稿件已经明确承认 mixed-source baseline、seed 数量不足、channel-order 风险和效率测量范围有限。

这些修订改善了诚实性，但“承认限制”不能替代对摘要主结论和核心创新的直接证据。

## 6. 建议的修订优先级

### 第一优先级：先修复结果链条

1. 展示真正的六数据集 matched five-seed 表，或删除全部 five-seed 五胜一负表述；
2. 统一 ETTm1 与 Weather 在摘要、Highlights、Main Results、表格和 Conclusion 中的结论；
3. 提供逐数据集最终 recipe 与工件映射；
4. 为所有多种子表补充离差，并公开探索性检验表。

### 第二优先级：证明核心结构确有必要

1. 增加 early/intermediate/late fusion 的参数匹配对照；
2. 增加 gate 与 branch forecast 行为分析；
3. 增加 shared/separate mixer 的准确率—资源对照；
4. 增加 channel-order stress test。

### 第三优先级：删除无证据文字并整理投稿包

1. 删除或补证据支持 correlation、time-varying scale、ECL FITS 和 longer training schedule；
2. 补全效率协议，修正 Figure 2/4/5；
3. 增加近期直接竞品，或收缩比较性结论；
4. 建立干净的投稿源文件包。

## 7. 编译与一致性检查结果

- LaTeX 全流程编译成功：`pdflatex → bibtex → pdflatex ×2`；
- 编译错误：0；
- undefined citations/references：0；
- overfull boxes：0；
- 最终 PDF：14 页；
- PDF 与 `ddmamba (14).zip` 内的 `main.pdf` 字节一致；
- 存在若干 underfull box 警告，但未观察到正文截断或表格越界。

## 8. 最终判断

当前稿件的结构、写作克制程度和负结果披露已经达到较好的投稿水平，但从时序预测审稿角度，**核心排名结论尚未与可见表格对齐，核心 late-fusion 主张尚未被直接验证，数据集特定 recipe 也尚未达到可复现要求**。因此建议仍为 **Major Revision，不建议当前版本直接接收**。

若只做文字修订而不补实验，最少也必须解决 P0-1、P0-4、P0-5 和 P1-1；若希望显著提高接收概率，还需要完成 P0-2、P0-3 与 P0-6。
