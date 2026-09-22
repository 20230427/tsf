# DD-Mamba 核心创新核查报告（对照 `references/ddmamba` 论文引用项目与常用 Benchmark）

> 本文件核查本仓库**实际实现**的 Core 创新点，对比对象为 **DD-Mamba 论文手稿（`references/ddmamba`）中提到的参考项目**与**常用 Benchmark 的基线/数据集/协议**。三类证据交叉验证：① 论文手稿（`2_RelatedWork.tex` 引用清单、`4_Method.tex` 方法声明、`tables/` 实验表）；② 本仓库 `src/models/` 源码；③ `configs/` 全部 14 个数据集 shipped 配置。"实际使用"以 shipped config 为准——论文/默认配置提供开关但未被任何 benchmark 配置启用的机制，不计入主结果证据。

---

## 1. 对比对象清单

### 1.1 Related Work 引用的参考项目（`2_RelatedWork.tex` + `refs.bib`）

| 类别 | 项目（bib key） | 论文对其定位 |
|---|---|---|
| 深度预测架构 | DeepAR (`deepar`)、TFT (`tft`) | 循环/自回归时序建模 |
| | Informer、Autoformer、FEDformer (`informer,autoformer,fedformer`) | 稀疏注意力 / 分解 / 频率增强注意力 |
| | PatchTST、iTransformer、TimesNet (`patchtst,itransformer,timesnet`) | patch token / 变量 token / 多周期表示 |
| 线性/轻量 | DLinear (`dlinear`)、TiDE (`tide`)、TimeMixer (`timemixer`)、TCM (`tcm`)（另有 RLinear、CombinationTS 入 bib） | 分解线性投影 / 残差密集编码 / 多尺度混合 / 时空分离 MLP |
| 频域感知与双域 | FreTS (`frets`)、FITS (`fits`) | 直接谱域线性学习 |
| | AMSFormer (`amsformer`)、DFCon (`dfcon`) | 频率引导的自适应多尺度划分 / 主频对比增强 |
| | **DualNet** (`dualnet`)、**TF4TF** (`tf4tf`)、**WaveFormer** (`waveformer`)、**TFKAN** (`tfkan`) | 双域网络：自适应补偿 / 时频多语义融合 / 小波处理 / 并行非线性映射 |
| | SCINet (`scinet`)、STD (`std`)、Autoformer | 分解类方法（趋势-季节-离散/多尺度分解后再预测） |
| 时空与 SSM | Crossformer、iTransformer、SOFTS、ModernTCN、DyTimeNet (`crossformer,softs,moderntcn,dytimenet`) | 变量段注意力 / star 聚合核心 / 大核卷积 / 动态稀疏跨变量依赖 |
| | Mamba / Mamba2 (`mamba,mamba2`) | 选择性状态空间扫描（线性复杂度） |
| | **S-Mamba** (`smamba`)、TimeMachine (`timemachine`)、DecMamba (`decmamba`)（另有 ms-Mamba 入 bib） | 变维双向 SSM / CI+CM 四 Mamba 多尺度 / SSM+趋势季节分解 |
| 归一化 | RevIN (`revin`) | 可逆实例归一化（方法章引用） |

**论文的核心差异化主张**（Related Work 末段）：前人方法把时序编码、周期表示、变量交互 **集中在单一骨干**或**在中间特征层耦合**（"introduce spectral information inside an intermediate encoder or fuse it with temporal features **before either pathway forms a prediction**"）；DD-Mamba 以"因果 SSM 时域路径 + 复数频域路径 + 可选双向变维 mixer + **预测级**门控融合两条完整预测"回应。

### 1.2 实验 Benchmark 基线（`tables/`）

| 实验表 | 对比基线 | 数值来源 |
|---|---|---|
| `table_horizons_ett.tex` / `table_horizons_general.tex` | S-Mamba、iTransformer、PatchTST、TiDE、DLinear、FEDformer、Autoformer 等 | 文献报告值（转载自 S-Mamba Tab.2 / iTransformer Appx. F.4） |
| `table_pems_traffic.tex` | S-Mamba、iTransformer、PatchTST、Crossformer、TiDE、DLinear、FEDformer、Autoformer | 同上（`table_pems_traffic.tex` caption 明示） |
| `table_training_efficiency.tex` | iTransformer、TimesNet、S-Mamba、PatchTST、DLinear | 同环境实测（"All baseline values are measured in the same experimental environment"） |

**Benchmark 协议**：`seq_len=96` 固一、horizons {96,192,336,720}（PEMS 沿用 S-Mamba 协议）、ETT canonical 12/4/4 月边界（`data.split_protocol: ETTh|ETTm`）、MSE/MAE、早期停止仅用 validation split——与 Autoformer/TimesNet/iTransformer/S-Mamba 通用协议对齐（README "Metrics" 节）。

### 1.3 Benchmark 数据集

| 论文报告（11 个） | 仓库 config（14 个） | 差异 |
|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2、Weather、Solar-Energy、ECL、Traffic、Exchange、PEMS03/04/07/08（`table_datasets.tex` + `table_pems_traffic.tex`） | 上述 11 个 + **Illness** | Illness 有 tuned config（`configs/illness.yaml`）但**论文数据表未报告** |

---

## 2. 核心创新 × 参考项目差异总表

| # | 创新（代码位置） | 最接近的参考项目及其做法 | DD-Mamba 的实际差异 | Benchmark 使用 | 证据状态 |
|---|---|---|---|---|---|
| 1 | **双分支各出完整预测 + 预测级逐变量门控凸融合**（`src/models/fusion.py`；论文 Eq.(10)） | 双域类：DualNet（双路径自适应补偿）、TF4TF（时频多语义）、WaveFormer（小波）、TFKAN（并行 KAN 映射）、FEDformer（频率增强注意力）——均在**特征/中间层**融合，无独立的双预测 | 每个分支输出自己的 (B,C,H) 预测；g=σ(MLP([feat_t;feat_f]))∈(0,1)^C 逐变量凸组合，无旁路路径；`return_components=True` 可导出分支预测（case-study Fig.5）与门控统计（Fig.4） | **全部 14 个 config / 论文全部 11 个数据集**（`fusion: gated` 无一例外） | ✅ 主结果承重 |
| 2 | **"线性锚 + 零初始化修正"跨分支联合初始化**（`time_branch.py:149-152`、`freq_branch.py:100-103,176-179`、`fusion.py:57-69`；论文 Eq.(5)(7)(9)） | DLinear/RLinear（纯线性）、TiDE（残差非线性映射）、TimeMixer/TCM（多尺度/分离混合）——单一投影空间、无域分工；FITS——纯频域线性 | 时域 = DLinear 式分解线性锚 + 零初始化修正头；频域 = 零初始化 FITS 式频谱插值锚（`auto` 头零初始化）；门控 bias=2.2→g≈0.9 起步、additive 模式 FREQ_INIT_WEIGHT=0.1 规避双零梯度消失。初始模型 ≈ 精确线性双域组合，深度部分只学修正 | 全部数据集（`zero_init: true` 默认无覆盖）；其中 FITS 频谱锚仅 **ETTh1（ρ=0.4）、ETTh2（ρ=0.3）、Weather（ρ=0）** | ✅ 主结果承重 |
| 3 | **跨分支权重共享变维 mixer**（`mixer_placement: shared`，`dual_domain_model.py:153-154`） | S-Mamba（单路径变维 BiMamba，无分支可共享）、TimeMachine（CI+CM 四 Mamba）、iTransformer（O(C²) 变维注意力）、SOFTS/ModernTCN/DyTimeNet | 一个 BiMamba mixer 权重绑定服务时/频两分支，参数约减半；traffic ΔMSE +0.0007（CI 含 0）、electricity ΔMSE −0.0001 验证精度中性（`results/Ablation_{traffic,electricity}_mixer.json`） | **ECL、Traffic、PEMS03/04/07/08**（6 个高通道 benchmark）；Weather 为 `both`（曾试 shared 后 revert）；Solar `both`（1 层） | ✅ 6 个数据集使用 |
| 4 | **alpha-RevIN**（`revin_alpha: fixed/learned/channel`，`dual_domain_model.py:178-203`） | RevIN——硬开/关二元开关 | 归一化强度 α∈[0,1] 连续化（sigmoid 参数化、可逐通道），训练集拟合以消除 Solar "RevIN off" 这类测试集消融选择偏差 | **无任何 benchmark config 启用**（默认 `fixed` 1.0 = 标准 RevIN；仅 `revin_alpha` 消融链） | ⚠️ 已实现、无 benchmark 证据 |
| 5 | **Dispersion head**（多分辨率统计→逐 horizon 尺度/位置反归一化，`dual_domain_model.py:205-211,329-338`） | STD（seasonal-trend-dispersion **分解**后预测，2023）——同为 dispersion 概念但用于分解，非反归一化尺度预测 | 预测正的逐 horizon 缩放因子替代固定 μ/σ 反归一化，初始化退化为 RevIN | **无 config 启用**（默认 `dispersion: none`，仅消融链） | ⚠️ 已实现、无 benchmark 证据 |
| 6 | 频域 BiMamba（bin 级双向扫描，`freq_branch.py:119-130`） | FreTS/FITS（谱域线性）；`freq_branch.py` docstring 自注承袭 "FMamba-style spectral SSMs" | 频率 bin 作为序列双向扫描（概念非原创，双向选择为自有） | **无 config 启用**（所有 benchmark `freq_encoder: linear`）；且论文 `4_Method.tex` 只描述 complex-linear encoder + 可选 FITS map，**方法章未主张该机制** | ⚠️ 仅 A/B 开关，论文亦未主张 |

## 3. 分组对比要点

### 3.1 vs 双域/频域方法（DualNet、TF4TF、WaveFormer、TFKAN、FreTS、FITS、AMSFormer、DFCon、FEDformer）
- **共同点**：都认同"周期结构在谱空间更紧凑"。
- **差异**：上述方法的频谱信息进入中间编码器或在**形成预测之前**的特征层融合（早期耦合），域特异性对最终预测的贡献是隐式的；DD-Mamba 时/频两分支**各自形成完整预测**后再做逐变量凸门融合（创新 #1），且两分支分别带 DLinear/FITS 线性锚 + 零初始化修正（创新 #2）。纯频域方法（FreTS/FITS）则完全缺时域路径。

### 3.2 vs SSM 预测器（S-Mamba、TimeMachine、DecMamba、ms-Mamba）
- **共同点**：变维 mixer 直接继承 S-Mamba 的双向变维扫描设计（论文 Eq.(9)，`mamba_block.py:BiMambaEncoder`）。
- **差异**：这些模型提供高效时序或变量交互，但显式谱建模走的是另一条架构路线；DD-Mamba 在同一模型内组合因果 SSM 时域路径 + 复数频域路径 + 预测级融合。创新 #3（跨分支权重共享 mixer）是 SSM 预测器中独有的参数效率设计。

### 3.3 vs 线性/轻量方法（DLinear、RLinear、TiDE、TimeMixer、TCM）
- DLinear/FITS 组件被**显式吸收**为双分支各自的线性锚（继承，见 §4）；TiDE/TimeMixer/TCM 的残差/多尺度混合不区分域。DD-Mamba 的差异是"线性锚打底、深度修正"的零初始化结构（创新 #2），而非更强的线性基线本身。

### 3.4 vs 深度架构/时空方法（PatchTST、iTransformer、TimesNet、Crossformer、SOFTS、ModernTCN、DyTimeNet）
- 变量交互机制对比：iTransformer/Crossformer 为注意力（O(C²)，高通道数据集昂贵）；SOFTS star 聚合、ModernTCN 大核卷积；DD-Mamba 用 S-Mamba 式变维 BiMamba（线性复杂度），并可在两分支间共享权重（创新 #3）。通道独立（mixer=0）退路用于少通道小数据集（ETT/Exchange），对应论文 "optional bidirectional variate mixer"。

## 4. 继承组件（非创新，引用归功）

| 组件 | 来源项目（bib key） | 本仓库位置 |
|---|---|---|
| 变维 BiMamba mixer（含逐层 FFN block） | S-Mamba (`smamba`) | `mamba_block.py:BiMambaEncoder` + 两分支 `channel_mixer` |
| 时域线性锚（分解 + 分量线性映射） | DLinear (`dlinear`) | `time_branch.py:SeriesDecomp/lin_seasonal/lin_trend` |
| 频域线性锚（频谱插值 L→L+H） | FITS (`fits`) | `freq_branch.py:spec_backbone` |
| 实例归一化 | RevIN (`revin`) | `dual_domain_model.py` forward |
| 序列分解 / 滑动平均趋势 | Autoformer/DLinear (`autoformer,dlinear`) | `time_branch.py:MovingAvg` |
| 选择性 SSM 本体（纯 torch 回退 + 官方核自动切换） | Mamba (`mamba`) | `mamba_block.py:MambaSSM` |
| 高通道"MLP over time + Mamba over variates"配方 | S-Mamba (`smamba`) | ECL/Traffic/PEMS configs（`time_encoder: mlp`） |

## 5. Benchmark 数据集 × 组件矩阵（shipped configs）

| 组件 | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Exchange | Weather | ECL | Traffic | Solar | PEMS03/04/07/08 | Illness* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 双分支 + 门控融合 + 零初始化（#1/#2 基座） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FITS 频谱锚（#2 频域侧） | ✓ 0.4 | ✓ 0.3 | ✗ | ✗ | ✗ | ✓ 0.0 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 变维 mixer（层数/位置） | 0 | 0 | 0 | 0 | 0 | 2/both | 2/shared | 2/shared | 1/both | 2/shared | 0 |
| `shared` 权重共享（#3） | — | — | — | — | — | ✗ | ✓ | ✓ | ✗ | ✓ | — |
| 时域编码器 | mamba | mamba | mamba | mamba | mamba | mamba | **mlp** | **mlp** | mamba | **mlp** | mamba |
| RevIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ | ✓ |
| alpha-RevIN / dispersion / 频域 BiMamba（#4/#5/#6） | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

\* Illness 仅有 config，论文数据表未报告。Solar 关闭 RevIN 是逐数据集消融决策，非架构创新。

## 6. 与累计消融表（`output/cumulative_ablation/cumulative_ablation.xlsx`）的互证

黄色（`FFFF99`）单元格 = no-op：所消融组件在该数据集配置中本已禁用，数值与上一累计变体逐字节相同，并在 "Min (best across variants, no-op blocks excluded)" 行排除。分布与 §5 矩阵完全一致：
- **w/o variate mixer 标黄**：ETT×4、Exchange、Illness —— mixer 层数为 0 的数据集；
- **w/o freq enhancements 标黄**：ETTm1、ECL、PEMS04/08、Solar —— 未启用 FITS 锚（Exchange 同为 no-op 但漏标）；
- **w/o RevIN 标黄**：仅 Solar —— `use_revin: false`。

## 7. 结论与写作警示

1. **实际承重且全量 benchmark 使用的核心创新 = #1（预测级门控双域融合）与 #2（线性锚 + 零初始化修正）**；#3（跨分支共享 mixer）用于 6 个高通道 benchmark 且有精度中性验证。这与论文 Related Work 的差异化主张（"prediction-level gated fusion of two complete forecasts"）一致。
2. **#4（alpha-RevIN）、#5（dispersion head）无任何 benchmark 配置启用**；#6（频域 BiMamba）连论文方法章都未主张（`4_Method.tex` 只写 complex-linear encoder + 可选 FITS map）——三者均**不能**列为论文贡献，除非补多 seed 消融实验。
3. **论文-实现不一致（需修正文稿或补实验）**：`4_Method.tex` 第 86 行声明 "The proposed temporal branch uses the same causal Mamba encoder for every forecasting task"，但 ECL/Traffic/PEMS 的 shipped config 实为 `time_encoder: mlp`（时域无 Mamba，S-Mamba 配方）。"causal state-space temporal pathway" 的贡献声明只在 ETT×4/Weather/Solar/Exchange/Illness 上成立，在 6 个高通道 benchmark 上不成立。
4. 主结果表中基线数值为**文献转载**（S-Mamba Tab.2 / iTransformer Appx. F.4），非本仓库复跑；效率表为同环境实测——论文引用时应保持该口径。
5. 遵守仓库实验纪律：Electricity d512+shared 等 "requires rerun" 配置在发表前必须以 `run_unified_baselines.py --seeds 5` 补 provenance-enabled 重跑（README/AGENTS.md：an expected value is not a result）。
