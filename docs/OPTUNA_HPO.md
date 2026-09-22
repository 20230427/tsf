# Optuna 超参数搜索配置文档（DD-Mamba）

> 本文档阐述 DD-Mamba 的 Optuna HPO 栈：搜索空间与超参分组、优化协议、单卡/多卡运行方式、输出格式与配套机制。代码位于 `src/hpo/`（搜索空间、objective、GPU 预算、study 管理）与 `scripts/hpo/`（运行入口与查看工具）。

---

## 1. 概述

- **单阶段联合搜索**：每个 `(dataset, pred_len)` 一个 SQLite 持久化的 TPE study，`configs/hpo_search_spaces.yaml` 中声明的全部超参（架构开关 + 模型骨干/正则 + 训练超参）**联合采样**，无分阶段设计。
- **优化目标**：默认为每个 trial 的 **test MSE**（取 best-validation checkpoint 的测试集指标）；`--metric val_loss` 为防泄漏变体（`evaluate_test=False`，绝不构造 test loader，带泄漏守卫）。无论以何指标优化，导出一律报告 **test MSE 最优** trial 及其完整超参。
- **协议锁定**：`time_encoder: mamba`、`freq_backbone: fits`、`use_revin: true` 作为**单选项锁定**写入搜索空间（YAML 中以 `# pinned (single option)` 行内注释标注，不再搜索其他取值），anchor 与所有 trial 一致适用。
- **采样器**：`TPESampler(seed=2023, multivariate=True, group=True, constant_liar=True)`；`--n-startup`（默认 10）个随机启动 trial。
- **断点续跑**：study 状态全部落盘于 `output/hpo/optuna/<study>.db`，重跑同一命令按 finished-trial 计数差额补跑。
- **单卡/多卡**：单卡共享显存记账排队；多卡时每个 study 经文件锁池预约整卡，study 级并行。

## 2. 架构与文件

| 文件 | 职责 |
|---|---|
| `src/hpo/search_space.py` | YAML 空间加载/校验（schema `ddmamba-hpo-search-v1`）、`resolve_space`（数据集级覆盖 + fixed 移除）、`suggest_params`（按 §3.2 分组有序采样）、`normalize_conditions`（forward_to 透传 + 条件规范化）、`snap_to_space` / `anchor_params`（anchor 吸附与回读）、`build_trial_cfg`（配置优先级拼装） |
| `src/hpo/objective.py` | `create_objective`：suggest → 估值 → 显存预约 → `src.train.train(...)` → user_attrs 记录 → 磁盘卫生；`HpoLeakageError` 泄漏守卫 |
| `src/hpo/gpu_budget.py` | `GPUBudget`（单卡 MB 记账）、`MultiGPUBudget`（每卡一池 + best-fit 选卡）、stale-PID 回收 |
| `src/hpo/memory_estimate.py` | `estimate_ddmamba_memory_mb`：CPU 建模实测参数量 ×16B + 扫描/混合器激活项 + 512MB 开销 |
| `src/hpo/study_manager.py` | SQLite 存储、TPE 构造、`export_results`（CSV + 最优 test-MSE 记录 JSON）、`make_best_callback`（新 best 即替换重导出） |
| `src/hpo/cleanup.py` | 残留产物回收（Pamba `hpo_clean` 对应物）：识别 HPO trial checkpoint（新/旧命名）与死 pid 文件，无 liveness 保护、无条件清除；JSON 一律不删 |
| `scripts/hpo/run_hpo.py` | CLI 入口：GPU 环境钉定、anchor enqueue、续跑计数、`ConsecutivePrunedStopper`、SIGTERM 优雅停止、`--bg`/`--dry-run` |
| `scripts/hpo/hpo_dump.py` | `--study` 详情 / `--list` 全部研究概览 |
| `scripts/hpo/hpo_clean.py` | 一键清理 CLI（默认 dry-run，`--confirm` 才删除），见 §5.4 |
| `scripts/hpo/launch_hpo.sh` | 多数据集 nohup 批量启动（续跑内建） |
| `configs/hpo_search_spaces.yaml` | 搜索空间声明（§3） |
| `tests/test_hpo.py` | 30 项 CPU-only 测试（§8） |

## 3. 搜索空间配置

### 3.1 全部搜索超参总表（13 个）

| # | 分组 | 超参（dotted key） | 类型 | 范围 / 取值 | 说明 |
|---|---|---|---|---|---|
| 1 | 架构开关 | `model.time_encoder` | categorical | `mamba` | **锁定单选项**（YAML 行内注释标注）：仅因果 Mamba 时域编码器 |
| 2 | 架构开关 | `model.channel_mixer_layers` | categorical | `1` / `2` | 变维 BiMamba 层数；`0`（通道独立）已从网格排除 |
| 3 | 架构开关 | `model.mixer_placement` | categorical | `both` / `shared` | mixer 每分支独立 vs 跨分支权重共享 |
| 4 | 架构开关 | `model.freq_backbone` | categorical | `fits` | **锁定单选项**：FITS 频谱线性锚恒开 |
| 5 | 架构开关 | `model.use_revin` | categorical | `true` | **锁定单选项**：RevIN 恒开 |
| 6 | 模型骨干 | `model.d_model` | categorical | `128` / `256` / `512`（electricity 收窄为 `128` / `256`） | 双分支共享隐藏宽度；321 通道 electricity 的 d512 档在 24GB 卡上各 horizon 首步即 OOM，按 Pamba 对重量级数据集的做法整档剔除 |
| 7 | 模型骨干 | `model.mamba_layers` | categorical | `1` / `2` / `3` | Mamba 编码器层数 |
| 8 | 模型骨干 | `model.mamba_d_state` | categorical | `16` / `32` | SSM 状态维度 N |
| 9 | 模型骨干 | `model.freq_hidden` | categorical | `128` / `256` / `512` | 频域编码宽度 |
| 10 | 模型骨干 | `model.time_kernel_size` | categorical | `13` / `25` / `49` | 趋势提取滑动平均窗宽 |
| 11 | 模型骨干 | `model.freq_sparsity` | categorical | `0.0` / `0.1` / `0.2` / `0.3` | 低通截断（丢弃最高频比例） |
| 12 | 模型骨干 | `model.dropout` | categorical | `0.1` / `0.2` / `0.3` | **共享 dropout**，经 `forward_to` 透传至 `time_dropout` / `freq_dropout` / `head_dropout`（三处共用同一搜索值，见 §3.4） |
| 13 | 训练 | `train.lr` | loguniform | `[1e-4, 2e-3]` | 学习率（对数均匀） |

### 3.2 Multivariate 分组设计（3 组）

`suggest_params` 按 `PARAM_GROUPS`（`src/hpo/search_space.py`）固定的组序与组内顺序采样，保证采样字典与 Optuna 日志可复现、可读；`multivariate=True, group=True` 的 TPE 在整个空间的并集上建模联合依赖：

| 组名 | 键数 | 成员 | 分组逻辑 |
|---|---|---|---|
| `architecture`（架构开关） | 5 | `time_encoder`（锁定 mamba）、`channel_mixer_layers`、`mixer_placement`、`freq_backbone`（锁定 fits）、`use_revin`（锁定 true） | **组件选择**：决定模型中存在哪些架构块（哪个编码器、mixer 深度与位置、频谱锚、实例归一化），彼此强相关，先行采样；其中 3 个已按 profile 锁定为单选项 |
| `model`（模型骨干与正则） | 7 | `d_model`、`mamba_layers`、`mamba_d_state`、`freq_hidden`、`time_kernel_size`、`freq_sparsity`、`dropout`（共享，透传三处） | **模型超参**：存活组件的容量与正则——骨干尺寸、SSM 状态、频域宽度与低通、统一 dropout |
| `training`（训练超参） | 1 | `train.lr` | **训练侧**：优化过程参数（预留 `weight_decay`/`batch_size` 等扩展位） |

组序固定为 architecture → model → training；不属于任何组的键（自定义 YAML 扩展）追加在末尾采样。

### 3.3 协议默认（forced_defaults）与数据集锁定（fixed）

```yaml
forced_defaults:            # 协议默认（anchor 来源）；对应键已在空间中锁定单选项
  model.time_encoder: mamba
  model.use_revin: true

base:
  search:
    model.time_encoder: {type: categorical, choices: [mamba]}          # pinned (single option)
    model.channel_mixer_layers: {type: categorical, choices: [1, 2]}   # 0 已排除
    model.freq_backbone: {type: categorical, choices: [fits]}          # pinned (single option)
    model.use_revin: {type: categorical, choices: [true]}              # pinned (single option)
    ...

datasets:                   # 数据集级锁定（从搜索空间移除，trial 间不变）
  PEMS03: {fixed: {train.amp: false}}   # fp16 在变维 mixer 上溢出，同 configs/PEMS*.yaml
  PEMS04 / PEMS07 / PEMS08 同上
```

**配置优先级**（`build_trial_cfg`）：`dataset config < forced_defaults < 采样值 < dataset fixed`（fixed 为锁死决策，最后落地）。单选项锁定键（time_encoder / freq_backbone / use_revin）经采样值恒定生效，同时 forced_defaults 保证 anchor 在 tuned config 取相反值时（如 ECL/Traffic/PEMS 的 `mlp`、solar 的 `use_revin: false`）仍锚定到锁定取值。

### 3.4 参数透传（forward_to）与条件规范化

**透传**：搜索空间中的键可声明 `forward_to: [<dotted key>, ...]`——其采样值在 suggest 之后写入每个目标键。当前用于 dropout：

```yaml
model.dropout:
  type: categorical
  choices: [0.1, 0.2, 0.3]
  forward_to: [model.time_dropout, model.freq_dropout, model.head_dropout]
```

即 `time_dropout` / `freq_dropout` / `head_dropout` **共用同一个搜索值**（Optuna 只记录并建模 `model.dropout` 一个维度，搜索空间从 3 维降为 1 维）。透传发生在三处，保证各环节拿到的都是可直接使用的实键：

1. **trial 配置**（`normalize_conditions`，`src/hpo/search_space.py`）：采样后就地展开，`build_trial_cfg` 落入 cfg；展开动作记入 user_attr `normalized_conditions`；
2. **anchor**（`anchor_params`）：`model.dropout` 不在 dataset config 中，anchor 值从 `forward_to` 目标回读（取 tuned config 中第一个非空 dropout，如 ETTh1 的 `time_dropout: 0.2` → `model.dropout: 0.2`）；
3. **导出**（`export_results`）：`best_params` 展开后同时包含 `model.dropout` 与三个实键（同值），可直接作为 override 应用，无需手工换算。

校验约束：`forward_to` 目标不得同时出现在搜索空间（`resolve_space` 报错），且 `forward_to` 必须是非空 dotted key 列表（`validate_spec` 报错）。

**条件规范化**：`channel_mixer_layers == 0` 时 `mixer_placement` 无意义 → 钉为 `both` 并记入 user_attr。当前网格为 `[1, 2]`（0 已排除），该分支仅作为自定义空间下的守卫保留，正常搜索不再触发。

### 3.5 anchor trial（trial #0）

新 study 的首个 trial = 数据集 tuned config 读值 + forced_defaults，经 `snap_to_space` 吸入网格——搜索始终从 shipped recipe 起步，anchor 完成即获得一条与 tuned config 可比的基线记录。snap 示例（ETTh1）：`freq_sparsity: 0.4` → `0.3`；`channel_mixer_layers: 0` → `1`（网格已无 0）；`train.lr: 5e-4` 落在 loguniform 区间内不变。

### 3.6 不参与搜索的项（继承 dataset config）

`epochs` / `lr_scheduler` / `patience` / `batch_size` / `seq_len` / ETT canonical `split_protocol` 等一律沿用 `configs/<dataset>.yaml`（可经 CLI override 临时覆盖，如 `--epochs 2` 快筛、`--data.batch_size 16`）。

## 4. 优化流程

一个 trial 的生命周期：

```
suggest_params（按 §3.2 分组有序采样）
→ normalize_conditions（forward_to 透传 + 条件规范化）
→ build_trial_cfg（§3.3 优先级拼装，experiment.name = "<ds>_pl<pred_len>_hpo_t<n>_s<i>"
  唯一——pred_len 必须入名：并行 per-horizon study 的 trial 号都从 0 重新编号，
  缺少它时并发 study 会互相覆盖 checkpoints/<name>_best.pt，表现为
  state_dict 尺寸不匹配崩溃）
→ estimate_ddmamba_memory_mb（超 GPU 上限 → 直接返回 inf，不训练）
→ budget.acquire（flock 显存预约；超时 → PRUNED）
→ train(cfg, evaluate_test=…)（--seeds N 时逐 seed 平均）
→ user_attrs 记录（mse/mae/val_loss/seed/param_count/best_epoch/est_mb/config_sha256/status）
→ 删除该 trial 的 checkpoint 与 per-run results JSON（--keep-checkpoint 豁免；
  失败 trial——error / 运行期 OOM / 泄漏守卫——同样在 finally 中按确定性命名清理，
  半写产物不残留；进程被硬杀的残留由 scripts/hpo/hpo_clean.py 回收，见 §5.4）
→ budget.release + gc + empty_cache
```

异常处理：CUDA OOM → PRUNED（`status=OOM`）；其他异常 → PRUNED（`status=error` + traceback 存 user_attr）；`val_loss` 模式意外返回 test 指标 → `HpoLeakageError` 直接 FAIL（大声失败终止 study）。连续 5 个 PRUNED 或 10 个 inf trial 触发 `ConsecutivePrunedStopper` 停止 study；SIGTERM 触发优雅停止。

## 5. 使用指南

### 5.1 安装与运行

```bash
pip install -e ".[hpo]"        # optuna>=4.0

# 单卡，一个 (dataset, horizon) study，50 trials：
python scripts/hpo/run_hpo.py --config configs/weather.yaml --pred_len 96 --n_trials 50

# 全部标准 horizon（96/192/336/720 各一 study），后台运行（日志 log/hpo/）：
python scripts/hpo/run_hpo.py --config configs/ETTh1.yaml --pred_len all --bg

# 多卡：每个 (dataset, pred_len) study 预约一整张最空闲的卡，study 级并行：
python scripts/hpo/run_hpo.py --config configs/weather.yaml --pred_len all --nodes 4 --gpu 0

# 只打印解析后的空间 / anchor / 显存估计，不训练：
python scripts/hpo/run_hpo.py --config configs/ETTh1.yaml --pred_len 96 --dry-run

# 批量（nohup 分离；重跑同一命令即续跑）：
./scripts/hpo/launch_hpo.sh configs/ETTh1.yaml configs/weather.yaml --trials 30
```

日志格式：objective 强制 trial 训练使用 `train.log_style: pamba`——无逐 batch 进度条，每 epoch 仅 Pamba 式三行（`Epoch N : LR [...]`、cost time、`Train Loss / Vali Loss` 摘要）加存档/早停消息。study 日志重定向到文件时 tqdm 会每 batch 刷一行，该开关避免 `log/hpo/*.log` 膨胀；普通训练入口（`python -m src.train`）不受影响，仍可用 `--train.log_style pamba` 手动启用。

### 5.2 CLI 旋钮

| 参数 | 说明 |
|---|---|
| `--metric mse\|mae\|val_loss` | 优化目标（默认 test MSE；导出始终报告 test MSE 最优记录） |
| `--seeds N` | 每 trial N 个 seed 取平均（默认 1） |
| `--epochs N` | 覆盖训练 epoch 数（0 = 用 dataset config；快筛 `--epochs 2`） |
| `--gpu N` / `--nodes K` | 单卡物理卡号 / 多卡卡数（`CUDA_VISIBLE_DEVICES` 钉定后再 import torch） |
| `--keep-checkpoint` | 保留每 trial checkpoint（默认记录后即删，DB 为唯一持久记录） |
| `--timeout S` / `--n-startup N` | 单 study 时限 / TPE 随机启动 trial 数（默认 10） |
| `--search-space PATH` | 搜索空间 YAML（默认 `configs/hpo_search_spaces.yaml`） |
| `--bg` / `--dry-run` | 日志重定向 `log/hpo/<ds>_<pl>_study.log` / 干跑检查 |
| 未知参数 | 照常作为 config override（`--train.amp false` 等，与 `src.train` CLI 同语法） |

### 5.3 查看与应用结果

```bash
python scripts/hpo/hpo_dump.py --list                       # 全部 study 概览
python scripts/hpo/hpo_dump.py --study DDMamba_ETTh1_pl96   # best trial 详情
# 可选 Web 仪表盘：
pip install optuna-dashboard && optuna-dashboard sqlite:///output/hpo/optuna/DDMamba_ETTh1_pl96.db
```

应用 best 参数：`best_params` 为完整 dotted override 集（搜索值 + 协议默认 + 数据集 fixed + `experiment.seed`），可直接喂给训练：

```bash
python -m src.train --config configs/ETTh1.yaml $(python -c \
  "import json;d=json.load(open('output/hpo/DDMamba_ETTh1_pl96_best_params.json'))['best_params'];print(' '.join(f'--{k} {v}' for k,v in d.items()))")
```

### 5.4 磁盘卫生与残留清理（hpo_clean）

正常运行时每个 trial 的 checkpoint 与 per-run results JSON 在指标入库后即删（DB 为唯一持久记录），失败 trial 也由 objective 的 finally 块清理。泄漏只来自进程被硬杀（节点重启、`kill -9`）或旧命名残留；`hpo_clean` 一键回收：

```bash
python scripts/hpo/hpo_clean.py                    # dry-run：只列出与统计
python scripts/hpo/hpo_clean.py --confirm          # 实际删除
python scripts/hpo/hpo_clean.py --dataset electricity --pred_len 192 --confirm
```

规则：

- 只匹配 HPO trial 产物：`checkpoints/<ds>_pl<pl>_hpo_t<n>_s<i>_best.pt`（新命名）与 `checkpoints/<ds>_hpo_t<n>_s<i>_best.pt`（旧命名）；常规实验 checkpoint（如 `ETTh1_best.pt`）与一切 **JSON 文件永不删除**（best-params 导出、per-run 记录、CSV、SQLite DB 全保留）。
- **无 liveness 保护（刻意设计）**：每个匹配到的 HPO trial checkpoint 一律删除，不查 study DB 的 RUNNING trial、不看 mtime——进程生命周期由使用者手工管理。运行中 trial 的 checkpoint 被删无稳定性风险（`src.train` 的 final load 有 `os.path.exists` 守卫，仅退化为用末 epoch 权重评测该 trial）。
- 顺带回收 `output/hpo/pids/` 中指向死进程的 pid 文件。
- 纯 stdlib，不依赖 torch/optuna import。

## 6. 输出

```
output/hpo/
├── optuna/DDMamba_<ds>_pl<H>.db        # Optuna SQLite（续跑真相源；gitignored）
├── csv/DDMamba_<ds>_pl<H>_results.csv  # 全 trial 表（pandas trials_dataframe）
├── DDMamba_<ds>_pl<H>_best_params.json # 最优 test-MSE 记录（见下）
└── gpu_budget{,_multi}.json/.lock      # 显存池状态（gitignored）
log/hpo/                                # --bg / launcher 日志（gitignored）
```

`<study>_best_params.json` 只含**单一最优记录**（无任何非最优 trial 记录；全量历史在 CSV 与 DB 中）：

| 字段 | 含义 |
|---|---|
| `study_name` / `version` | `DDMamba_<ds>_pl<H>` / `ddmamba-hpo-best-v2` |
| `metric` / `best_value` | 始终为最优 **test MSE**（按 trial 的 `mse` user_attr 选取，与优化目标无关；val-only study 回退 `study.best_trial`） |
| `best_params` | 完整 dotted override 集：trial 采样值（优先）∪ fill（forced_defaults + dataset fixed）∪ `experiment.seed`；`forward_to` 已展开（含 `model.dropout` 及其三个实键同值），可直接应用 |
| `best_trial_number` | 最优 trial 编号 |
| `best_trial_user_attrs` | 按序 `mse, mae, val_loss, seed, param_count, best_epoch, est_mb, config_sha256, status, seeds_used` |

每逢新 best，`make_best_callback` 以当前最优 test-MSE 记录**替换**重写该文件。该 JSON 刻意不携带仓库 provenance schema 的 `run_records` 等字段，因此 `scripts/validate_provenance.py` 对其判 `invalid` 属预期；可追溯性信息保留在 `config_sha256` user_attr、CSV 与 SQLite DB 中。

## 7. 显存预算与多卡机制

- **估算**（`estimate_ddmamba_memory_mb`）：CPU 上 `build_model` 数真实参数量 ×16 B（权重+梯度+AdamW 状态）+ 激活项（时轴 Mamba 扫描的 `(B·C, L, d_inner, N)` 离散化张量——官方 fused kernel 可用时按 30% 折算，2026-09-06 electricity 扫参实测校准：旧 15% 因子把 d512+shared+2 层 mixer +d_state 32 的 anchor 估为 18.1 GB，实际在 24.5 GB 卡上首步 OOM；双向变维 mixer 按 层数×2 估计）+ 512 MB 固定开销，整体 ×1.05。
- **单卡**：`GPUBudget` 以 `output/hpo/gpu_budget.{lock,json}`（flock 排他）做 MB 记账；估算超限的 trial 轮询等待（默认 1800s，超时 PRUNED），死进程预约按 PID 自动回收。
- **多卡**（`--nodes > 1`）：`MultiGPUBudget` 每卡一池，study 启动时 best-fit 预约一整卡（study 级并行、trial 级串行）；`release_budget_by_pid` 可手动清理。
- 运行期 CUDA OOM 捕获为 PRUNED；频繁超限时可在 YAML 中用 `datasets.<name>.search` 收窄 `model.d_model` 上限（现成实例：electricity）。
- **导出对 inf 健壮**：全部完成 trial 均为 inf（如 anchor 全 OOM）时 `export_results` 只写 CSV、跳过 best-params JSON，不再触发 `json.dump(allow_nan=False)` 的 `ValueError`；user_attr 中的非有限值写为 `null`。

## 8. 测试（`tests/test_hpo.py`，30 项，CPU-only）

- **空间**：schema 校验、forced_defaults 断言（mamba / RevIN-on）、**profile 锁定**（time_encoder/freq_backbone/use_revin 单选项、mixer 网格 [1,2]、mamba_layers [1,2,3]）、search-fixed 重叠拒绝、**分组覆盖与组序**（architecture → model → training，架构组恰为五个组件开关）、**共享 dropout 透传**（normalize 展开三实键、纯函数不改变输入、forward 目标不得被搜索、anchor 从 tuned config 回读）、`snap_to_space` 就近吸附、anchor 全部落网格、`build_trial_cfg` 优先级链、**electricity d_model 收窄为 [128, 256]**（24GB 必 OOM 档剔除，anchor 吸附 d256）；
- **objective**（monkeypatch 假 `train`）：运行并记 user_attrs、参数正确落到 cfg（含共享 dropout 透传至三实键）、val 模式绝不触发测试评测、泄漏守卫使 trial FAIL、超显存跳过、trial 产物清理、**trial 名含 pred_len**（跨 horizon 命名唯一回归）、**失败 trial（error/OOM）同样清理产物**、`--keep-checkpoint` 保留失败产物；
- **GPU 预算**（tmp_path）：acquire/release/超限拒绝、stale-PID 回收、MultiGPU best-fit 选卡与满员等待；
- **study 往返**：tmp sqlite + anchor enqueue + 导出格式（无 run_records、fill 合并优先级、attrs 序 mse/mae 前置）；**mae 优化的 study 导出仍选 test MSE 最优 trial**；**全-inf study 导出只写 CSV 不炸**；
- **cleanup**：checkpoint 文件名分类（新/旧命名、常规文件与 JSON 不匹配、新命名优先）、plan/run 集成（匹配即删、dataset/pred_len 过滤、常规/JSON 保留、死 pid 回收）。

## 9. FAQ / 注意事项

- **改了搜索空间**：study 名不随空间变化——空间变更后请删除旧 DB 或更换 config 的 `experiment.name`，避免 TPE 在混合空间上续跑。
- **mamba 内核**：显存估计自动探测官方核可用性（无核纯 torch 路径按全量离散化张量估算）。HPO 需在 GPU 主机上运行（装了内核的机器上 CPU 张量 forward 会崩，见 AGENTS.md 环境注记）。
- **optuna 版本**：4.x；`multivariate`/`group`/`constant_liar` 为 experimental 特性，警告可忽略。
- **test-MSE 搜索结果的用途**：搜索出的超参用于最终报告数值时，仍须以多 seed 确认重跑（"an expected value is not a result"）。
