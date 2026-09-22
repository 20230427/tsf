# DD-Mamba 复现材料说明

本目录保存论文审阅、模型实现和实验证据中适合公开版本控制的核心材料。上传目标是让读者能够检查数据协议、解析实验配置、运行模型和审计已保存的结果；它不包含原始数据、模型权重或本机缓存。

## 1. 内容索引

| 路径 | 内容 | 用途 |
|---|---|---|
| [`src/`](src/) | DD-Mamba、基线模型、数据加载、训练、评估和 HPO 实现 | 核心代码 |
| [`configs/`](configs/) | 13 个主要数据集及默认/搜索空间配置 | 数据集级实验配方 |
| [`scripts/`](scripts/) | 主结果、消融、基线、鲁棒性、效率、统计和 provenance 脚本 | 重建实验 |
| [`tests/`](tests/) | 模型、配置、结果、统计与审计测试 | 实现验证 |
| [`results/`](results/) | 经过筛选并纳入版本控制的 JSON 结果 | 数值证据 |
| [`generated/statistical_audit/`](generated/statistical_audit/) | 配对效应、置信区间、p/q 值及精确符号翻转敏感性分析 | 统计审计 |
| [`generated/ablation_matrix_audit.json`](generated/ablation_matrix_audit.json) | 消融矩阵机器可读审计 | 消融证据 |
| [`DATASETS.md`](DATASETS.md) | 数据集来源、规模、变量、划分和统计特征 | 数据说明 |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | 实验状态、命令和注意事项 | 实验说明 |
| [`docs/MODEL_SPECIFICATION.md`](docs/MODEL_SPECIFICATION.md) | 模型设计约束和实现说明 | 方法核查 |
| [`docs/OPTUNA_HPO.md`](docs/OPTUNA_HPO.md) | Optuna HPO 协议 | 超参数搜索 |
| [`assets/`](assets/) | 原始设计说明、HPO 说明及效率结果工作簿 | 补充材料与表格数据 |
| [`PROJECT_README.md`](PROJECT_README.md) | 完整项目命令、结果说明和环境注意事项 | 使用入口 |
| [`README_PEER_REVIEW_V14.md`](README_PEER_REVIEW_V14.md) | 针对论文 v14 的模拟同行评审 | 修订依据 |

## 2. 未上传内容

以下内容因体积、授权或可再生性原因不进入 Git：

- `data/` 下的 CSV、NPZ、PT 缓存等原始或派生数据；
- `checkpoints/` 下的模型权重与临时结果；
- HPO 数据库、GPU 调度状态、日志和运行缓存；
- Python 虚拟环境、`__pycache__`、`.pyc` 和测试缓存；
- 本地参考文献仓库、临时论文构建目录及个人报销等无关文件。

`assets/ddmamba_eff_results_1.xlsx` 是经过筛选的小型结果工作簿，因此随仓库保存；它不包含原始训练数据。

数据文件应从其原始公开来源获取，并按 [`DATASETS.md`](DATASETS.md) 和 `configs/*.yaml` 放置。项目的数据加载器可能优先使用与 CSV 同名的 `.pt` 缓存；实际加载路径会打印为 `[data] using pt cache:`，复现时必须记录该信息。

## 3. 环境安装

Python 要求为 3.10 或更高版本。PyTorch 需根据本机 CUDA 单独安装，然后安装项目依赖：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

上面的 PyTorch wheel 仅是 CUDA 12.8 示例。应选择与本机驱动和 GPU 兼容的官方 wheel。`mamba-ssm` 与 `causal-conv1d` 为可选依赖；CPU 运行建议显式使用 `--model.use_official_mamba false`。

## 4. 最小验证与训练

无需真实数据的端到端检查：

```bash
python scripts/run_synthetic.py
```

运行测试：

```bash
pytest -q
```

训练和评估示例：

```bash
python -m src.train --config configs/ETTh1.yaml
python -m src.evaluate --checkpoint checkpoints/ETTh1_best.pt --plot forecast.png
```

批量实验与 HPO：

```bash
./scripts/run_benchmarks.sh ETTh1 weather
python scripts/hpo/run_hpo.py --config configs/weather.yaml --pred_len 96 --n_trials 50
```

详细命令和环境差异见 [`PROJECT_README.md`](PROJECT_README.md)。

## 5. 数据划分与防泄漏要求

- ETTh/ETTm 必须采用配置中的固定 `ETTh`/`ETTm` 月份边界，不能改成比例划分后仍与文献结果直接比较。
- 标准化器只能在训练区间拟合；滑窗构造和测试集访问必须遵守 `src/data/` 的实现。
- 消融和基线筛选优先使用 `--validation-only`；筛选阶段不得构造测试集。
- Optuna HPO 若使用测试 MSE，应明确标为 Pamba 兼容协议，不得与泄漏安全的 validation-only 结论混写。
- 所有新结果应保留 resolved config、配置哈希、代码提交、随机种子和数据路径/来源。

## 6. 结果与统计证据边界

`results/` 中既有带 provenance schema 的新结果，也可能包含早期 legacy artifacts。使用前运行：

```bash
python scripts/validate_provenance.py results/<file>.json
```

`legacy_unverifiable` 表示文件仍可用于历史检查，但不能被当作可复现的新证据合并。完整统计审计的机器可读真值为：

```text
generated/statistical_audit/complete_statistical_audit.json
```

可用下列命令重新生成 Markdown 和 LaTeX 审计附录：

```bash
python scripts/generate_statistical_audit.py
```

早期结果保存了运行顺序和运行次数，但未始终保存真实 seed ID；统计审计以匿名 `pair_index` 配对，不能把它误报为 seed ID。

## 7. 当前复现限制

本次上传显著改善了代码、配置和结果可见性，但不等于所有论文表格都已达到完全复现状态：

1. 部分历史结果仍缺少真实 seed ID 或完整运行环境信息；
2. 原始数据和 checkpoint 未随仓库分发；
3. 已发表基线数字与本地 matched-run 结果必须继续分开解释；
4. 论文中的 dataset–horizon 配方、模型选择记录和统计表仍应在正文或补充材料中明确对应。

这些限制与优先修订项详见 [`README_PEER_REVIEW_V14.md`](README_PEER_REVIEW_V14.md)。

## 8. 上传清单与隐私处理

上传前已排除包含本机绝对路径的旧 `generated/submission_audit.json`，并扫描常见 API key、token、password 和私钥模式。本次上传不包含检测到的凭据、原始数据或个人文档。
