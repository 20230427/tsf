# Ablation Matrix Statistical Audit

Primary endpoint: per-seed arithmetic mean MSE across requested horizons.

## Horizon-average primary comparisons

| Dataset | Variant | Family | n | Full | Variant | ΔMSE | 95% CI | raw p | BH q | exact p |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| ETTh2 | `freq_mamba` | spectral_strategy | 6 | 0.375843 | 0.375617 | -0.000226551 | [-0.0026464, 0.0021933] | 0.819373 | 0.819373 | 0.84375 |
| ETTh2 | `freq_only` | component_removal | 6 | 0.375843 | 0.372732 | -0.00311111 | [-0.00642006, 0.000197834] | 0.0603483 | 0.120697 | 0.0625 |
| ETTh2 | `fusion_sum` | fusion_strategy | 6 | 0.375843 | 0.372282 | -0.00356117 | [-0.00676148, -0.000360863] | 0.0353884 | 0.0353884 | 0.0625 |
| ETTh2 | `no_linear_backbone` | component_removal | 6 | 0.375843 | 0.376036 | 0.000193131 | [-0.00260175, 0.00298801] | 0.865984 | 0.865984 | 0.875 |
| ETTh2 | `no_revin` | component_removal | 6 | 0.375843 | 0.540957 | 0.165114 | [0.153742, 0.176485] | 2.59959e-07 | 1.03984e-06 | 0.03125 |
| ETTh2 | `time_mlp` | encoder_replacement | 6 | 0.375843 | 0.377503 | 0.00165952 | [-0.000260094, 0.00357914] | 0.0769069 | 0.0769069 | 0.125 |
| ETTh2 | `time_only` | component_removal | 6 | 0.375843 | 0.378361 | 0.00251742 | [-0.00185887, 0.0068937] | 0.199278 | 0.265704 | 0.21875 |

Per-horizon rows and exact BH values are retained in the JSON.
