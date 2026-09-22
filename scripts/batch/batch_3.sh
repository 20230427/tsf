#!/bin/bash
#SBATCH --nodes=1 --gres=gpu:4 --time=48:00:00 --mail-type=ALL
#SBATCH --job-name=ddmamba-hpo-ettm1
# DD-Mamba HPO: 8 datasets x 4 pred_lens = 32 studies across 4 GPUs

set -euo pipefail

bash scripts/hpo/hpo_batch.sh --n_gpus 4 \
    --study "ETTm1:96:200:0"             \
    --study "ETTm1:192:200:1"            \
    --study "ETTm1:336:200:2"            \
    --study "ETTm1:720:200:3"            

    # --study "ETTh1:96:200:0"           \
    # --study "ETTh1:192:200:1"          \
    # --study "ETTh1:336:200:2"          \
    # --study "ETTh1:720:200:3"          \

    # --study "ETTh2:96:200:0"           \
    # --study "ETTh2:192:200:1"          \
    # --study "ETTh2:336:200:2"          \
    # --study "ETTh2:720:200:3"          \

    # --study "ETTm2:96:200:0"           \
    # --study "ETTm2:192:200:1"          \
    # --study "ETTm2:336:200:2"          \
    # --study "ETTm2:720:200:3"          \

    # --study "weather:96:200:0"         \
    # --study "weather:192:200:1"        \
    # --study "weather:336:200:2"        \
    # --study "weather:720:200:3"        \

    # --study "electricity:96:200:0"     \
    # --study "electricity:192:200:1"    \
    # --study "electricity:336:200:2"    \
    # --study "electricity:720:200:3"    \

    # --study "exchange_rate:96:200:0"   \
    # --study "exchange_rate:192:200:1"  \
    # --study "exchange_rate:336:200:2"  \
    # --study "exchange_rate:720:200:3"  \

    # --study "solar:96:200:0"           \
    # --study "solar:192:200:1"          \
    # --study "solar:336:200:2"          \
    # --study "solar:720:200:3"          \

    # --study "traffic:96:200:0"         \
    # --study "traffic:192:200:1"        \
    # --study "traffic:336:200:2"        \
    # --study "traffic:720:200:3"        \

    # --study "PEMS03:12:200:0"          \
    # --study "PEMS03:24:200:1"          \
    # --study "PEMS03:48:200:2"          \
    # --study "PEMS03:96:200:3"          \

    # --study "PEMS04:12:200:0"          \
    # --study "PEMS04:24:200:1"          \
    # --study "PEMS04:48:200:2"          \
    # --study "PEMS04:96:200:3"          \

    # --study "PEMS07:12:200:0"          \
    # --study "PEMS07:24:200:1"          \
    # --study "PEMS07:48:200:2"          \
    # --study "PEMS07:96:200:3"          \

    # --study "PEMS08:12:200:0"          \
    # --study "PEMS08:24:200:1"          \
    # --study "PEMS08:48:200:2"          \
    # --study "PEMS08:96:200:3"          \

    # --study "illness:24:200:0"         \
    # --study "illness:36:200:1"         \
    # --study "illness:48:200:2"         \
    # --study "illness:60:200:3"         \

