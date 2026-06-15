#!/usr/bin/env bash
#SBATCH --job-name=ocean-bench-resnet
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/vol/biomedic3/kw1025/ocean/logs/%x-%j.out
#SBATCH --error=/vol/biomedic3/kw1025/ocean/logs/%x-%j.err

set -euo pipefail

REPO=/vol/biomedic3/kw1025/ocean
ENV_FILE=${REPO}/scripts/envs/.envA_ch7_attn_select_v9c
OUT_DIR=${REPO}/out/benchmark_resnet18_ch7_$(date +%Y%m%d_%H%M%S)
VENV=/vol/biomedic3/kw1025/oceanvenv

cd "$REPO"
mkdir -p "$REPO/logs"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo: $REPO"
echo "ENV_FILE: $ENV_FILE"
echo "OUT_DIR: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DEVICE="${DEVICE:-cuda}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-12}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

export BENCH_EPOCHS="${BENCH_EPOCHS:-100}"
export BENCH_LR="${BENCH_LR:-0.0003}"
export BENCH_WEIGHT_DECAY="${BENCH_WEIGHT_DECAY:-0.0001}"
export BENCH_EARLY_STOP_PATIENCE="${BENCH_EARLY_STOP_PATIENCE:-20}"
export BENCH_EARLY_STOP_MIN_DELTA="${BENCH_EARLY_STOP_MIN_DELTA:-0.0001}"

echo "=== ResNet18 Classification Benchmark ==="

python scripts/run_train_benchmark_classifier.py \
    --env_path "$ENV_FILE" \
    --out_subpath "$OUT_DIR" \
    --model resnet18

date
echo "=== Done. Output: ${OUT_DIR} ==="
