#!/usr/bin/env bash
#SBATCH --job-name=ocean-sa-ch7
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/vol/biomedic3/kw1025/ocean/logs/%x-%j.out
#SBATCH --error=/vol/biomedic3/kw1025/ocean/logs/%x-%j.err

set -euo pipefail

REPO=${REPO:-/vol/biomedic3/kw1025/ocean}
VENV=${VENV:-/vol/biomedic3/kw1025/oceanvenv}
ENV_FILE=${ENV_FILE:-${REPO}/scripts/envs/.env_sa_ch7_64}
RUN_NAME=${RUN_NAME:-sa_ch7_64_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
OUT_DIR=${OUT_DIR:-${REPO}/out/${RUN_NAME}}
RELOAD_SA=${RELOAD_SA:-}

cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/out"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

REPO="$REPO" VENV="$VENV" ENV_FILE="$ENV_FILE" RUN_NAME="$RUN_NAME" OUT_DIR="$OUT_DIR" RELOAD_SA="$RELOAD_SA" \
  bash scripts/train_slot_attention_ch7.sh
