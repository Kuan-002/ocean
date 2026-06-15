#!/usr/bin/env bash
#SBATCH --job-name=ocean-v1-tau
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/vol/biomedic3/kw1025/ocean/logs/%x-%j.out
#SBATCH --error=/vol/biomedic3/kw1025/ocean/logs/%x-%j.err

set -euo pipefail

REPO=/vol/biomedic3/kw1025/ocean
RUN_DIR=${RUN_DIR:-${REPO}/out/selector_v1_ac_no_stop_tau_ch7_20260614_073514}
TAUS=${TAUS:-0.65,0.70,0.75,0.80,0.85}
VENV=/vol/biomedic3/kw1025/oceanvenv

cd "$REPO"
mkdir -p "$REPO/logs"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "RUN_DIR: ${RUN_DIR}"
echo "TAUS: ${TAUS}"
date

# shellcheck source=/dev/null
source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DEVICE="${DEVICE:-cuda}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-12}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

python scripts/eval_selector_v1_ac_tau_sweep.py \
    --run_dir "${RUN_DIR}" \
    --taus "${TAUS}"

date
echo "=== Done. ==="
