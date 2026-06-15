#!/usr/bin/env bash
#SBATCH --job-name=ocean-v1-viz
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/vol/biomedic3/kw1025/ocean/logs/%x-%j.out
#SBATCH --error=/vol/biomedic3/kw1025/ocean/logs/%x-%j.err

set -euo pipefail

REPO=/vol/biomedic3/kw1025/ocean
RUN_DIR=${RUN_DIR:-${REPO}/out/selector_v1_ac_no_pos_no_classmin_ch7_20260614_052944}
OUT_DIR=${OUT_DIR:-${RUN_DIR}/visualizations/test_slot_paths_post}
SPLIT=${SPLIT:-test}
PER_CLASS_CORRECT=${PER_CLASS_CORRECT:-5}
PER_CLASS_WRONG=${PER_CLASS_WRONG:-5}
VENV=/vol/biomedic3/kw1025/oceanvenv

cd "$REPO"
mkdir -p "$REPO/logs" "$OUT_DIR" "$REPO/.matplotlib"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "RUN_DIR: ${RUN_DIR}"
echo "OUT_DIR: ${OUT_DIR}"
echo "SPLIT: ${SPLIT}"
echo "PER_CLASS_CORRECT: ${PER_CLASS_CORRECT}"
echo "PER_CLASS_WRONG: ${PER_CLASS_WRONG}"
date

# shellcheck source=/dev/null
source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DEVICE="${DEVICE:-cuda}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-12}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO}/.matplotlib}"

python scripts/visualize_selector_v1_ac_paths.py \
    --run_dir "${RUN_DIR}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --per_class_correct "${PER_CLASS_CORRECT}" \
    --per_class_wrong "${PER_CLASS_WRONG}" \
    --device "${DEVICE}"

date
echo "=== Done. Output: ${OUT_DIR} ==="
