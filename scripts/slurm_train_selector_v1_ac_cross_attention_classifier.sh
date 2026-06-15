#!/usr/bin/env bash
#SBATCH --job-name=ocean-v1-xattncls
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/vol/biomedic3/kw1025/ocean/logs/%x-%j.out
#SBATCH --error=/vol/biomedic3/kw1025/ocean/logs/%x-%j.err

set -euo pipefail

REPO=/vol/biomedic3/kw1025/ocean
ENV_FILE=${REPO}/scripts/envs/.envA_ch7_attn_select_v9c
SA_CKPT=${REPO}/out/sa/999_ckpt.pt
OUT_DIR=${REPO}/out/selector_v1_ac_cross_attention_classifier_no_pos_no_classmin_ch7_$(date +%Y%m%d_%H%M%S)
VENV=/vol/biomedic3/kw1025/oceanvenv

cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/.matplotlib"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo: $REPO"
echo "ENV_FILE: $ENV_FILE"
echo "SA_CKPT: $SA_CKPT"
echo "OUT_DIR: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing env file: $ENV_FILE" >&2
    exit 1
fi
if [[ ! -f "$SA_CKPT" ]]; then
    echo "Missing SA checkpoint: $SA_CKPT" >&2
    exit 1
fi
if [[ ! -d "$VENV" ]]; then
    echo "Missing virtualenv: $VENV" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export DEVICE="${DEVICE:-cuda}"
export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-12}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO}/.matplotlib}"

python scripts/train_selector_v1_ac.py \
    --env_path "${ENV_FILE}" \
    --sa_checkpoint "${SA_CKPT}" \
    --out_subpath "${OUT_DIR}" \
    --stage0_epochs 0 \
    --stage1_epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --embed_dim 512 \
    --pos_dim 0 \
    --eval_every 1 \
    --eval_batches 0 \
    --min_steps 3 \
    --class_min_slots "" \
    --target_min_slots 3 \
    --target_max_slots 0 \
    --lambda_slot 0.01 \
    --lambda_over 0.0 \
    --lambda_under 0.0 \
    --entropy_coef 0.005 \
    --class_coef 0.5 \
    --full_order_class_coef 0.2 \
    --value_coef 0.25 \
    --early_exit_conf 0.8 \
    --max_grad_norm 1.0 \
    --early_stop_patience 20 \
    --early_stop_min_delta 0.0001 \
    --cross_attention_classifier

date
echo "=== Done. Output: ${OUT_DIR} ==="
