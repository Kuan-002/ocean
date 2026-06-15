#!/usr/bin/env bash
#SBATCH --job-name=ocean-v1-ac
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/vol/biomedic3/kw1025/ocean/logs/%x-%j.out
#SBATCH --error=/vol/biomedic3/kw1025/ocean/logs/%x-%j.err

set -euo pipefail

REPO=/vol/biomedic3/kw1025/ocean
ENV_FILE=${REPO}/scripts/envs/.envA_ch7_attn_select_v9c
SA_CKPT=${REPO}/out/sa/999_ckpt.pt
OUT_DIR=${REPO}/out/selector_v1_ac_no_stop_tau_ch7_$(date +%Y%m%d_%H%M%S)
VENV=/vol/biomedic3/kw1025/oceanvenv

cd "$REPO"
mkdir -p /vol/biomedic3/kw1025/ocean/logs

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

# Slot Selector V1: no prewarm, no teacher, step-wise actor-critic from epoch 0.
export AC_LR="${AC_LR:-1e-4}"
export AC_WEIGHT_DECAY="${AC_WEIGHT_DECAY:-1e-4}"
export AC_STAGE1_EPOCHS="${AC_STAGE1_EPOCHS:-100}"
export AC_EMBED_DIM="${AC_EMBED_DIM:-512}"
export AC_POS_DIM="${AC_POS_DIM:-0}"
export AC_MIN_STEPS="${AC_MIN_STEPS:-3}"
export AC_CLASS_MIN_SLOTS="${AC_CLASS_MIN_SLOTS:-}"
export AC_TARGET_MIN_SLOTS="${AC_TARGET_MIN_SLOTS:-3}"
export AC_TARGET_MAX_SLOTS="${AC_TARGET_MAX_SLOTS:-0}"
export AC_LAMBDA_SLOT="${AC_LAMBDA_SLOT:-0.01}"
export AC_LAMBDA_OVER="${AC_LAMBDA_OVER:-0.0}"
export AC_LAMBDA_UNDER="${AC_LAMBDA_UNDER:-0.0}"
export AC_ENTROPY_COEF="${AC_ENTROPY_COEF:-0.005}"
export AC_CLASS_COEF="${AC_CLASS_COEF:-0.5}"
export AC_FULL_ORDER_CLASS_COEF="${AC_FULL_ORDER_CLASS_COEF:-0.2}"
export AC_VALUE_COEF="${AC_VALUE_COEF:-0.25}"
export AC_MAX_GRAD_NORM="${AC_MAX_GRAD_NORM:-1.0}"
export AC_EARLY_STOP_PATIENCE="${AC_EARLY_STOP_PATIENCE:-20}"
export AC_EARLY_STOP_MIN_DELTA="${AC_EARLY_STOP_MIN_DELTA:-0.0001}"
export AC_EARLY_EXIT_CONF="${AC_EARLY_EXIT_CONF:-0.8}"
export AC_DISABLE_STOP_ACTION="${AC_DISABLE_STOP_ACTION:-1}"

echo "=== Slot Selector V1 No-Prewarm Actor-Critic ==="
echo "AC_POS_DIM: ${AC_POS_DIM}"
echo "AC_CLASS_MIN_SLOTS: ${AC_CLASS_MIN_SLOTS}"
echo "AC_TARGET_MAX_SLOTS: ${AC_TARGET_MAX_SLOTS}"
echo "AC_LAMBDA_OVER: ${AC_LAMBDA_OVER}"
echo "AC_LAMBDA_UNDER: ${AC_LAMBDA_UNDER}"
echo "AC_ENTROPY_COEF: ${AC_ENTROPY_COEF}"
echo "AC_EARLY_EXIT_CONF: ${AC_EARLY_EXIT_CONF}"
echo "AC_DISABLE_STOP_ACTION: ${AC_DISABLE_STOP_ACTION}"

EXTRA_ARGS=()
if [[ "${AC_DISABLE_STOP_ACTION}" == "1" ]]; then
    EXTRA_ARGS+=(--disable_stop_action)
fi

python scripts/train_selector_v1_ac.py \
    --env_path "${ENV_FILE}" \
    --sa_checkpoint "${SA_CKPT}" \
    --out_subpath "${OUT_DIR}" \
    --stage0_epochs 0 \
    --stage1_epochs "${AC_STAGE1_EPOCHS}" \
    --lr "${AC_LR}" \
    --weight_decay "${AC_WEIGHT_DECAY}" \
    --embed_dim "${AC_EMBED_DIM}" \
    --pos_dim "${AC_POS_DIM}" \
    --eval_every 1 \
    --eval_batches 0 \
    --min_steps "${AC_MIN_STEPS}" \
    --class_min_slots "${AC_CLASS_MIN_SLOTS}" \
    --target_min_slots "${AC_TARGET_MIN_SLOTS}" \
    --target_max_slots "${AC_TARGET_MAX_SLOTS}" \
    --lambda_slot "${AC_LAMBDA_SLOT}" \
    --lambda_over "${AC_LAMBDA_OVER}" \
    --lambda_under "${AC_LAMBDA_UNDER}" \
    --entropy_coef "${AC_ENTROPY_COEF}" \
    --class_coef "${AC_CLASS_COEF}" \
    --full_order_class_coef "${AC_FULL_ORDER_CLASS_COEF}" \
    --value_coef "${AC_VALUE_COEF}" \
    --early_exit_conf "${AC_EARLY_EXIT_CONF}" \
    --max_grad_norm "${AC_MAX_GRAD_NORM}" \
    --early_stop_patience "${AC_EARLY_STOP_PATIENCE}" \
    --early_stop_min_delta "${AC_EARLY_STOP_MIN_DELTA}" \
    "${EXTRA_ARGS[@]}"

date
echo "=== Done. Output: ${OUT_DIR} ==="
