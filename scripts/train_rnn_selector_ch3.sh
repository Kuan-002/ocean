#!/usr/bin/env bash
# RNN slot selector: imitation (action + class_head vs y_hat) + GRPO. CLEVR-Hans3 / SA 64 / 11 slots.
# Env: scripts/envs/.envA_ch3_rnn (see RNN_SEL_IMITATION_ALPHA_CLASS, RNN_SEL_EVAL_DISABLE_CONF_EARLY_EXIT).
# Prerequisites: SA + DeepSets checkpoints; DATASET_PATH in env.
#
# Eval best checkpoint (after training):
#   python scripts/eval_rnn_selector.py \
#     --env_path scripts/envs/.envA_ch3_rnn \
#     --sa_checkpoint out/sa_ch3_64_1/checkpoints/sa/999_ckpt.pt \
#     --selector_checkpoint <OUT_DIR>/rnn_selector_best.pt \
#     --split val
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_FILE="${ENV_FILE:-scripts/envs/.envA_ch3_rnn}"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$REPO/out/rnn_selector_ch3_$(date +%Y%m%d_%H%M%S)}"
SA_CKPT="${SA_CKPT:-out/sa_ch3_64_1/checkpoints/sa/999_ckpt.pt}"
CLS_CKPT="${CLS_CKPT:-out/deepsets_classification_1/deepsets_classifier_best.pt}"

exec "$PYTHON" scripts/run_train_rnn_selector.py \
  --env_path "$ENV_FILE" \
  --out_subpath "$OUT" \
  --sa_checkpoint "$SA_CKPT" \
  --cls_checkpoint "$CLS_CKPT"
