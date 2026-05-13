#!/usr/bin/env bash
# RNN slot selector: full-order imitation (action + class_head vs y_hat) + GRPO.
# CLEVR-Hans3 / SA 64 / 11 slots.
# Env: scripts/envs/.envA_ch3_rnn (see RNN_SEL_FORCE_FULL_ROLLOUT, RNN_SEL_AREA_WEIGHT).
# Prerequisites: SA + DeepSets checkpoints; DATASET_PATH in env.
#
# Eval best checkpoint (after training):
#   python scripts/eval_rnn_selector.py \
#     --env_path scripts/envs/.envA_ch3_rnn \
#     --sa_checkpoint out/sa.pt \
#     --selector_checkpoint <OUT_DIR>/rnn_selector_best.pt \
#     --split val
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_FILE="${ENV_FILE:-scripts/envs/.envA_ch3_rnn}"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$REPO/out/rnn_selector_ch3_$(date +%Y%m%d_%H%M%S)}"
SA_CKPT="${SA_CKPT:-out/sa.pt}"
CLS_CKPT="${CLS_CKPT:-out/deepsets_classifier_best.pt}"

exec "$PYTHON" scripts/run_train_rnn_selector.py \
  --env_path "$ENV_FILE" \
  --out_subpath "$OUT" \
  --sa_checkpoint "$SA_CKPT" \
  --cls_checkpoint "$CLS_CKPT"
