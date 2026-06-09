#!/usr/bin/env bash
# RNN slot selector: GlobalAttn head (h0=0 + class-head sees ALL slots) — CLEVR-Hans7.
#
# head_type: all_slots_attn  (SlotSelectionPolicyGlobalAttn)
#   - Action GRU: h0 = 0, sequential slot selection (unchanged)
#   - Class head: cross-attn over ALL K=11 slot embeddings (no selection mask)
#   - Slot-count reward: t_star ≈ class-specific object count [2,2,3,5,3,3,2]
#
# Prerequisites: SA + DeepSets checkpoints already trained.
#
# Usage:
#   cd /path/to/ocean && bash scripts/train_rnn_global_attn_ch7.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_FILE="${ENV_FILE:-$REPO/scripts/envs/.envA_ch7_global_attn}"
PYTHON="${PYTHON:-python3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$REPO/out/rnn_ch7_global_attn_${TIMESTAMP}}"
SA_CKPT="${SA_CKPT:-$REPO/out/sa_ch7_64_4/checkpoints/sa}"
CLS_CKPT="${CLS_CKPT:-$REPO/out/cls_ch7_1/deepsets_classifier_best.pt}"

echo "ENV_FILE : $ENV_FILE"
echo "OUT      : $OUT"
echo "SA       : $SA_CKPT"
echo "CLS      : $CLS_CKPT"

exec "$PYTHON" scripts/run_train_rnn_variant.py \
  --env_path    "$ENV_FILE" \
  --out_subpath "$OUT" \
  --sa_checkpoint  "$SA_CKPT" \
  --cls_checkpoint "$CLS_CKPT"
