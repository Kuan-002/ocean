#!/usr/bin/env bash
# =============================================================================
# CLEVR-Hans3 pipeline: Slot Attention (500 ep) -> DeepSets classifier (500 ep)
# -> RNN slot selector (imitation + GRPO). Each stage reads checkpoints from the
# previous stage (paths passed explicitly on the command line).
#
# Usage:
#   cd /path/to/ocean && ./scripts/run_pipeline_ch3.sh
# Optional:
#   ENV_FILE=scripts/envs/.env_pipeline_ch3 PYTHON=/path/to/python ./scripts/run_pipeline_ch3.sh
#   PIPELINE_ROOT=/path/to/custom_root ./scripts/run_pipeline_ch3.sh   # fixed root (must not exist)
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_FILE="${ENV_FILE:-$REPO/scripts/envs/.env_pipeline_ch3}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ -n "${PIPELINE_ROOT:-}" ]]; then
  ROOT="$PIPELINE_ROOT"
else
  ROOT="$REPO/out/pipeline_ch3_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -e "$ROOT" ]]; then
  echo "ERROR: PIPELINE_ROOT already exists: $ROOT" >&2
  exit 1
fi

SA_DIR="$ROOT/01_sa"
CLS_DIR="$ROOT/02_deepsets"
RNN_DIR="$ROOT/03_rnn"

echo "=================================================================="
echo "Pipeline root: $ROOT"
echo "Env:           $ENV_FILE"
echo "Python:        $PYTHON"
echo "=================================================================="

# --- 1) Slot Attention only ---
echo ""
echo "[1/3] Training Slot Attention (EPOCHS=500 in env file)..."
"$PYTHON" "$REPO/main.py" \
  --env_path "$ENV_FILE" \
  --out_subpath "$SA_DIR" \
  --type only_sa

SA_CKPT_DIR="$SA_DIR/checkpoints/sa"
if [[ ! -d "$SA_CKPT_DIR" ]]; then
  echo "ERROR: SA checkpoint directory missing: $SA_CKPT_DIR" >&2
  exit 1
fi
SA_CKPT="$("$PYTHON" -c "
import sys
sys.path.insert(0, '$REPO')
from src.utils import load_latest_checkpoint
print(load_latest_checkpoint('$SA_CKPT_DIR'))
")"
echo "Using SA checkpoint: $SA_CKPT"

# --- 2) DeepSets classifier on frozen SA slots ---
echo ""
echo "[2/3] Training DeepSets classifier (epochs from DS_CLS_EPOCHS in env)..."
"$PYTHON" "$REPO/scripts/run_train_classifier.py" \
  --env_path "$ENV_FILE" \
  --out_subpath "$CLS_DIR" \
  --sa_checkpoint "$SA_CKPT"

CLS_BEST="$(find "$CLS_DIR" -name 'deepsets_classifier_best.pt' -type f 2>/dev/null | head -1 || true)"
if [[ -z "$CLS_BEST" || ! -f "$CLS_BEST" ]]; then
  echo "ERROR: deepsets_classifier_best.pt not found under $CLS_DIR" >&2
  exit 1
fi
echo "Using classifier checkpoint: $CLS_BEST"

# --- 3) RNN selector (imitation + GRPO) ---
echo ""
echo "[3/3] Training RNN slot selector..."
"$PYTHON" "$REPO/scripts/run_train_rnn_selector.py" \
  --env_path "$ENV_FILE" \
  --out_subpath "$RNN_DIR" \
  --sa_checkpoint "$SA_CKPT" \
  --cls_checkpoint "$CLS_BEST"

RNN_BEST="$(find "$RNN_DIR" -name 'rnn_selector_best.pt' -type f 2>/dev/null | head -1 || true)"
RNN_LAST="$(find "$RNN_DIR" -name 'rnn_selector_last.pt' -type f 2>/dev/null | head -1 || true)"
RNN_REF="$(find "$RNN_DIR" -name 'rnn_selector_pi_ref.pt' -type f 2>/dev/null | head -1 || true)"

MANIFEST="$ROOT/MANIFEST.txt"
{
  echo "pipeline_root=$ROOT"
  echo "env_file=$ENV_FILE"
  echo "sa_checkpoint=$SA_CKPT"
  echo "cls_checkpoint=$CLS_BEST"
  echo "rnn_pi_ref=$RNN_REF"
  echo "rnn_best=$RNN_BEST"
  echo "rnn_last=$RNN_LAST"
  echo ""
  echo "Eval RNN (val):"
  echo "  $PYTHON $REPO/scripts/eval_rnn_selector.py \\"
  echo "    --env_path $ENV_FILE \\"
  echo "    --sa_checkpoint $SA_CKPT \\"
  echo "    --selector_checkpoint ${RNN_BEST:-$RNN_LAST} \\"
  echo "    --split val"
} | tee "$MANIFEST"

echo ""
echo "Done. Summary written to: $MANIFEST"
