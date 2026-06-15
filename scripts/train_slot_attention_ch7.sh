#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vol/biomedic3/kw1025/ocean}
VENV=${VENV:-/vol/biomedic3/kw1025/oceanvenv}
ENV_FILE=${ENV_FILE:-${REPO}/scripts/envs/.env_sa_ch7_64}
RUN_NAME=${RUN_NAME:-sa_ch7_64_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${REPO}/out/${RUN_NAME}}
RELOAD_SA=${RELOAD_SA:-}

cd "$REPO"
mkdir -p "$REPO/logs" "$REPO/out" "$REPO/.matplotlib"

source "$VENV/bin/activate"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export DEVICE="${DEVICE:-cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO}/.matplotlib}"

echo "Repo: $REPO"
echo "Env: $ENV_FILE"
echo "Out: $OUT_DIR"
echo "Reload SA: ${RELOAD_SA:-none}"
date

ARGS=(
  --type only_sa
  --env_path "$ENV_FILE"
  --out_subpath "$OUT_DIR"
)

if [[ -n "$RELOAD_SA" ]]; then
  ARGS+=(--reload_sa_checkpoint_path "$RELOAD_SA")
fi

python main.py "${ARGS[@]}"

date
echo "Best checkpoint: ${OUT_DIR}/checkpoints/sa/best_ckpt.pt"
