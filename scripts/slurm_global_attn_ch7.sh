#!/bin/bash
#SBATCH --job-name=rnn_global_attn
#SBATCH --output=logs/rnn_global_attn_%j.out
#SBATCH --error=logs/rnn_global_attn_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# RNN slot selector: GlobalAttn head (h0=0, class-head sees ALL K slots) — CLEVR-Hans7
#
# Architecture: SlotSelectionPolicyGlobalAttn
#   - Action GRU: h0 = 0 (never warm-started)
#   - Class head: cross-attn(h_t, ALL K slot embeddings)  [no selection mask]
#   - Slot-count reward: t_star ≈ class-required object count [2,2,3,5,3,3,2]

REPO=/vol/biomedic3/kw1025/ocean
ENV_FILE=$REPO/scripts/envs/.envA_ch7_global_attn
SA_CKPT=$REPO/out/sa_ch7_64_4/checkpoints/sa
CLS_CKPT=$REPO/out/cls_ch7_1/deepsets_classifier_best.pt
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$REPO/out/rnn_ch7_global_attn_${TIMESTAMP}

cd $REPO
mkdir -p logs

/vol/biomedic3/kw1025/oceanvenv/bin/python scripts/run_train_rnn_variant.py \
    --env_path    "$ENV_FILE"  \
    --out_subpath "$OUT_DIR"   \
    --sa_checkpoint  "$SA_CKPT"  \
    --cls_checkpoint "$CLS_CKPT"
