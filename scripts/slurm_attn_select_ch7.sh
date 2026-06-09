#!/bin/bash
#SBATCH --job-name=rnn_attn_sel
#SBATCH --output=logs/rnn_ch7_attn_select_%j.out
#SBATCH --error=logs/rnn_ch7_attn_select_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

REPO=/vol/biomedic3/kw1025/ocean
ENV_FILE=$REPO/scripts/envs/.envA_ch7_attn_select
SA_CKPT=$REPO/out/sa_ch7_64_4/checkpoints/sa
CLS_CKPT=$REPO/out/cls_ch7_1/deepsets_classifier_best.pt
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$REPO/out/rnn_ch7_attn_select_${TIMESTAMP}

cd $REPO
mkdir -p logs

/vol/biomedic3/kw1025/oceanvenv/bin/python scripts/run_train_rnn_variant.py \
    --env_path       "$ENV_FILE"  \
    --out_subpath    "$OUT_DIR"   \
    --sa_checkpoint  "$SA_CKPT"  \
    --cls_checkpoint "$CLS_CKPT"
