#!/bin/bash
#SBATCH --job-name=eval_viz_attn_sel
#SBATCH --output=logs/eval_viz_attn_select_%j.out
#SBATCH --error=logs/eval_viz_attn_select_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

REPO=/vol/biomedic3/kw1025/ocean
ENV_FILE=$REPO/scripts/envs/.envA_ch7_attn_select
SA_CKPT=$REPO/out/sa_ch7_64_4/checkpoints/sa

# Best checkpoint from the attn_select run (job 65396)
SELECTOR=$REPO/out/rnn_ch7_attn_select_20260606_045456/rnn_selector_best.pt
OUT_DIR=$REPO/out/rnn_ch7_attn_select_20260606_045456/test_viz

cd $REPO
mkdir -p logs

/vol/biomedic3/kw1025/oceanvenv/bin/python scripts/eval_visualize_test_global_attn.py \
    --env_path    "$ENV_FILE"  \
    --sa_checkpoint  "$SA_CKPT"  \
    --selector_checkpoint "$SELECTOR" \
    --split test \
    --max_per_class 10 \
    --out_dir "$OUT_DIR"
