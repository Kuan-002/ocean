#!/bin/bash

python ./main.py --env_path './scripts/envs/.envC_ch3' --out_subpath './out/exp_C_ch3/' --latest --reload_sa_checkpoint_path './out/sa_ch3_128/checkpoints/sa' --type 'e2e_em'
python ./main.py --env_path './scripts/envs/.envC_ch7' --out_subpath './out/exp_C_ch7/' --latest --reload_sa_checkpoint_path './out/sa_ch7_128/checkpoints/sa' --type 'e2e_em'
