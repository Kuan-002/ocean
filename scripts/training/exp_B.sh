#!/bin/bash

python ./main.py --env_path './scripts/envs/.envB_ch3' --out_subpath './out/exp_B_ch3/' --latest --reload_sa_checkpoint_path './out/sa_ch3_64/checkpoints/sa' --type 'e2e_em'
python ./main.py --env_path './scripts/envs/.envB_ch7' --out_subpath './out/exp_B_ch7/' --latest --reload_sa_checkpoint_path './out/sa_ch7_64/checkpoints/sa' --type 'e2e_em'
python ./main.py --env_path './scripts/envs/.envB_mds' --out_subpath './out/exp_B_mds/' --latest --reload_sa_checkpoint_path './out/sa_mds_64/checkpoints/sa' --type 'e2e_em'
