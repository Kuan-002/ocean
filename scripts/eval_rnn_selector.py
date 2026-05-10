#!/usr/bin/env python3
"""Evaluate trained RNN slot selector (frozen SA only; no §4.1/4.2 in forward)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.clevr_hans_dataset import setup_dataloaders
from src.config import Config
from src.explanation.rnn_selector import SlotSelectionPolicyGRU
from src.explanation.rnn_selector_training import SelectorConfig, eval_rnn_selector
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def load_selector(path: str, model: SlotSelectionPolicyGRU, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument("--sa_checkpoint", type=str, required=True)
    parser.add_argument("--selector_checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    device = torch.device(config.device if config.use_gpu else "cpu")

    sa_ckpt = resolve_checkpoint_path(args.sa_checkpoint)
    selector_ckpt = resolve_checkpoint_path(args.selector_checkpoint)
    print(f"SA: {sa_ckpt}")
    print(f"SELECTOR: {selector_ckpt}")

    sa, _ = reconstruct_autoencoder(sa_ckpt, config)
    sa.to(device).eval()
    for p in sa.parameters():
        p.requires_grad = False

    selector = SlotSelectionPolicyGRU(
        slot_dim=config.slot_dim,
        embed_dim=config.rnn_sel_embed_dim,
        hidden_dim=config.rnn_sel_hidden_dim,
        num_slots=config.num_slots,
        num_classes=config.ds_num_classes,
    ).to(device)
    meta = load_selector(selector_ckpt, selector, device)
    if meta:
        print(f"Selector meta: {meta}")
    selector.eval()

    rl_cfg = SelectorConfig(
        num_slots=config.num_slots,
        tau=config.ds_tau,
        epsilon=config.ds_epsilon,
        max_steps=config.rnn_sel_max_steps,
        lambda_len=config.rnn_sel_lambda_len,
        success_reward=config.rnn_sel_success_reward,
        fail_penalty=config.rnn_sel_fail_penalty,
        grpo_group_size=config.rnn_sel_grpo_group_size,
        grpo_beta=config.rnn_sel_grpo_beta,
        grpo_eps=config.rnn_sel_grpo_eps,
        alpha_class=config.rnn_sel_alpha_class,
        max_grad_norm=config.rnn_sel_max_grad_norm,
        grpo_adv_clip=config.rnn_sel_grpo_adv_clip,
        imitation_alpha_class=config.rnn_sel_imitation_alpha_class,
        eval_require_tau_to_stop=config.rnn_sel_eval_require_tau_to_stop,
        eval_disable_conf_early_exit=config.rnn_sel_eval_disable_conf_early_exit,
    )

    split_loader = dataloaders[args.split]
    max_samples = None if args.max_samples < 0 else args.max_samples
    stats = eval_rnn_selector(
        sa,
        selector,
        split_loader,
        device,
        rl_cfg,
        max_samples=max_samples,
    )
    print("\n=== RNN selector metrics ===")
    for k in ["success_rate", "avg_subset_size", "mean_steps", "avg_final_p", "mean_R"]:
        print(f"{k}: {stats[k]:.6f}")


if __name__ == "__main__":
    main()
