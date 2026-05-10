#!/usr/bin/env python3
"""Print greedy action sequences (slot indices + STOP) for a few val images."""

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
from src.explanation.rnn_selector_training import SelectorConfig, run_grpo_rollout
from src.classification.training import batch_slots
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def rollout_end_reason(r: int, out: dict) -> str:
    if bool(out["stopped_with_stop"][r].item()):
        return "STOP_argmax"
    if bool(out["timed_out"][r].item()):
        return "max_steps_no_done"
    return "eval_early_exit_p>=tau"


def decode_trace(
    steps: list[torch.Tensor], stop_idx: int, batch_row: int
) -> tuple[list[int], list[str]]:
    seq: list[int] = []
    labels: list[str] = []
    for t, a in enumerate(steps):
        act = int(a[batch_row].item())
        seq.append(act)
        if act == stop_idx:
            labels.append("STOP")
            break
        labels.append(f"slot_{act}")
    return seq, labels


def build_sel_cfg(config: Config, require_tau_to_stop: bool, no_conf_early_exit: bool) -> SelectorConfig:
    return SelectorConfig(
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
        eval_require_tau_to_stop=require_tau_to_stop or config.rnn_sel_eval_require_tau_to_stop,
        eval_disable_conf_early_exit=no_conf_early_exit
        or config.rnn_sel_eval_disable_conf_early_exit,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", type=str, default="scripts/envs/.envA_ch3_rnn")
    p.add_argument("--sa_checkpoint", type=str, required=True)
    p.add_argument("--selector_checkpoint", type=str, required=True)
    p.add_argument("--n_batches", type=int, default=1)
    p.add_argument("--rows", type=int, default=8, help="Print first N rows of each batch")
    p.add_argument(
        "--require_tau_to_stop",
        action="store_true",
        help="Force eval_require_tau_to_stop=True for this run.",
    )
    p.add_argument(
        "--no_conf_early_exit",
        action="store_true",
        help="Match training: disable p>=tau early exit after a slot (eval_disable_conf_early_exit).",
    )
    args = p.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    device = torch.device(config.device if config.use_gpu else "cpu")

    dls = setup_dataloaders(config)
    sa, _ = reconstruct_autoencoder(resolve_checkpoint_path(args.sa_checkpoint), config)
    sa.to(device).eval()

    policy = SlotSelectionPolicyGRU(
        slot_dim=config.slot_dim,
        embed_dim=config.rnn_sel_embed_dim,
        hidden_dim=config.rnn_sel_hidden_dim,
        num_slots=config.num_slots,
        num_classes=config.ds_num_classes,
    ).to(device)
    ckpt = torch.load(
        resolve_checkpoint_path(args.selector_checkpoint), map_location=device, weights_only=True
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    sel_cfg = build_sel_cfg(config, args.require_tau_to_stop, args.no_conf_early_exit)
    ref = policy
    for bi, batch in enumerate(dls["val"]):
        if bi >= args.n_batches:
            break
        images = batch[0].to(device)
        gt = batch[2].to(device)
        slots = batch_slots(sa, images, device)
        trace: list[torch.Tensor] = []
        out = run_grpo_rollout(policy, ref, slots, gt, sel_cfg, train=False, action_trace=trace)
        n_sel = out["selected_mask"].float().sum(-1)
        print(f"\n=== val batch {bi} B={images.size(0)} ===")
        for r in range(min(args.rows, images.size(0))):
            _, lab = decode_trace(trace, policy.stop_idx, r)
            why = rollout_end_reason(r, out)
            print(
                f"  row{r}: k={int(n_sel[r].item())} p_conf={out['p_conf'][r].item():.4f} "
                f"gt={int(gt[r].item())} end={why} trace={' -> '.join(lab)}"
            )
        mean_k = n_sel.mean().item()
        print(f"  mean_k (slots selected) = {mean_k:.3f}")


if __name__ == "__main__":
    main()
