#!/usr/bin/env python3
"""Evaluate trained RNN slot selector (frozen SA only; no §4.1/4.2 in forward)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.classification.checkpoints import load_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier
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
    parser.add_argument(
        "--cls_checkpoint",
        type=str,
        default=None,
        help="Frozen DeepSets for val t* (force_full_rollout only). Default: meta or out/deepsets_classifier_best.pt",
    )
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
        use_stop_action=not config.rnn_sel_force_full_rollout,
    ).to(device)
    meta = load_selector(selector_ckpt, selector, device)
    if meta:
        print(f"Selector meta: {meta}")

    clf = None
    if config.rnn_sel_force_full_rollout:
        cls_path = args.cls_checkpoint
        if cls_path is None:
            cls_path = meta.get("cls_checkpoint") if meta else None
        if cls_path is None:
            cls_path = str(ROOT / "out" / "deepsets_classifier_best.pt")
        cls_path = resolve_checkpoint_path(cls_path)
        print(f"Classifier (metrics only): {cls_path}")
        clf = DeepSetsClassifier(
            slot_dim=config.slot_dim,
            phi_hidden=config.ds_phi_hidden,
            rho_hidden=config.ds_rho_hidden,
            num_classes=config.ds_num_classes,
            aggregate=config.ds_aggregate,
        ).to(device)
        clf.load_state_dict(load_deepsets_checkpoint(cls_path, map_location=device)[0])
        clf.eval()
        for p in clf.parameters():
            p.requires_grad = False

    selector.eval()

    slot_kw: dict = {"num_slots": config.num_slots, "slot_dim": config.slot_dim}
    if config.rnn_sel_sa_deterministic_slots:
        slot_kw["sa_deterministic"] = True
        slot_kw["sa_noise_seed"] = config.rnn_sel_sa_noise_seed

    rl_cfg = SelectorConfig(
        num_slots=config.num_slots,
        tau=config.ds_tau,
        epsilon=config.ds_epsilon,
        max_steps=config.rnn_sel_max_steps,
        lambda_len=config.rnn_sel_lambda_len,
        force_full_rollout=config.rnn_sel_force_full_rollout,
        global_init=config.rnn_sel_global_init,
        area_weight=config.rnn_sel_area_weight,
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
        tstar_p_full_scale=config.rnn_sel_tstar_p_full_scale,
        reward_cls_acc_weight=config.rnn_sel_reward_cls_acc_weight,
        reward_cls_ce_bonus=config.rnn_sel_reward_cls_ce_bonus,
        grpo_class_teacher_kl=config.rnn_sel_grpo_class_teacher_kl,
        grpo_class_teacher_temp=config.rnn_sel_grpo_class_teacher_temp,
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
        clf=clf,
        slot_opts=slot_kw,
    )
    print("\n=== RNN selector metrics ===")
    for k in ["success_rate", "avg_subset_size", "mean_steps", "avg_final_p", "mean_R", "avg_t_star"]:
        print(f"{k}: {stats[k]:.6f}")


if __name__ == "__main__":
    main()
