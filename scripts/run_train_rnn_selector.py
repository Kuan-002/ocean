#!/usr/bin/env python3
"""Train sequential slot selector: imitation (action + class_head) then GRPO; frozen SA & DeepSets."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from src.clevr_hans_dataset import setup_dataloaders
from src.classification.checkpoints import load_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier
from src.config import Config
from src.explanation.rnn_selector import SlotSelectionPolicyGRU
from src.explanation.rnn_selector_training import (
    SelectorConfig,
    eval_rnn_selector,
    train_grpo_epoch,
    train_imitation_epoch,
)
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all

DEFAULT_SA = "/homes/kw1025/ocean/out/sa_ch3_64_1/checkpoints/sa/999_ckpt.pt"
DEFAULT_CLS = "/homes/kw1025/ocean/out/deepsets_classification_1/deepsets_classifier_best.pt"


def save_selector_checkpoint(path: str, model: SlotSelectionPolicyGRU, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="RNN slot selector: imitation + GRPO")
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument("--out_subpath", type=str, default="./out/rnn_selector/")
    parser.add_argument("--sa_checkpoint", type=str, default=DEFAULT_SA)
    parser.add_argument("--cls_checkpoint", type=str, default=DEFAULT_CLS)
    args = parser.parse_args()

    config = Config(args.env_path, args.out_subpath)
    seed_all(config.seed, config.deterministic)

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    device = torch.device(config.device if config.use_gpu else "cpu")

    sa_ckpt = resolve_checkpoint_path(args.sa_checkpoint)
    cls_ckpt = resolve_checkpoint_path(args.cls_checkpoint)
    print(f"SA checkpoint: {sa_ckpt}")
    print(f"Classifier checkpoint: {cls_ckpt}")

    sa, _ = reconstruct_autoencoder(sa_ckpt, config)
    sa.to(device)
    sa.eval()
    for p in sa.parameters():
        p.requires_grad = False

    clf = DeepSetsClassifier(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)
    clf.load_state_dict(load_deepsets_checkpoint(cls_ckpt, map_location=device)[0])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    policy = SlotSelectionPolicyGRU(
        slot_dim=config.slot_dim,
        embed_dim=config.rnn_sel_embed_dim,
        hidden_dim=config.rnn_sel_hidden_dim,
        num_slots=config.num_slots,
        num_classes=config.ds_num_classes,
    ).to(device)

    sel_cfg = SelectorConfig(
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

    optimiser = torch.optim.AdamW(
        policy.parameters(), lr=config.rnn_sel_lr, weight_decay=config.rnn_sel_weight_decay
    )

    os.makedirs(config.out_subpath, exist_ok=True)

    print(
        f"Imitation: {config.rnn_sel_imitation_epochs} epochs "
        f"(imitation_alpha_class={sel_cfg.imitation_alpha_class})"
    )
    for epoch in tqdm(range(config.rnn_sel_imitation_epochs), desc="Imitation"):
        im_stats = train_imitation_epoch(
            sa, clf, policy, dataloaders["train"], optimiser, device, sel_cfg
        )
        cls_part = im_stats.get("imitation_cls_loss", 0.0)
        print(
            f"imit_epoch={epoch} imitation_loss={im_stats['imitation_loss']:.4f} "
            f"imitation_cls_loss={cls_part:.4f}"
        )

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    save_selector_checkpoint(
        os.path.join(config.out_subpath, "rnn_selector_pi_ref.pt"),
        policy,
        {
            "phase": "after_imitation",
            "sa_checkpoint": sa_ckpt,
            "cls_checkpoint": cls_ckpt,
        },
    )

    best_success = -1.0
    print(f"GRPO: {config.rnn_sel_epochs} epochs, G={sel_cfg.grpo_group_size}")
    for epoch in tqdm(range(config.rnn_sel_epochs), desc="GRPO"):
        train_stats = train_grpo_epoch(
            sa, clf, policy, ref_policy, dataloaders["train"], optimiser, device, sel_cfg
        )

        val_stats = eval_rnn_selector(
            sa,
            policy,
            dataloaders["val"],
            device,
            sel_cfg,
            max_samples=config.rnn_sel_eval_samples,
        )
        print(
            f"epoch={epoch} loss={train_stats['loss']:.4f} "
            f"L_grpo={train_stats['L_grpo']:.4f} L_class={train_stats['L_class']:.4f} "
            f"train_succ={train_stats['success_rate']:.4f} train_k={train_stats['avg_subset_size']:.2f} "
            f"val_succ={val_stats['success_rate']:.4f} val_k={val_stats['avg_subset_size']:.2f} "
            f"val_p={val_stats['avg_final_p']:.4f}"
        )
        if val_stats["success_rate"] > best_success:
            best_success = val_stats["success_rate"]
            save_selector_checkpoint(
                os.path.join(config.out_subpath, "rnn_selector_best.pt"),
                policy,
                {
                    "epoch": epoch,
                    "val_stats": val_stats,
                    "sa_checkpoint": sa_ckpt,
                    "cls_checkpoint": cls_ckpt,
                },
            )

    save_selector_checkpoint(
        os.path.join(config.out_subpath, "rnn_selector_last.pt"),
        policy,
        {"best_success_rate": best_success},
    )
    print(f"Best val success_rate: {best_success:.4f}")


if __name__ == "__main__":
    main()
