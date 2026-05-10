#!/usr/bin/env python3
"""Train DeepSetsSubsetEvaluator via KL distillation to a frozen classifier (plan §4.2)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from src.clevr_hans_dataset import setup_dataloaders
from src.classification.checkpoints import load_deepsets_checkpoint, save_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier, DeepSetsSubsetEvaluator
from src.classification.training import train_subset_evaluator_epoch
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument(
        "--out_subpath",
        type=str,
        default="./out/deepsets_subset_eval/",
    )
    parser.add_argument(
        "--sa_checkpoint",
        type=str,
        required=True,
        help="SA .pt file or directory of .pt (latest used).",
    )
    parser.add_argument(
        "--cls_checkpoint",
        type=str,
        required=True,
        help="Frozen DeepSetsClassifier: .pt file or directory of .pt (latest used).",
    )
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
    print(f"Using SA checkpoint: {sa_ckpt}")
    print(f"Using classifier checkpoint: {cls_ckpt}")
    sa, _ = reconstruct_autoencoder(sa_ckpt, config)
    sa.to(device)
    for p in sa.parameters():
        p.requires_grad = False

    teacher = DeepSetsClassifier(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)
    state, _ = load_deepsets_checkpoint(cls_ckpt, map_location=device)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = DeepSetsSubsetEvaluator(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)

    optimiser = torch.optim.AdamW(
        student.parameters(),
        lr=config.ds_subset_lr,
        weight_decay=config.ds_subset_weight_decay,
    )

    best_loss = float("inf")
    last_loss = 0.0
    for epoch in tqdm(range(config.ds_subset_epochs), desc="SubsetEvaluator KL"):
        last_loss = train_subset_evaluator_epoch(
            sa,
            teacher,
            student,
            dataloaders["train"],
            optimiser,
            device,
            num_slots=config.num_slots,
            temperature=config.ds_distill_temperature,
        )
        if last_loss < best_loss:
            best_loss = last_loss
            path = os.path.join(config.out_subpath, "deepsets_subset_evaluator_best.pt")
            save_deepsets_checkpoint(
                path,
                student.state_dict(),
                meta={
                    "kl_loss": last_loss,
                    "epoch": epoch,
                    "sa_checkpoint": sa_ckpt,
                    "cls_checkpoint": cls_ckpt,
                    "temperature": config.ds_distill_temperature,
                },
            )

    save_deepsets_checkpoint(
        os.path.join(config.out_subpath, "deepsets_subset_evaluator_last.pt"),
        student.state_dict(),
        meta={
            "kl_loss": last_loss,
            "sa_checkpoint": sa_ckpt,
            "cls_checkpoint": cls_ckpt,
        },
    )
    print(f"Best train KL: {best_loss:.6f}")


if __name__ == "__main__":
    main()
