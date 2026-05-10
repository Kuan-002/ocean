#!/usr/bin/env python3
"""Train DeepSetsClassifier on frozen SlotAutoencoder slots (plan §4.1)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from src.clevr_hans_dataset import setup_dataloaders
from src.classification.checkpoints import save_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier
from src.classification.training import eval_classifier_accuracy, train_classifier_epoch
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default=".env")
    parser.add_argument(
        "--out_subpath",
        type=str,
        default="./out/deepsets_classification/",
        help="Run directory (Config copies .env and creates checkpoints/).",
    )
    parser.add_argument(
        "--sa_checkpoint",
        type=str,
        required=True,
        help="Frozen SlotAutoencoder: a .pt file or a directory of .pt (uses latest by ctime).",
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
    print(f"Using SA checkpoint: {sa_ckpt}")
    sa, _ = reconstruct_autoencoder(sa_ckpt, config)
    sa.to(device)
    for p in sa.parameters():
        p.requires_grad = False

    clf = DeepSetsClassifier(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)

    optimiser = torch.optim.AdamW(
        clf.parameters(),
        lr=config.ds_cls_lr,
        weight_decay=config.ds_cls_weight_decay,
    )

    best_acc = 0.0
    for epoch in tqdm(range(config.ds_cls_epochs), desc="DeepSetsClassifier"):
        train_classifier_epoch(sa, clf, dataloaders["train"], optimiser, device)
        acc = eval_classifier_accuracy(sa, clf, dataloaders["val"], device)
        if acc > best_acc:
            best_acc = acc
            path = os.path.join(config.out_subpath, "deepsets_classifier_best.pt")
            save_deepsets_checkpoint(
                path,
                clf.state_dict(),
                meta={"val_accuracy": acc, "epoch": epoch, "sa_checkpoint": sa_ckpt},
            )

    final_path = os.path.join(config.out_subpath, "deepsets_classifier_last.pt")
    save_deepsets_checkpoint(
        final_path,
        clf.state_dict(),
        meta={"val_accuracy": acc, "sa_checkpoint": sa_ckpt},
    )
    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
