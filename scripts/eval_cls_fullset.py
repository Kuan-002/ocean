#!/usr/bin/env python3
"""Evaluate DeepSets classifier on full slot set (mask=None) for any split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.classification.checkpoints import load_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier, EnhancedDeepSetsClassifier
from src.classification.training import batch_slots
from src.clevr_hans_dataset import setup_dataloaders
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate full-set DeepSets classifier")
    parser.add_argument("--env_path",       type=str, required=True)
    parser.add_argument("--sa_checkpoint",  type=str, required=True)
    parser.add_argument("--cls_checkpoint", type=str, required=True)
    parser.add_argument("--split",          type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    device = torch.device(config.device if config.use_gpu else "cpu")

    if config.dataset == "ch":
        loaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        loaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    loader = loaders[args.split]
    num_classes = len(config.labels)

    sa_path = resolve_checkpoint_path(args.sa_checkpoint)
    cls_path = resolve_checkpoint_path(args.cls_checkpoint)
    print(f"SA:  {sa_path}")
    print(f"CLS: {cls_path}")
    print(f"Split: {args.split}  |  classes: {num_classes}")

    sa, _ = reconstruct_autoencoder(sa_path, config)
    sa.to(device).eval()
    for p in sa.parameters():
        p.requires_grad_(False)

    cls_sd, cls_meta = load_deepsets_checkpoint(cls_path, map_location=device)
    if cls_meta.get("enhanced", False):
        clf = EnhancedDeepSetsClassifier(
            slot_dim=config.slot_dim,
            phi_hidden=config.ds_phi_hidden,
            rho_hidden=config.ds_rho_hidden,
            num_classes=config.ds_num_classes,
            dropout=config.ds_dropout,
        ).to(device)
    else:
        clf = DeepSetsClassifier(
            slot_dim=config.slot_dim,
            phi_hidden=config.ds_phi_hidden,
            rho_hidden=config.ds_rho_hidden,
            num_classes=config.ds_num_classes,
            aggregate=config.ds_aggregate,
        ).to(device)
    clf.load_state_dict(cls_sd)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)

    correct_per_cls = torch.zeros(num_classes, dtype=torch.long)
    total_per_cls   = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, _, labels in loader:
            labels = labels.to(device)
            slots  = batch_slots(sa, images, device)          # [B, K, D]
            logits = clf(slots, None)                          # full-set, no mask
            preds  = logits.argmax(dim=-1)
            for c in range(num_classes):
                mask = labels == c
                correct_per_cls[c] += (preds[mask] == c).sum().cpu()
                total_per_cls[c]   += mask.sum().cpu()

    overall_acc = correct_per_cls.sum().item() / total_per_cls.sum().item()

    print(f"\n{'Class':>6}  {'Correct':>8}  {'Total':>8}  {'Acc':>8}")
    print("-" * 38)
    for c in range(num_classes):
        acc = correct_per_cls[c].item() / max(total_per_cls[c].item(), 1)
        print(f"  {c:>4}  {correct_per_cls[c].item():>8}  {total_per_cls[c].item():>8}  {acc:>8.4f}")
    print("-" * 38)
    print(f"  {'ALL':>4}  {correct_per_cls.sum().item():>8}  {total_per_cls.sum().item():>8}  {overall_acc:>8.4f}")


if __name__ == "__main__":
    main()
