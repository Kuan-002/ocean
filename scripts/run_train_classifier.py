#!/usr/bin/env python3
"""Train DeepSetsClassifier (or EnhancedDeepSetsClassifier) on frozen SlotAutoencoder slots."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.clevr_hans_dataset import setup_dataloaders
from src.classification.checkpoints import save_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier, EnhancedDeepSetsClassifier
from src.classification.training import batch_slots, eval_classifier_accuracy
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def train_epoch(sa, clf, loader, optimiser, device, label_smoothing=0.0, grad_clip=1.0):
    clf.train()
    total, n = 0.0, 0
    for images, _, labels in loader:
        labels = labels.to(device)
        slots = batch_slots(sa, images, device)
        logits = clf(slots)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(clf.parameters(), grad_clip)
        optimiser.step()
        total += loss.item() * labels.size(0)
        n += labels.size(0)
    return total / max(n, 1)


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
    parser.add_argument(
        "--enhanced",
        action="store_true",
        default=False,
        help="Use EnhancedDeepSetsClassifier (sum+max aggregation, deeper phi/rho, LayerNorm+GELU).",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clip (<=0 disables).")
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

    if args.enhanced:
        clf = EnhancedDeepSetsClassifier(
            slot_dim=config.slot_dim,
            phi_hidden=config.ds_phi_hidden,
            rho_hidden=config.ds_rho_hidden,
            num_classes=config.ds_num_classes,
            dropout=config.ds_dropout,
        ).to(device)
        print(
            f"[EnhancedDeepSets] phi_hidden={config.ds_phi_hidden}  rho_hidden={config.ds_rho_hidden}"
            f"  dropout={config.ds_dropout}  label_smoothing={config.ds_label_smoothing}"
        )
    else:
        clf = DeepSetsClassifier(
            slot_dim=config.slot_dim,
            phi_hidden=config.ds_phi_hidden,
            rho_hidden=config.ds_rho_hidden,
            num_classes=config.ds_num_classes,
            aggregate=config.ds_aggregate,
        ).to(device)
        print(f"[DeepSetsClassifier] phi_hidden={config.ds_phi_hidden}  rho_hidden={config.ds_rho_hidden}")

    n_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    print(f"Classifier parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(
        clf.parameters(),
        lr=config.ds_cls_lr,
        weight_decay=config.ds_cls_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.ds_cls_epochs, eta_min=config.ds_cls_lr * 0.01
    )

    best_acc = 0.0
    log_every = max(1, config.ds_cls_epochs // 20)
    for epoch in tqdm(range(config.ds_cls_epochs), desc="DeepSets"):
        loss = train_epoch(
            sa, clf, dataloaders["train"], optimiser, device,
            label_smoothing=config.ds_label_smoothing,
            grad_clip=args.grad_clip,
        )
        scheduler.step()
        acc = eval_classifier_accuracy(sa, clf, dataloaders["val"], device)
        if acc > best_acc:
            best_acc = acc
            path = os.path.join(config.out_subpath, "deepsets_classifier_best.pt")
            save_deepsets_checkpoint(
                path,
                clf.state_dict(),
                meta={
                    "val_accuracy": acc,
                    "epoch": epoch,
                    "sa_checkpoint": sa_ckpt,
                    "enhanced": args.enhanced,
                },
            )
        if epoch % log_every == 0 or epoch == config.ds_cls_epochs - 1:
            print(
                f"  epoch {epoch:4d}  loss={loss:.4f}  val_acc={acc:.4f}"
                f"  best={best_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )

    final_path = os.path.join(config.out_subpath, "deepsets_classifier_last.pt")
    save_deepsets_checkpoint(
        final_path,
        clf.state_dict(),
        meta={"val_accuracy": acc, "sa_checkpoint": sa_ckpt, "enhanced": args.enhanced},
    )
    print(f"\nBest val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
