#!/usr/bin/env python3
"""Train pure classification baselines for CLEVR-Hans.

Baselines:
  - resnet18: image -> class
  - slot_mlp: frozen SlotAttention slots -> class
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from src.clevr_hans_dataset import setup_dataloaders
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, seed_all


class SlotMLPClassifier(nn.Module):
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        pooling: str = "sum",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        if pooling == "flatten":
            in_dim = num_slots * slot_dim
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.phi = None
            self.rho = None
            return

        self.phi = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.net = None

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if self.pooling == "flatten":
            return self.net(slots.flatten(start_dim=1))
        slot_features = self.phi(slots)
        if self.pooling == "mean":
            pooled = slot_features.mean(dim=1)
        elif self.pooling == "sum":
            pooled = slot_features.sum(dim=1)
        else:
            raise ValueError(f"Unknown slot pooling: {self.pooling}")
        return self.rho(pooled)


def build_resnet18(num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pure classification baselines.")
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--out_subpath", required=True)
    parser.add_argument("--model", choices=["resnet18", "slot_mlp"], required=True)
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("BENCH_EPOCHS", "100")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("BENCH_LR", "0.0003")))
    parser.add_argument("--weight_decay", type=float, default=float(os.getenv("BENCH_WEIGHT_DECAY", "0.0001")))
    parser.add_argument("--hidden_dim", type=int, default=int(os.getenv("BENCH_HIDDEN_DIM", "512")))
    parser.add_argument("--dropout", type=float, default=float(os.getenv("BENCH_DROPOUT", "0.1")))
    parser.add_argument(
        "--slot_pooling",
        choices=["sum", "mean", "flatten"],
        default=os.getenv("BENCH_SLOT_POOLING", "sum"),
        help="SlotMLP aggregation. sum/mean are permutation-invariant; flatten is the old order-sensitive baseline.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=float(os.getenv("BENCH_LABEL_SMOOTHING", "0.0")),
    )
    parser.add_argument("--eval_every", type=int, default=int(os.getenv("BENCH_EVAL_EVERY", "1")))
    parser.add_argument("--eval_batches", type=int, default=int(os.getenv("BENCH_EVAL_BATCHES", "0")))
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=int(os.getenv("BENCH_EARLY_STOP_PATIENCE", "20")),
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=float(os.getenv("BENCH_EARLY_STOP_MIN_DELTA", "0.0001")),
    )
    return parser.parse_args()


def round_logs(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: round_logs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_logs(v) for v in obj]
    return obj


def make_per_image_slot_init_noise(
    images: torch.Tensor,
    num_slots: int,
    slot_dim: int,
    base_seed: int,
) -> torch.Tensor:
    b, _, _, _ = images.shape
    device = images.device
    dtype = images.dtype
    noise = torch.empty(b, num_slots, slot_dim, device=device, dtype=dtype)
    flat = images.detach().reshape(b, -1)
    for i in range(b):
        generator = torch.Generator(device=device)
        image_seed = int(flat[i].sum().item() * 1e6) % (2**31 - 2)
        generator.manual_seed(int((base_seed + image_seed) % (2**31 - 1)))
        noise[i] = torch.randn(num_slots, slot_dim, generator=generator, device=device, dtype=dtype)
    return noise


@torch.no_grad()
def batch_slots(sa, images: torch.Tensor, device: torch.device, config: Config) -> torch.Tensor:
    sa.eval()
    x = images.to(device)
    slot_noise = None
    if config.rnn_sel_sa_deterministic_slots or config.deterministic:
        slot_noise = make_per_image_slot_init_noise(
            x,
            config.num_slots,
            config.slot_dim,
            config.rnn_sel_sa_noise_seed,
        )
    slots, _ = sa.forward_slots_only(x, slot_init_noise=slot_noise)
    return slots


def batch_logits(
    model: nn.Module,
    model_name: str,
    images: torch.Tensor,
    device: torch.device,
    config: Config,
    sa: nn.Module | None = None,
) -> torch.Tensor:
    if model_name == "resnet18":
        return model(images.to(device))
    if sa is None:
        raise ValueError("slot_mlp requires a frozen SlotAttention model")
    return model(batch_slots(sa, images, device, config))


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    pred = logits.argmax(dim=-1)
    correct = (pred == labels).sum().item()
    per_class_correct = torch.zeros(num_classes, dtype=torch.float64)
    per_class_total = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        mask = labels == cls
        per_class_total[cls] += mask.sum().item()
        if mask.any():
            per_class_correct[cls] += (pred[mask] == labels[mask]).sum().item()
    return correct, per_class_correct, per_class_total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    model_name: str,
    loader,
    device: torch.device,
    config: Config,
    num_classes: int,
    *,
    sa: nn.Module | None = None,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    per_class_correct = torch.zeros(num_classes, dtype=torch.float64)
    per_class_total = torch.zeros(num_classes, dtype=torch.float64)
    for batch_idx, (images, _, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        labels = labels.to(device)
        logits = batch_logits(model, model_name, images, device, config, sa)
        loss = F.cross_entropy(logits, labels, reduction="sum")
        batch_correct, cls_correct, cls_total = compute_metrics(logits, labels, num_classes)
        total += labels.size(0)
        correct += batch_correct
        loss_sum += loss.item()
        per_class_correct += cls_correct
        per_class_total += cls_total

    out: dict[str, float] = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": float(total),
    }
    for cls in range(num_classes):
        count = per_class_total[cls].item()
        out[f"class_{cls}_accuracy"] = per_class_correct[cls].item() / max(count, 1.0)
        out[f"class_{cls}_count"] = count
    return out


def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader,
    device: torch.device,
    config: Config,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    *,
    sa: nn.Module | None = None,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for images, _, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = batch_logits(model, model_name, images, device, config, sa)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()
        batch_correct, _, _ = compute_metrics(logits.detach(), labels, num_classes)
        total += labels.size(0)
        correct += batch_correct
        loss_sum += loss.item() * labels.size(0)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": float(total),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    meta: dict,
    val: dict[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "meta": round_logs(meta),
    }
    if val is not None:
        payload["val"] = round_logs(val)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.model == "slot_mlp" and not args.sa_checkpoint:
        raise ValueError("--sa_checkpoint is required for --model slot_mlp")

    config = Config(args.env_path, args.out_subpath)
    out_dir = Path(config.out_subpath)
    seed_all(config.seed, config.deterministic)
    device = torch.device("cuda" if config.use_gpu else "cpu")
    num_classes = len(config.labels)

    if config.dataset == "ch":
        loaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        loaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    sa = None
    sa_best_loss = None
    if args.model == "slot_mlp":
        sa, sa_best_loss = reconstruct_autoencoder(args.sa_checkpoint, config)
        sa.to(device)
        sa.eval()
        for p in sa.parameters():
            p.requires_grad = False
        model = SlotMLPClassifier(
            config.num_slots,
            config.slot_dim,
            num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pooling=args.slot_pooling,
        )
    else:
        model = build_resnet18(num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    meta = {
        "args": vars(args),
        "num_classes": num_classes,
        "labels": config.labels,
        "sa_best_loss": None if sa_best_loss is None else float(sa_best_loss),
    }
    (out_dir / "benchmark_meta.json").write_text(
        json.dumps(round_logs(meta), indent=2),
        encoding="utf-8",
    )

    best_val = -1.0
    best_epoch = -1
    evals_since_improvement = 0
    history = []
    for epoch in range(args.epochs):
        train = train_one_epoch(
            model,
            args.model,
            loaders["train"],
            device,
            config,
            optimizer,
            num_classes,
            sa=sa,
            label_smoothing=args.label_smoothing,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}}

        if (epoch + 1) % args.eval_every == 0:
            val = evaluate(
                model,
                args.model,
                loaders["val"],
                device,
                config,
                num_classes,
                sa=sa,
                max_batches=args.eval_batches,
            )
            row.update({f"val_{k}": v for k, v in val.items()})
            improved = val["accuracy"] > best_val + args.early_stop_min_delta
            if improved:
                best_val = val["accuracy"]
                best_epoch = epoch
                evals_since_improvement = 0
                save_checkpoint(out_dir / "benchmark_best.pt", model, optimizer, epoch, meta, val)
            else:
                evals_since_improvement += 1
            row["best_val_accuracy"] = best_val
            row["best_epoch"] = best_epoch
            row["early_stop_wait"] = evals_since_improvement

        save_checkpoint(out_dir / "benchmark_last.pt", model, optimizer, epoch, meta)
        row = round_logs(row)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)

        if args.early_stop_patience > 0 and evals_since_improvement >= args.early_stop_patience:
            print(
                "Early stopping: "
                f"best_val_accuracy={best_val:.4f} at epoch={best_epoch}, "
                f"wait={evals_since_improvement}.",
                flush=True,
            )
            break

    best_path = out_dir / "benchmark_best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, args.model, loaders["test"], device, config, num_classes, sa=sa)
    test = round_logs(test)
    (out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(json.dumps({"test": test, "best_epoch": best_epoch}, sort_keys=True), flush=True)
    print(f"Done. Best val accuracy: {best_val:.4f}. Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
