#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clevr_hans_dataset import setup_dataloaders  # noqa: E402
from src.config import Config  # noqa: E402
from src.runtime import seed_all  # noqa: E402
from src.slot_autoencoder import SlotAutoencoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ocean Slot Attention on CLEVR-Hans.")
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--out_subpath", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: SlotAutoencoder,
    loader,
    device: torch.device,
    *,
    train: bool,
    max_batches: int,
    max_norm: float,
) -> float:
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    for batch_index, (images, _, _) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(device)
        if train:
            model.optimiser.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            reconstruction, _, _, _, _ = model(images)
            loss = model.loss(images, reconstruction)
        if train:
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            model.optimiser.step()
        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("No samples were processed; check the dataset and batch limits")
    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    config = Config(args.env_path, args.out_subpath)
    if config.dataset != "ch":
        raise ValueError(f"This minimal trainer supports CLEVR-Hans, got {config.dataset!r}")
    if args.num_workers is not None:
        config.dataset_num_workers = args.num_workers
    if args.max_train_batches is not None:
        config.max_train_batches = args.max_train_batches
    if args.max_val_batches is not None:
        config.max_val_batches = args.max_val_batches
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")

    seed_all(config.seed, config.deterministic)
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
    loaders = setup_dataloaders(config)
    model = SlotAutoencoder(config).to(device)
    best_loss = float("inf")
    if args.resume is not None:
        best_loss = float(model.load(str(args.resume)))

    out_dir = Path(config.out_subpath)
    checkpoint_dir = Path(config.checkpoint_path_sa)
    epochs = config.epochs if args.epochs is None else args.epochs
    history = []
    for epoch in range(epochs):
        train_loss = run_epoch(
            model,
            loaders["train"],
            device,
            train=True,
            max_batches=config.max_train_batches,
            max_norm=config.max_norm,
        )
        val_loss = run_epoch(
            model,
            loaders["val"],
            device,
            train=False,
            max_batches=config.max_val_batches,
            max_norm=config.max_norm,
        )
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            model.save(str(checkpoint_dir / "best_ckpt.pt"), best_loss)
        model.save(str(checkpoint_dir / "last_ckpt.pt"), best_loss)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss": round(val_loss, 8),
            "best_val_loss": round(best_loss, 8),
        }
        history.append(row)
        (out_dir / "sa_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)

    print(str(checkpoint_dir / "best_ckpt.pt"), flush=True)


if __name__ == "__main__":
    main()
