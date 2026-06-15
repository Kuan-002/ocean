#!/usr/bin/env python3
"""Evaluate a trained Slot Selector V1 AC checkpoint with multiple tau values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clevr_hans_dataset import setup_dataloaders
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.selector_v1_ac import ACConfig, SlotSelectorAC, evaluate_greedy
from src.utils import reconstruct_autoencoder, seed_all
from scripts.train_selector_v1_ac import batch_slots, full_order_accuracy, round_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_v1_ac_best.pt")
    parser.add_argument("--env_path", default="")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--taus", default="0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--out_name", default="tau_sweep_metrics.json")
    parser.add_argument("--eval_batches", type=int, default=0)
    return parser.parse_args()


def load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "selector_v1_ac_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def make_ac_config(raw: dict) -> ACConfig:
    cfg = dict(raw)
    if cfg.get("class_min_slots") is not None:
        cfg["class_min_slots"] = tuple(cfg["class_min_slots"])
    return ACConfig(**cfg)


def parse_taus(raw: str) -> list[float]:
    taus = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not taus:
        raise ValueError("--taus must contain at least one value")
    return taus


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = load_meta(run_dir)

    env_path = args.env_path or meta["args"]["env_path"]
    sa_checkpoint = args.sa_checkpoint or meta["args"]["sa_checkpoint"]
    config = Config(env_path, str(run_dir))
    seed_all(config.seed, config.deterministic)
    device = torch.device("cuda" if config.use_gpu else "cpu")
    num_classes = len(config.labels)

    if config.dataset == "ch":
        loaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        loaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    sa, sa_best_loss = reconstruct_autoencoder(sa_checkpoint, config)
    sa.to(device)
    sa.eval()
    for param in sa.parameters():
        param.requires_grad = False

    ac_cfg = make_ac_config(meta["ac_config"])
    model = SlotSelectorAC(ac_cfg).to(device)
    ckpt_path = run_dir / args.checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    slot_fn = lambda images: batch_slots(sa, images, device, config, pos_dim=ac_cfg.pos_dim)
    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "sa_checkpoint": sa_checkpoint,
        "sa_best_loss": float(sa_best_loss),
        "disable_stop_action": ac_cfg.disable_stop_action,
        "taus": {},
    }
    full_order = full_order_accuracy(
        model,
        loaders["test"],
        slot_fn,
        device,
        max_batches=args.eval_batches,
    )

    for tau in parse_taus(args.taus):
        model.cfg.early_exit_conf = tau
        val = evaluate_greedy(
            model,
            loaders["val"],
            slot_fn,
            device,
            num_classes,
            max_batches=args.eval_batches,
        )
        test = evaluate_greedy(
            model,
            loaders["test"],
            slot_fn,
            device,
            num_classes,
            max_batches=args.eval_batches,
        )
        test["full_order_accuracy"] = full_order
        results["taus"][f"{tau:.2f}"] = {
            "val": round_logs(val),
            "test": round_logs(test),
        }
        print(
            json.dumps(
                {
                    "tau": tau,
                    "val_accuracy": val["accuracy"],
                    "val_avg_selected": val["avg_selected"],
                    "test_accuracy": test["accuracy"],
                    "test_avg_selected": test["avg_selected"],
                    "test_selected_count_11": test.get("selected_count_11", 0.0),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    out_path = run_dir / args.out_name
    out_path.write_text(json.dumps(round_logs(results), indent=2), encoding="utf-8")
    print(f"Done. Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
