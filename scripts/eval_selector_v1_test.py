#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clevr_hans_dataset import setup_dataloaders  # noqa: E402
from src.config import Config  # noqa: E402
from src.selector_v1_ac import ACConfig, SlotSelectorAC, evaluate_greedy  # noqa: E402
from src.slot_autoencoder import SlotAutoencoder  # noqa: E402


def seed_all(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def reconstruct_autoencoder(checkpoint: Path, config: Config) -> SlotAutoencoder:
    model = SlotAutoencoder(config)
    model.load(str(checkpoint))
    return model


def round_logs(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {key: round_logs(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_logs(value) for value in obj]
    return obj


def make_per_image_slot_init_noise(
    images: torch.Tensor,
    num_slots: int,
    slot_dim: int,
    base_seed: int,
) -> torch.Tensor:
    batch_size = images.size(0)
    noise = torch.empty(
        batch_size,
        num_slots,
        slot_dim,
        device=images.device,
        dtype=images.dtype,
    )
    flat_images = images.detach().reshape(batch_size, -1)
    for index in range(batch_size):
        generator = torch.Generator(device=images.device)
        image_seed = int(flat_images[index].sum().item() * 1e6) % (2**31 - 2)
        generator.manual_seed(int((base_seed + image_seed) % (2**31 - 1)))
        noise[index] = torch.randn(
            num_slots,
            slot_dim,
            generator=generator,
            device=images.device,
            dtype=images.dtype,
        )
    return noise


@torch.no_grad()
def attention_to_xy(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 3:
        raise ValueError(f"Expected attention [B, K, N], got {tuple(attention.shape)}")
    spatial_size = attention.size(-1)
    side = int(spatial_size**0.5)
    if side * side != spatial_size:
        raise ValueError(f"Attention spatial size must be square, got N={spatial_size}")
    ys = torch.linspace(0.0, 1.0, side, device=attention.device, dtype=attention.dtype)
    xs = torch.linspace(0.0, 1.0, side, device=attention.device, dtype=attention.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    weights = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bkn,nc->bkc", weights, grid)


@torch.no_grad()
def batch_slots(
    autoencoder: SlotAutoencoder,
    images: torch.Tensor,
    device: torch.device,
    config: Config,
    *,
    pos_dim: int,
):
    autoencoder.eval()
    images = images.to(device)
    slot_noise = None
    if config.rnn_sel_sa_deterministic_slots or config.deterministic:
        slot_noise = make_per_image_slot_init_noise(
            images,
            config.num_slots,
            config.slot_dim,
            config.rnn_sel_sa_noise_seed,
        )
    slots, attention = autoencoder.forward_slots_only(images, slot_init_noise=slot_noise)
    if pos_dim == 0:
        return slots
    return slots, attention_to_xy(attention)


@torch.no_grad()
def full_order_accuracy(
    model: SlotSelectorAC,
    loader,
    slot_fn,
    device: torch.device,
    *,
    max_batches: int = 0,
) -> float:
    model.eval()
    total = 0
    correct = 0
    for batch_index, (images, _, labels) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        labels = labels.to(device)
        slot_batch = slot_fn(images)
        slots, slot_positions = slot_batch if isinstance(slot_batch, tuple) else (slot_batch, None)
        slot_embeddings = model.embed_slots(slots, slot_positions)
        batch_size, num_slots, _ = slot_embeddings.shape
        hidden, evidence = model.initial_state(slot_embeddings)
        selected = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        for slot_index in range(num_slots):
            action = torch.full((batch_size,), slot_index, dtype=torch.long, device=device)
            hidden, evidence, selected = model.update_with_action(
                hidden,
                evidence,
                selected,
                slot_embeddings,
                action,
                active,
            )
        logits = model.classify(
            hidden,
            evidence,
            model.selected_pool(slot_embeddings, selected),
            slot_embeddings,
            selected,
        )
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ocean Slot Selector V1 AC/GRPO on the test split.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--method", choices=["ac", "grpo"], required=True)
    parser.add_argument("--env_path", type=Path, default=None)
    parser.add_argument("--sa_checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--config_out_dir",
        type=Path,
        default=None,
        help="Scratch output directory for Config side effects. Defaults to <output parent>/config.",
    )
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DATASET_NUM_WORKERS; use 0 for restricted local environments.",
    )
    parser.add_argument(
        "--early_exit_conf",
        type=float,
        default=None,
        help="Override the checkpoint/config early-exit confidence threshold for greedy evaluation.",
    )
    parser.add_argument("--require_cuda", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_name = "selector_v1_ac_meta.json" if args.method == "ac" else "selector_v1_grpo_meta.json"
    checkpoint_name = "selector_v1_ac_best.pt" if args.method == "ac" else "selector_v1_grpo_best.pt"
    meta = json.loads((args.run_dir / meta_name).read_text(encoding="utf-8"))
    meta_args = meta.get("args", {})
    env_path = Path(args.env_path or meta_args["env_path"])
    if not env_path.exists() and (args.run_dir / ".env").exists():
        env_path = args.run_dir / ".env"
    sa_checkpoint = Path(args.sa_checkpoint or meta_args["sa_checkpoint"])
    checkpoint_path = Path(args.checkpoint or (args.run_dir / checkpoint_name))
    output = Path(args.output or (args.run_dir / "test_metrics.json"))
    config_out_dir = Path(args.config_out_dir or (output.parent / "config"))

    config = Config(str(env_path), str(config_out_dir))
    if args.num_workers is not None:
        config.dataset_num_workers = args.num_workers
    seed_all(config.seed, config.deterministic)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

    if config.dataset != "ch":
        raise ValueError(f"This minimal evaluator supports the CLEVR-Hans experiment, got {config.dataset!r}")
    loaders = setup_dataloaders(config, eval=True)
    sa = reconstruct_autoencoder(sa_checkpoint, config)
    sa.to(device).eval()
    for param in sa.parameters():
        param.requires_grad = False

    ac_cfg = ACConfig(**meta["ac_config"])
    if args.early_exit_conf is not None:
        ac_cfg.early_exit_conf = args.early_exit_conf
    model = SlotSelectorAC(ac_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    slot_fn = lambda images: batch_slots(sa, images, device, config, pos_dim=ac_cfg.pos_dim)
    num_classes = len(config.labels)

    test = evaluate_greedy(model, loaders["test"], slot_fn, device, num_classes, max_batches=args.eval_batches)
    test["full_order_accuracy"] = full_order_accuracy(model, loaders["test"], slot_fn, device, max_batches=args.eval_batches)
    result = round_logs(test)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "checkpoint": str(checkpoint_path), "test": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
