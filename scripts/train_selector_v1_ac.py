#!/usr/bin/env python3
"""Train the no-prewarm Slot Selector V1 with step-wise actor-critic."""

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
from src.selector_v1_ac import ACConfig, SlotSelectorAC, evaluate_greedy, rollout_actor_critic
from src.utils import reconstruct_autoencoder, seed_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Slot Selector V1 actor-critic without prewarm.")
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--sa_checkpoint", required=True)
    parser.add_argument("--out_subpath", required=True)
    parser.add_argument("--stage0_epochs", type=int, default=0, help="Compatibility only. Must be 0.")
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--pos_dim", type=int, default=2, choices=[0, 2])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means num_slots.")
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument(
        "--class_min_slots",
        default="",
        help="Comma-separated per-class minimum selected slots. Empty uses --min_steps.",
    )
    parser.add_argument("--target_min_slots", type=int, default=3)
    parser.add_argument("--target_max_slots", type=int, default=5)
    parser.add_argument("--lambda_slot", type=float, default=0.01)
    parser.add_argument("--r_correct", type=float, default=1.0)
    parser.add_argument("--r_wrong", type=float, default=1.0)
    parser.add_argument("--lambda_over", type=float, default=0.10)
    parser.add_argument("--lambda_under", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--class_coef", type=float, default=0.3)
    parser.add_argument("--full_order_class_coef", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--global_init", action="store_true")
    parser.add_argument("--early_exit_conf", type=float, default=0.8)
    parser.add_argument(
        "--ordered_classifier",
        action="store_true",
        help="Replace the classifier selected-set mean-pool input with the sequential hidden state.",
    )
    parser.add_argument(
        "--cross_attention_classifier",
        action="store_true",
        help="Replace the classifier selected-set mean-pool input with cross-attention over selected slots.",
    )
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    return parser.parse_args()


def parse_class_min_slots(raw: str, num_classes: int) -> tuple[int, ...] | None:
    raw = raw.strip()
    if not raw:
        return None
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != num_classes:
        raise ValueError(f"--class_min_slots must contain {num_classes} values, got {len(values)}")
    if any(value < 0 for value in values):
        raise ValueError("--class_min_slots values must be non-negative")
    return values


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
    b = images.size(0)
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
def attention_to_xy(attn: torch.Tensor) -> torch.Tensor:
    """Return per-slot normalized x/y centroids from Slot Attention maps."""
    if attn.ndim != 3:
        raise ValueError(f"Expected attention [B, K, N], got {tuple(attn.shape)}")
    n = attn.size(-1)
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"Attention spatial size must be square, got N={n}")
    ys = torch.linspace(0.0, 1.0, side, device=attn.device, dtype=attn.dtype)
    xs = torch.linspace(0.0, 1.0, side, device=attn.device, dtype=attn.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    weights = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bkn,nc->bkc", weights, grid)


@torch.no_grad()
def batch_slots(
    sa,
    images: torch.Tensor,
    device: torch.device,
    config: Config,
    *,
    pos_dim: int,
):
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
    slots, attn = sa.forward_slots_only(x, slot_init_noise=slot_noise)
    if pos_dim == 0:
        return slots
    return slots, attention_to_xy(attn)


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
    for batch_idx, (images, _, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        labels = labels.to(device)
        slot_batch = slot_fn(images)
        if isinstance(slot_batch, tuple):
            slots, slot_pos = slot_batch
        else:
            slots, slot_pos = slot_batch, None
        slot_embeds = model.embed_slots(slots, slot_pos)
        b, k, _ = slot_embeds.shape
        h, evidence = model.initial_state(slot_embeds)
        selected = torch.zeros(b, k, dtype=torch.bool, device=device)
        active = torch.ones(b, dtype=torch.bool, device=device)
        for idx in range(k):
            action = torch.full((b,), idx, dtype=torch.long, device=device)
            h, evidence, selected = model.update_with_action(h, evidence, selected, slot_embeds, action, active)
        logits = model.classify(h, evidence, model.selected_pool(slot_embeds, selected), slot_embeds, selected)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train_one_epoch(
    model: SlotSelectorAC,
    loader,
    slot_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "actor_loss": 0.0,
        "value_loss": 0.0,
        "class_loss": 0.0,
        "full_order_class_loss": 0.0,
        "entropy": 0.0,
        "mean_reward": 0.0,
        "mean_return": 0.0,
        "mean_advantage": 0.0,
        "positive_reward_rate": 0.0,
        "accuracy": 0.0,
        "avg_selected": 0.0,
    }
    total = 0
    stop_accum = torch.zeros(model.cfg.max_steps, dtype=torch.float64)
    for images, _, labels in loader:
        labels = labels.to(device)
        slot_batch = slot_fn(images)
        if isinstance(slot_batch, tuple):
            slots, slot_pos = slot_batch
        else:
            slots, slot_pos = slot_batch, None
        optimizer.zero_grad(set_to_none=True)
        out = rollout_actor_critic(model, slots, labels, slot_pos=slot_pos)
        if out.loss is None:
            continue
        out.loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        n = labels.size(0)
        total += n
        totals["loss"] += out.loss.detach().item() * n
        totals["actor_loss"] += out.actor_loss.item() * n
        totals["value_loss"] += out.value_loss.item() * n
        totals["class_loss"] += out.class_loss.item() * n
        totals["full_order_class_loss"] += out.full_order_class_loss.item() * n
        totals["entropy"] += out.entropy.item() * n
        totals["mean_reward"] += out.mean_reward.item() * n
        totals["mean_return"] += out.mean_return.item() * n
        totals["mean_advantage"] += out.mean_advantage.item() * n
        totals["positive_reward_rate"] += out.positive_reward_rate.item() * n
        totals["accuracy"] += (out.logits.argmax(dim=-1) == labels).sum().item()
        totals["avg_selected"] += out.selected_counts.detach().sum().item()
        stop_accum += out.stop_rate_by_step.detach().cpu().double() * n

    metrics = {key: value / max(total, 1) for key, value in totals.items()}
    for step, value in enumerate(stop_accum.tolist()):
        metrics[f"stop_rate_step_{step}"] = value / max(total, 1)
    metrics["total"] = float(total)
    return metrics


def save_checkpoint(
    path: Path,
    model: SlotSelectorAC,
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
    if args.stage0_epochs != 0:
        raise ValueError("This V1 implementation is explicitly no-prewarm; --stage0_epochs must be 0.")

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

    sa, sa_best_loss = reconstruct_autoencoder(args.sa_checkpoint, config)
    sa.to(device)
    sa.eval()
    for param in sa.parameters():
        param.requires_grad = False

    class_min_slots = parse_class_min_slots(args.class_min_slots, num_classes)
    ac_cfg = ACConfig(
        slot_dim=config.slot_dim,
        pos_dim=args.pos_dim,
        embed_dim=args.embed_dim,
        num_slots=config.num_slots,
        num_classes=num_classes,
        max_steps=args.max_steps or config.num_slots,
        min_steps=args.min_steps,
        class_min_slots=class_min_slots,
        gamma=args.gamma,
        lambda_slot=args.lambda_slot,
        target_min_slots=args.target_min_slots,
        target_max_slots=args.target_max_slots,
        r_correct=args.r_correct,
        r_wrong=args.r_wrong,
        lambda_over=args.lambda_over,
        lambda_under=args.lambda_under,
        value_coef=args.value_coef,
        class_coef=args.class_coef,
        full_order_class_coef=args.full_order_class_coef,
        entropy_coef=args.entropy_coef,
        dropout=args.dropout,
        global_init=args.global_init,
        early_exit_conf=args.early_exit_conf,
        ordered_classifier=args.ordered_classifier,
        cross_attention_classifier=args.cross_attention_classifier,
    )
    model = SlotSelectorAC(ac_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    slot_fn = lambda images: batch_slots(sa, images, device, config, pos_dim=args.pos_dim)

    meta = {
        "args": vars(args),
        "ac_config": vars(ac_cfg),
        "labels": config.labels,
        "sa_best_loss": float(sa_best_loss),
        "prewarm": False,
        "teacher_distillation": False,
        "ppo": False,
        "grpo": False,
    }
    (out_dir / "selector_v1_ac_meta.json").write_text(
        json.dumps(round_logs(meta), indent=2),
        encoding="utf-8",
    )

    best_val = -1.0
    best_epoch = -1
    evals_since_improvement = 0
    history = []
    for epoch in range(args.stage1_epochs):
        train = train_one_epoch(
            model,
            loaders["train"],
            slot_fn,
            optimizer,
            device,
            max_grad_norm=args.max_grad_norm,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}}

        if (epoch + 1) % args.eval_every == 0:
            val = evaluate_greedy(
                model,
                loaders["val"],
                slot_fn,
                device,
                num_classes,
                max_batches=args.eval_batches,
            )
            val["full_order_accuracy"] = full_order_accuracy(
                model,
                loaders["val"],
                slot_fn,
                device,
                max_batches=args.eval_batches,
            )
            row.update({f"val_{k}": v for k, v in val.items()})
            improved = val["accuracy"] > best_val + args.early_stop_min_delta
            if improved:
                best_val = val["accuracy"]
                best_epoch = epoch
                evals_since_improvement = 0
                save_checkpoint(out_dir / "selector_v1_ac_best.pt", model, optimizer, epoch, meta, val)
            else:
                evals_since_improvement += 1
            row["best_val_accuracy"] = best_val
            row["best_epoch"] = best_epoch
            row["early_stop_wait"] = evals_since_improvement

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(out_dir / "selector_v1_ac_last.pt", model, optimizer, epoch, meta)
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

    best_path = out_dir / "selector_v1_ac_best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate_greedy(model, loaders["test"], slot_fn, device, num_classes)
    test["full_order_accuracy"] = full_order_accuracy(model, loaders["test"], slot_fn, device)
    test = round_logs(test)
    (out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(json.dumps({"test": test, "best_epoch": best_epoch}, sort_keys=True), flush=True)
    print(f"Done. Best val accuracy: {best_val:.4f}. Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
