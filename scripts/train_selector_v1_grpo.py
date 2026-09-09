#!/usr/bin/env python3
"""Train Slot Selector V1 with group-relative policy optimization.

This keeps the same selector architecture and reward surface as
``train_selector_v1_ac.py``.  The only algorithmic change is replacing the
learned value baseline with within-image group-normalized advantages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_selector_v1_ac import (  # noqa: E402
    batch_slots,
    full_order_accuracy,
    make_per_image_slot_init_noise,
    parse_class_min_slots,
    round_logs,
    save_checkpoint,
)
from src.clevr_hans_dataset import setup_dataloaders  # noqa: E402
from src.config import Config  # noqa: E402
from src.runtime import reconstruct_autoencoder, seed_all  # noqa: E402
from src.selector_v1_ac import ACConfig, SlotSelectorAC, evaluate_greedy, multiclass_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--sa_checkpoint", required=True)
    parser.add_argument("--out_subpath", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--pos_dim", type=int, default=0, choices=[0, 2])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--class_min_slots", default="")
    parser.add_argument("--target_min_slots", type=int, default=3)
    parser.add_argument("--target_max_slots", type=int, default=0)
    parser.add_argument("--lambda_slot", type=float, default=0.01)
    parser.add_argument("--r_correct", type=float, default=1.0)
    parser.add_argument("--r_wrong", type=float, default=1.0)
    parser.add_argument("--lambda_over", type=float, default=0.0)
    parser.add_argument("--lambda_under", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--grpo_advantage", choices=["step", "return"], default="step")
    parser.add_argument("--stop_reward_mode", choices=["correct", "score_delta"], default="correct")
    parser.add_argument("--stop_reward_scale", type=float, default=1.0)
    parser.add_argument("--premature_stop_coef", type=float, default=0.0)
    parser.add_argument("--future_gain_margin", type=float, default=0.0)
    parser.add_argument("--class_coef", type=float, default=0.5)
    parser.add_argument("--full_order_class_coef", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--global_init", action="store_true")
    parser.add_argument("--first_step_cross_attention", action="store_true")
    parser.add_argument("--first_step_num_heads", type=int, default=4)
    parser.add_argument("--early_exit_conf", type=float, default=0.8)
    parser.add_argument("--ordered_classifier", action="store_true")
    parser.add_argument("--cross_attention_classifier", action="store_true")
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--skip_test", action="store_true", default=False)
    return parser.parse_args()


def group_advantage(reward: torch.Tensor, mask: torch.Tensor, batch: int, group: int, eps: float = 1e-4) -> torch.Tensor:
    reward_g = reward.detach().view(batch, group, -1)
    mask_g = mask.detach().view(batch, group, -1)
    denom = mask_g.sum(1, keepdim=True).clamp_min(1.0)
    mean = (reward_g * mask_g).sum(1, keepdim=True) / denom
    centered = (reward_g - mean) * mask_g
    scale = (centered.square().sum(1, keepdim=True) / denom).sqrt().clamp_min(eps)
    return (centered / scale).reshape(batch * group, -1)


def discounted_returns(reward: torch.Tensor, gamma: float) -> torch.Tensor:
    running = torch.zeros_like(reward[:, 0])
    values = []
    for step in range(reward.size(1) - 1, -1, -1):
        running = reward[:, step] + gamma * running
        values.append(running)
    values.reverse()
    return torch.stack(values, dim=1)


def best_future_logp_gain(
    model: SlotSelectorAC,
    h: torch.Tensor,
    evidence: torch.Tensor,
    selected: torch.Tensor,
    slot_embeds: torch.Tensor,
    labels: torch.Tensor,
    current_log_p_true: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    gains = []
    b, k, _ = slot_embeds.shape
    rows = torch.arange(b, device=slot_embeds.device)
    for idx in range(k):
        candidate_active = active & ~selected[:, idx]
        action = torch.full((b,), idx, dtype=torch.long, device=slot_embeds.device)
        cand_h, cand_evidence, cand_selected = model.update_with_action(
            h.detach(),
            evidence.detach(),
            selected.detach(),
            slot_embeds.detach(),
            action,
            candidate_active,
        )
        cand_logits = model.classify(
            cand_h,
            cand_evidence,
            model.selected_pool(slot_embeds.detach(), cand_selected),
            slot_embeds.detach(),
            cand_selected,
        )
        cand_log_p_true = F.log_softmax(cand_logits, dim=-1)[rows, labels]
        gain = cand_log_p_true - current_log_p_true
        gain = torch.where(candidate_active, gain, current_log_p_true.new_full((b,), float("-inf")))
        gains.append(gain)
    best = torch.stack(gains, dim=1).max(dim=1).values
    return torch.where(torch.isfinite(best), best, torch.zeros_like(best))


def rollout_grpo(
    model: SlotSelectorAC,
    slots: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    *,
    slot_pos: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    slot_embeds = model.embed_slots(slots, slot_pos)
    b, k, _ = slot_embeds.shape
    h, evidence = model.initial_state(slot_embeds)
    selected = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    active = torch.ones(b, dtype=torch.bool, device=slots.device)
    initial_score = slots.new_full((b,), -torch.log(slots.new_tensor(float(cfg.num_classes))))
    prev_score = initial_score
    final_logits = model.classify(h, evidence, model.selected_pool(slot_embeds, selected), slot_embeds, selected)
    target_min_slots = model.min_steps_for_classes(labels)
    log_probs, entropies, masks, rewards, class_losses = [], [], [], [], []

    for step in range(min(cfg.max_steps, k)):
        if not active.any():
            break
        logits_action = model.policy_logits(h, slot_embeds, selected, step=step)
        dist = torch.distributions.Categorical(logits=logits_action)
        action = dist.sample()
        is_stop = action == model.stop_idx
        select = active & ~is_stop
        h, evidence, selected = model.update_with_action(h, evidence, selected, slot_embeds, action, select)
        logits_cls = model.classify(h, evidence, model.selected_pool(slot_embeds, selected), slot_embeds, selected)
        final_logits = torch.where(active.unsqueeze(-1), logits_cls, final_logits)
        log_p_true = F.log_softmax(logits_cls, dim=-1).gather(1, labels[:, None]).squeeze(1)
        pred = logits_cls.argmax(dim=-1)
        selected_count = selected.sum(dim=1)
        select_reward = (log_p_true.detach() - prev_score) - cfg.lambda_slot
        too_few = (selected_count < target_min_slots).to(slots.dtype)
        too_many = (selected_count > cfg.target_max_slots).to(slots.dtype) if cfg.target_max_slots > 0 else torch.zeros_like(too_few)
        if args.stop_reward_mode == "score_delta":
            stop_reward = args.stop_reward_scale * (log_p_true.detach() - initial_score)
        else:
            stop_reward = torch.where(
                pred == labels,
                slots.new_full((b,), cfg.r_correct),
                slots.new_full((b,), -cfg.r_wrong),
            )
            stop_reward = args.stop_reward_scale * stop_reward
        if args.premature_stop_coef > 0:
            future_gain = best_future_logp_gain(
                model,
                h,
                evidence,
                selected,
                slot_embeds,
                labels,
                log_p_true.detach(),
                active,
            )
            stop_reward = stop_reward - args.premature_stop_coef * torch.relu(future_gain - args.future_gain_margin)
        stop_reward = stop_reward - cfg.lambda_under * too_few - cfg.lambda_over * too_many
        terminal = active & ~is_stop & (step == min(cfg.max_steps, k) - 1)
        reward = torch.where(is_stop | terminal, stop_reward, select_reward)
        reward = torch.where(active, reward, torch.zeros_like(reward))

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        masks.append(active.to(slots.dtype))
        rewards.append(reward)
        class_losses.append(F.cross_entropy(logits_cls, labels, reduction="none"))
        prev_score = torch.where(active, log_p_true.detach(), prev_score)
        active = active & ~is_stop

    stack = lambda values: torch.stack(values, dim=1) if values else slots.new_zeros((b, 0))
    return {
        "logits": final_logits,
        "selected_counts": selected.sum(dim=1),
        "log_probs": stack(log_probs),
        "entropies": stack(entropies),
        "mask": stack(masks),
        "reward": stack(rewards),
        "class_loss": stack(class_losses),
    }


def train_one_epoch(model, loader, slot_fn, optimizer, device, args, num_classes: int) -> dict[str, float]:
    model.train()
    total = 0
    sums = {"loss": 0.0, "policy_loss": 0.0, "class_loss": 0.0, "full_order_class_loss": 0.0, "entropy": 0.0, "mean_reward": 0.0, "avg_selected": 0.0}
    logits_all, labels_all = [], []
    for images, _, labels in loader:
        labels = labels.to(device)
        slot_batch = slot_fn(images)
        slots, slot_pos = slot_batch if isinstance(slot_batch, tuple) else (slot_batch, None)
        group = int(args.grpo_group_size)
        rep_slots = slots.repeat_interleave(group, 0)
        rep_labels = labels.repeat_interleave(group, 0)
        rep_pos = slot_pos.repeat_interleave(group, 0) if slot_pos is not None else None
        out = rollout_grpo(model, rep_slots, rep_labels, args, slot_pos=rep_pos)
        mask = out["mask"]
        reward_for_advantage = discounted_returns(out["reward"], args.gamma) if args.grpo_advantage == "return" else out["reward"]
        advantage = group_advantage(reward_for_advantage, mask, labels.numel(), group)
        denom = mask.sum().clamp_min(1.0)
        policy_loss = -(out["log_probs"] * advantage * mask).sum() / denom
        class_loss = (out["class_loss"] * mask).sum() / denom
        full_order_class_loss = F.cross_entropy(model.full_order_logits(model.embed_slots(slots, slot_pos)), labels)
        entropy = (out["entropies"] * mask).sum() / denom
        loss = policy_loss + args.class_coef * class_loss + args.full_order_class_coef * full_order_class_loss - args.entropy_coef * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        n = labels.numel()
        total += n
        eval_logits = out["logits"].view(n, group, -1)[:, 0].detach().cpu()
        logits_all.append(eval_logits)
        labels_all.append(labels.detach().cpu())
        sums["loss"] += float(loss.detach()) * n
        sums["policy_loss"] += float(policy_loss.detach()) * n
        sums["class_loss"] += float(class_loss.detach()) * n
        sums["full_order_class_loss"] += float(full_order_class_loss.detach()) * n
        sums["entropy"] += float(entropy.detach()) * n
        sums["mean_reward"] += float((out["reward"] * mask).sum().detach() / denom) * n
        sums["avg_selected"] += float(out["selected_counts"].view(n, group)[:, 0].sum().detach())
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    if logits_all:
        metrics.update(multiclass_metrics(torch.cat(logits_all), torch.cat(labels_all), num_classes))
    return metrics


def main() -> None:
    args = parse_args()
    config = Config(args.env_path, args.out_subpath)
    out_dir = Path(config.out_subpath)
    if args.max_train_batches is not None:
        config.max_train_batches = args.max_train_batches
    if args.num_workers is not None:
        config.dataset_num_workers = args.num_workers
    seed_all(config.seed, config.deterministic)
    device = torch.device("cuda" if config.use_gpu else "cpu")
    if config.dataset != "ch":
        raise ValueError(f"This minimal trainer supports CLEVR-Hans, got {config.dataset!r}")
    loaders = setup_dataloaders(config)
    sa, sa_best_loss = reconstruct_autoencoder(args.sa_checkpoint, config)
    sa.to(device).eval()
    for param in sa.parameters():
        param.requires_grad = False
    num_classes = len(config.labels)
    ac_cfg = ACConfig(
        slot_dim=config.slot_dim,
        pos_dim=args.pos_dim,
        embed_dim=args.embed_dim,
        num_slots=config.num_slots,
        num_classes=num_classes,
        max_steps=args.max_steps or config.num_slots,
        min_steps=args.min_steps,
        class_min_slots=parse_class_min_slots(args.class_min_slots, num_classes),
        gamma=args.gamma,
        lambda_slot=args.lambda_slot,
        target_min_slots=args.target_min_slots,
        target_max_slots=args.target_max_slots,
        r_correct=args.r_correct,
        r_wrong=args.r_wrong,
        lambda_over=args.lambda_over,
        lambda_under=args.lambda_under,
        value_coef=0.0,
        class_coef=args.class_coef,
        full_order_class_coef=args.full_order_class_coef,
        entropy_coef=args.entropy_coef,
        dropout=args.dropout,
        global_init=args.global_init,
        first_step_cross_attention=args.first_step_cross_attention,
        first_step_num_heads=args.first_step_num_heads,
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
        "algorithm": "grpo",
        "standard": "same selector/reward/eval as ocean AC; value baseline replaced by group-normalized advantages",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selector_v1_grpo_meta.json").write_text(json.dumps(round_logs(meta), indent=2), encoding="utf-8")
    best_val = -1.0
    best_epoch = -1
    wait = 0
    history = []
    for epoch in range(args.epochs):
        train = train_one_epoch(model, loaders["train"], slot_fn, optimizer, device, args, num_classes)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}}
        if (epoch + 1) % args.eval_every == 0:
            val = evaluate_greedy(model, loaders["val"], slot_fn, device, num_classes, max_batches=args.eval_batches)
            val["full_order_accuracy"] = full_order_accuracy(model, loaders["val"], slot_fn, device, max_batches=args.eval_batches)
            row.update({f"val_{k}": v for k, v in val.items()})
            improved = val["accuracy"] > best_val + args.early_stop_min_delta
            if improved:
                best_val = val["accuracy"]
                best_epoch = epoch
                wait = 0
                save_checkpoint(out_dir / "selector_v1_grpo_best.pt", model, optimizer, epoch, meta, val)
            else:
                wait += 1
            row["best_val_accuracy"] = best_val
            row["best_epoch"] = best_epoch
            row["early_stop_wait"] = wait
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(out_dir / "selector_v1_grpo_last.pt", model, optimizer, epoch, meta)
        row = round_logs(row)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.early_stop_patience > 0 and wait >= args.early_stop_patience:
            break
    if not args.skip_test:
        best_path = out_dir / "selector_v1_grpo_best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        test = evaluate_greedy(model, loaders["test"], slot_fn, device, num_classes)
        test["full_order_accuracy"] = full_order_accuracy(model, loaders["test"], slot_fn, device)
        (out_dir / "test_metrics.json").write_text(json.dumps(round_logs(test), indent=2), encoding="utf-8")
        print(json.dumps({"test": round_logs(test), "best_epoch": best_epoch}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
