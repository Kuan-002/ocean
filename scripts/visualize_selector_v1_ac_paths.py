#!/usr/bin/env python3
"""Visualize greedy slot-selection paths for Slot Selector V1 AC checkpoints."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.clevr_hans_dataset import ch7_classes, setup_dataloaders
from src.config import Config
from src.selector_v1_ac import ACConfig, SlotSelectorAC
from src.utils import reconstruct_autoencoder, seed_all
from scripts.train_selector_v1_ac import attention_to_xy, make_per_image_slot_init_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--env_path", default="")
    parser.add_argument("--checkpoint", default="selector_v1_ac_best.pt")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--per_class_correct", type=int, default=2)
    parser.add_argument("--per_class_wrong", type=int, default=2)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def denorm_image(image: torch.Tensor) -> torch.Tensor:
    return ((image.detach().cpu() * 127.5 + 127.5) / 255.0).clamp(0.0, 1.0)


@torch.no_grad()
def batch_sa_outputs(sa, images: torch.Tensor, device: torch.device, config: Config, pos_dim: int):
    x = images.to(device)
    slot_noise = None
    if config.rnn_sel_sa_deterministic_slots or config.deterministic:
        slot_noise = make_per_image_slot_init_noise(
            x,
            config.num_slots,
            config.slot_dim,
            config.rnn_sel_sa_noise_seed,
        )
    _, recons, masks, slots, attn = sa(x, slot_init_noise=slot_noise)
    slot_pos = attention_to_xy(attn) if pos_dim > 0 else None
    return slots, attn, slot_pos, recons, masks


@torch.no_grad()
def greedy_trace(model: SlotSelectorAC, slots: torch.Tensor, slot_pos: torch.Tensor | None):
    cfg = model.cfg
    slot_embeds = model.embed_slots(slots, slot_pos)
    b, k, _ = slot_embeds.shape
    if b != 1:
        raise ValueError("greedy_trace expects batch size 1")

    h, evidence = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    active = torch.ones(b, dtype=torch.bool, device=slots.device)
    final_logits = model.classify(
        h,
        evidence,
        model.selected_pool(slot_embeds, selected_mask),
        slot_embeds,
        selected_mask,
    )
    steps = []

    for step in range(min(cfg.max_steps, k)):
        action_logits = model.policy_logits(
            h,
            slot_embeds,
            selected_mask,
            step=step,
        )
        action = action_logits.argmax(dim=-1)
        action_prob = action_logits.softmax(dim=-1)
        is_stop = action == model.stop_idx
        select = active & ~is_stop

        steps.append(
            {
                "step": step,
                "action": int(action.item()),
                "is_stop": bool(is_stop.item()),
                "policy_prob": float(action_prob[0, action.item()].item()),
                "min_steps": int(cfg.min_steps),
                "selected_before": int(selected_mask.sum(dim=1).item()),
            }
        )

        h, evidence, selected_mask = model.update_with_action(
            h, evidence, selected_mask, slot_embeds, action, select
        )
        step_logits = model.classify(
            h,
            evidence,
            model.selected_pool(slot_embeds, selected_mask),
            slot_embeds,
            selected_mask,
        )
        step_prob = step_logits.softmax(dim=-1)
        steps[-1]["post_pred"] = int(step_prob.argmax(dim=-1).item())
        steps[-1]["post_conf"] = float(step_prob.max(dim=-1).values.item())
        final_logits = torch.where(active.unsqueeze(-1), step_logits, final_logits)
        active = active & ~is_stop
        conf = step_prob.max(dim=-1).values
        next_min_steps = cfg.min_steps
        enough = selected_mask.sum(dim=1) >= next_min_steps
        active = active & ~((conf >= cfg.early_exit_conf) & enough)
        if not active.any():
            if not bool(is_stop.item()):
                steps[-1]["early_exit"] = True
            break

    final_prob = final_logits.softmax(dim=-1)
    return {
        "steps": steps,
        "pred": int(final_prob.argmax(dim=-1).item()),
        "conf": float(final_prob.max(dim=-1).values.item()),
        "selected_count": int(selected_mask.sum(dim=1).item()),
        "selected_slots": [s["action"] for s in steps if not s["is_stop"]],
    }


def plot_trace(
    image: torch.Tensor,
    attn: torch.Tensor,
    recons: torch.Tensor,
    masks: torch.Tensor,
    label: int,
    trace: dict,
    class_names: dict[int, str],
    out_path: Path,
):
    selected_steps = [s for s in trace["steps"] if not s["is_stop"]]
    ncols = max(4, min(6, max(1, len(selected_steps))))
    slot_rows = max(1, math.ceil(max(1, len(selected_steps)) / ncols))
    fig = plt.figure(figsize=(3.0 * ncols, 4.2 + 2.5 * slot_rows), constrained_layout=True)
    gs = fig.add_gridspec(2 + slot_rows, ncols, height_ratios=[1.0, 1.0] + [1.15] * slot_rows)
    img = denorm_image(image).permute(1, 2, 0)
    ax_img = fig.add_subplot(gs[:2, 0])
    ax_img.imshow(img)
    ax_img.set_title(
        f"original\ntrue {label} | post pred {trace['pred']} ({trace['conf']:.2f})\n"
        f"{class_names.get(label, str(label)).replace(chr(10), ' ')}",
        fontsize=9,
    )
    ax_img.axis("off")

    ax_conf = fig.add_subplot(gs[:2, 1:])
    post_steps = trace["steps"]
    x_steps = [s["step"] + 1 for s in post_steps]
    y_conf = [s["post_conf"] for s in post_steps]
    pred_labels = [s["post_pred"] for s in post_steps]
    if x_steps:
        ax_conf.plot(x_steps, y_conf, marker="o", linewidth=2.0, color="#1f77b4")
        for x, y, pred in zip(x_steps, y_conf, pred_labels):
            ax_conf.annotate(
                str(pred),
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )
    ax_conf.axhline(trace["conf"], color="#555555", linestyle="--", linewidth=1.0, alpha=0.65)
    ax_conf.set_title("post confidence by step (labels are post predictions)", fontsize=10)
    ax_conf.set_xlabel("selection step")
    ax_conf.set_ylabel("post confidence")
    ax_conf.set_ylim(0.0, 1.02)
    ax_conf.set_xlim(0.5, max(1.5, len(post_steps) + 0.5))
    ax_conf.grid(True, alpha=0.25)

    for i, step in enumerate(selected_steps):
        row = 2 + i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        slot_id = step["action"]
        slot_recon = denorm_image(recons[slot_id]).permute(1, 2, 0)
        slot_mask = masks[slot_id, 0].detach().cpu().float().clamp(0.0, 1.0)
        white = torch.ones_like(slot_recon)
        masked_slot = slot_recon * slot_mask.unsqueeze(-1) + white * (1.0 - slot_mask.unsqueeze(-1))
        ax.imshow(masked_slot)
        ax.imshow(slot_mask, cmap="gray", alpha=0.16, vmin=0.0, vmax=1.0)
        suffix = " | early" if step.get("early_exit") else ""
        ax.set_title(
            f"step {step['step'] + 1} | slot {slot_id}{suffix}\n"
            f"post pred {step.get('post_pred', '-')} | conf {step.get('post_conf', 0.0):.3f}",
            fontsize=8,
        )
        ax.axis("off")

    for i in range(len(selected_steps), slot_rows * ncols):
        row = 2 + i // ncols
        col = i % ncols
        fig.add_subplot(gs[row, col]).axis("off")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_run(run_dir: Path, checkpoint_name: str, device: torch.device):
    meta = json.loads((run_dir / "selector_v1_ac_meta.json").read_text(encoding="utf-8"))
    ac_raw = meta["ac_config"]
    ac_raw["class_min_slots"] = (
        tuple(ac_raw["class_min_slots"]) if ac_raw.get("class_min_slots") is not None else None
    )
    cfg = ACConfig(**ac_raw)
    model = SlotSelectorAC(cfg).to(device)
    checkpoint = torch.load(run_dir / checkpoint_name, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return meta, model


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    run_dir = Path(args.run_dir)
    device = choose_device(args.device)
    meta, model = load_run(run_dir, args.checkpoint, device)
    env_path = args.env_path or meta["args"]["env_path"]
    sa_checkpoint = args.sa_checkpoint or meta["args"]["sa_checkpoint"]
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualizations" / f"{args.split}_slot_paths"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config(env_path, None)
    config.dataset_eval_batch_size = 1
    config.dataset_num_workers = 0
    loaders = setup_dataloaders(config, eval=True)
    sa, _ = reconstruct_autoencoder(sa_checkpoint, config)
    sa.to(device)
    sa.eval()
    for param in sa.parameters():
        param.requires_grad = False

    per_class = {int(label): {"correct": 0, "wrong": 0} for label in meta["labels"]}
    records = []
    total_seen = 0
    for idx, (images, _, labels) in enumerate(loaders[args.split]):
        label = int(labels.item())
        if label not in per_class:
            continue
        if args.max_items and len(records) >= args.max_items:
            break
        images = images.to(device)
        labels = labels.to(device)
        slots, attn, slot_pos, recons, masks = batch_sa_outputs(sa, images, device, config, model.cfg.pos_dim)
        trace = greedy_trace(model, slots, slot_pos)
        correct = trace["pred"] == label
        bucket = "correct" if correct else "wrong"
        limit = args.per_class_correct if correct else args.per_class_wrong
        if per_class[label][bucket] >= limit:
            total_seen += 1
            if all(
                counts["correct"] >= args.per_class_correct and counts["wrong"] >= args.per_class_wrong
                for counts in per_class.values()
            ):
                break
            continue

        fname = f"{args.split}_idx{idx:05d}_true{label}_pred{trace['pred']}_{bucket}.png"
        plot_trace(
            images[0].cpu(),
            attn[0].cpu(),
            recons[0].cpu(),
            masks[0].cpu(),
            label,
            trace,
            ch7_classes,
            out_dir / fname,
        )
        record = {
            "split": args.split,
            "dataset_index": idx,
            "true": label,
            "pred": trace["pred"],
            "correct": correct,
            "conf": trace["conf"],
            "selected_count": trace["selected_count"],
            "selected_slots": trace["selected_slots"],
            "steps": trace["steps"],
            "file": fname,
        }
        records.append(record)
        per_class[label][bucket] += 1
        total_seen += 1

        if all(
            counts["correct"] >= args.per_class_correct and counts["wrong"] >= args.per_class_wrong
            for counts in per_class.values()
        ):
            break

    (out_dir / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    html = ["<html><body><h1>Slot selection traces</h1>"]
    for rec in records:
        html.append(
            f"<h3>{rec['file']} | true={rec['true']} pred={rec['pred']} "
            f"conf={rec['conf']:.3f} selected={rec['selected_slots']}</h3>"
        )
        html.append(f"<img src='{rec['file']}' style='max-width:100%; border:1px solid #ccc;'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "records": len(records), "seen": total_seen}, sort_keys=True))


if __name__ == "__main__":
    main()
