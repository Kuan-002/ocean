#!/usr/bin/env python3
"""Evaluate deploy-time RNN-only inference for variant policy models.

Equivalent to eval_rnn_only_inference.py but uses build_policy() to
automatically instantiate the correct model class based on the head_type
stored in the checkpoint meta dict.

Supports:
  "linear"    → SlotSelectionPolicyLinear
  "crossattn" → SlotSelectionPolicyCrossAttn
  "evidence"  → SlotSelectionPolicyGRU   (default when head_type missing)

Also handles legacy checkpoints (pre-Evidence Head) via key remapping,
matching the behaviour of eval_rnn_only_inference.py.

This script does NOT modify any existing file in the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.classification.training import batch_slots
from src.clevr_hans_dataset import setup_dataloaders
from src.config import Config
from src.explanation.rnn_selector_variants import build_policy
from src.explanation.rnn_selector_training import run_rnn_inference_batch
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


# ── checkpoint loading ─────────────────────────────────────────────────────────

def peek_head_type(path: str) -> str:
    """Return the head_type stored in the checkpoint meta, defaulting to 'evidence'."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return ckpt.get("meta", {}).get("head_type", "evidence")
    except Exception:
        return "evidence"


def load_selector(path: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Load state dict into model; handles legacy key remapping for old checkpoints."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = ckpt["state_dict"]

    # ── backward-compat: remap old key names to new architecture ────────────
    # Old checkpoints (before Evidence Head) used "action_head" / "class_head".
    _legacy_remap = {
        "action_head.weight": "_policy_head.weight",
        "action_head.bias":   "_policy_head.bias",
        "class_head.weight":  "_class_fc.weight",
        "class_head.bias":    "_class_fc.bias",
    }
    is_legacy = any(k in sd for k in _legacy_remap)
    if is_legacy:
        sd = {_legacy_remap.get(k, k): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=not is_legacy)

    if is_legacy:
        import types

        def _class_head_legacy(self, h, slot_embeds=None, selected_mask=None, E=None):
            return self._class_fc(h), None

        model.class_head = types.MethodType(_class_head_legacy, model)
        model._legacy_global_init = True

    return ckpt.get("meta", {})


# ── visualisation helpers ──────────────────────────────────────────────────────

def to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    return ((image.detach().cpu() * 127.5) + 127.5).clamp(0, 255).byte()


def slot_to_uint8(slot_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((slot_img * mask + (1 - mask)) * 127.5) + 127.5).detach().cpu().clamp(0, 255).byte()


@torch.no_grad()
def run_rnn_inference_trace(
    policy,
    slots: torch.Tensor,
    y_true: int,
    tau: float,
    force_full: bool,
    global_init: bool,
) -> list[dict]:
    """Single-image trace used only for visualization."""
    bsz, k, _ = slots.shape
    if bsz != 1:
        raise ValueError("visualization expects a single image")

    device = slots.device
    if global_init and getattr(policy, "_legacy_global_init", False):
        with torch.no_grad():
            h = policy.input_proj(slots).mean(dim=1)
    else:
        h = policy.init_hidden(1, device)
    E = policy.init_evidence(1, device)
    slot_cache = policy.init_slot_cache(1, device)
    selected_mask = torch.zeros(1, k, device=device, dtype=torch.bool)
    slot_embeds = policy.precompute_slot_embeds(slots)
    _always_active = torch.ones(1, device=device, dtype=torch.bool)
    trace: list[dict] = []

    for step in range(k):
        action_logits = policy.apply_action_mask(policy.forward_logits(h), selected_mask)
        if action_logits.size(1) > k:
            fill = torch.finfo(action_logits.dtype).min / 2
            action_logits = action_logits.clone()
            action_logits[:, k:] = fill

        slot_id = int(action_logits.argmax(dim=-1).item())
        h_prev = h
        h, x = policy.step_hidden(slots[:, slot_id], h)
        delta = h - h_prev
        slot_cache = policy.update_slot_cache(slot_cache, x, _always_active)
        E = policy.step_evidence(delta, x, E, slot_cache=slot_cache)
        selected_mask[0, slot_id] = True

        cls_logits, _ = policy.class_head(h, slot_embeds, selected_mask, E=E)
        probs = F.softmax(cls_logits, dim=-1)
        pred = int(cls_logits.argmax(dim=-1).item())
        pmax = float(probs.max(dim=-1).values.item())
        p_true = float(probs[0, y_true].item())

        trace.append(
            {
                "step": step + 1,
                "slot": slot_id,
                "pred": pred,
                "pmax": pmax,
                "p_true": p_true,
                "stopped": pmax >= tau,
            }
        )

        if (pmax >= tau) and (not force_full):
            break

    return trace


def save_trace_plot(
    *,
    image: torch.Tensor,
    recon_combined: torch.Tensor,
    slot_imgs: torch.Tensor,
    masks: torch.Tensor,
    trace: list[dict],
    label: int,
    sample_index: int,
    split: str,
    tau: float,
    global_init: bool,
    head_type: str,
    selector_checkpoint: str,
    meta: dict,
    out_path: str,
) -> None:
    final = trace[-1]
    final_pred = final["pred"]
    final_pmax = final["pmax"]
    stop_step = final["step"] if final["stopped"] else None

    cols = min(4, max(1, len(trace)))
    slot_rows = math.ceil(len(trace) / cols)
    fig = plt.figure(figsize=(4 * cols, 3.7 * (slot_rows + 1)))
    gs = fig.add_gridspec(slot_rows + 1, cols, height_ratios=[1.1] + [1.0] * slot_rows)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(to_uint8_image(image[0]).permute(1, 2, 0).numpy())
    stop_text = f"stop_t={stop_step}" if stop_step is not None else "no stop before K"
    ax0.set_title(
        f"Original\ntrue={label}, final_pred={final_pred}\n"
        f"pmax={final_pmax:.3f}, {stop_text}"
    )
    ax0.axis("off")

    if cols > 1:
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(to_uint8_image(recon_combined[0]).permute(1, 2, 0).numpy())
        ax1.set_title("SA reconstruction")
        ax1.axis("off")

    if cols > 2:
        ax_text = fig.add_subplot(gs[0, 2:])
        ax_text.axis("off")
        meta_line = f"selector={Path(selector_checkpoint).name}  head={head_type}"
        if meta.get("epoch") is not None:
            meta_line += f"  epoch={meta['epoch']}"
        ax_text.text(
            0.0, 0.5,
            f"RNN-only inference  head={head_type}\n"
            f"split={split}  sample={sample_index}  tau={tau:.3f}\n"
            f"global_init={global_init}\n"
            f"{meta_line}\n"
            "Step title: step / slot / pred / confidence",
            va="center", fontsize=11,
        )

    for i, item in enumerate(trace):
        row = i // cols + 1
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        slot_id = item["slot"]
        slot_vis = slot_to_uint8(slot_imgs[0, slot_id], masks[0, slot_id])
        ax.imshow(slot_vis.permute(1, 2, 0).numpy())
        marker = " STOP" if item["stopped"] and (i == len(trace) - 1) else ""
        ax.set_title(
            f"t={item['step']} slot={slot_id}{marker}\n"
            f"pred={item['pred']} pmax={item['pmax']:.3f}\n"
            f"p_true={item['p_true']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    for i in range(len(trace), slot_rows * cols):
        row = i // cols + 1
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RNN variant inference and save trace examples"
    )
    parser.add_argument("--env_path", type=str, default="scripts/envs/.envA_ch3_rnn")
    parser.add_argument("--sa_checkpoint", type=str, default="out/sa.pt")
    parser.add_argument("--selector_checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--force_full", action="store_true",
                        help="Select all slots, ignore tau stop.")
    parser.add_argument("--out_dir", type=str,
                        default="out/rnn_variant_eval")
    parser.add_argument("--no_global_init", action="store_true",
                        help="Force zero h0 regardless of RNN_SEL_GLOBAL_INIT.")
    args = parser.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    device = torch.device(config.device if config.use_gpu else "cpu")
    tau = config.ds_tau if args.tau is None else args.tau
    global_init_effective = False if args.no_global_init else config.rnn_sel_global_init

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    sa, _ = reconstruct_autoencoder(resolve_checkpoint_path(args.sa_checkpoint), config)
    sa.to(device).eval()

    # ── build the right model from checkpoint meta ────────────────────────────
    selector_path = resolve_checkpoint_path(args.selector_checkpoint)
    head_type = peek_head_type(selector_path)
    print(f"head_type from checkpoint: {head_type}")

    policy = build_policy(
        head_type=head_type,
        slot_dim=config.slot_dim,
        embed_dim=config.rnn_sel_embed_dim,
        hidden_dim=config.rnn_sel_hidden_dim,
        num_slots=config.num_slots,
        num_classes=config.ds_num_classes,
        use_stop_action=not config.rnn_sel_force_full_rollout,
    ).to(device)
    meta = load_selector(selector_path, policy, device)
    policy.eval()

    out_dir = Path(args.out_dir)
    correct_dir = out_dir / "correct"
    wrong_dir = out_dir / "wrong"
    correct_dir.mkdir(parents=True, exist_ok=True)
    wrong_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    stopped = 0
    sum_steps = 0.0
    sum_pmax = 0.0
    sum_p_true = 0.0
    correct_examples = 0
    wrong_examples = 0
    sample_index = 0
    confusion = torch.zeros(config.ds_num_classes, config.ds_num_classes, dtype=torch.long)

    max_samples = None if args.max_samples < 0 else args.max_samples
    loader = tqdm(dataloaders[args.split], desc=f"RNN-variant [{head_type}] {args.split}")

    slot_kw: dict = {"num_slots": config.num_slots, "slot_dim": config.slot_dim}
    if config.rnn_sel_sa_deterministic_slots:
        slot_kw["sa_deterministic"] = True
        slot_kw["sa_noise_seed"] = config.rnn_sel_sa_noise_seed

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[2].to(device)
        if max_samples is not None and total >= max_samples:
            break

        if max_samples is not None:
            remaining = max_samples - total
            images = images[:remaining]
            labels = labels[:remaining]

        slots = batch_slots(sa, images, device, **slot_kw)
        out = run_rnn_inference_batch(
            policy, slots, labels, tau, args.force_full, global_init_effective
        )
        preds = out["pred"]
        is_correct = preds == labels
        bsz = images.size(0)

        total += bsz
        correct += int(is_correct.sum().item())
        stopped += int(out["stopped"].sum().item())
        sum_steps += float(out["steps"].sum().item())
        sum_pmax += float(out["pmax"].sum().item())
        sum_p_true += float(out["p_true"].sum().item())

        for y, p in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
            confusion[y, p] += 1

        for row in range(bsz):
            need_correct = is_correct[row].item() and correct_examples < args.num_examples
            need_wrong = (not is_correct[row].item()) and wrong_examples < args.num_examples
            if not (need_correct or need_wrong):
                sample_index += 1
                continue

            one_image = images[row : row + 1]
            one_slots = slots[row : row + 1]
            label = int(labels[row].item())
            _, channels, height, width = one_image.shape
            recon_combined, slot_imgs, masks, _, _ = sa._decode_slots_to_recon(
                1, channels, height, width, one_slots, None
            )
            trace = run_rnn_inference_trace(
                policy, one_slots, label, tau, args.force_full, global_init_effective
            )
            final = trace[-1]
            kind = "correct" if need_correct else "wrong"
            rank = correct_examples if need_correct else wrong_examples
            target_dir = correct_dir if need_correct else wrong_dir
            out_path = target_dir / (
                f"{kind}_{rank:02d}_sample{sample_index}_true{label}_pred{final['pred']}.png"
            )
            save_trace_plot(
                image=one_image,
                recon_combined=recon_combined,
                slot_imgs=slot_imgs,
                masks=masks,
                trace=trace,
                label=label,
                sample_index=sample_index,
                split=args.split,
                tau=tau,
                global_init=global_init_effective,
                head_type=head_type,
                selector_checkpoint=selector_path,
                meta=meta,
                out_path=str(out_path),
            )
            if need_correct:
                correct_examples += 1
            else:
                wrong_examples += 1
            sample_index += 1

        loader.set_postfix(
            acc=f"{correct / max(total, 1):.4f}",
            head=head_type,
            correct_imgs=correct_examples,
            wrong_imgs=wrong_examples,
        )

        if (
            correct_examples >= args.num_examples
            and wrong_examples >= args.num_examples
            and max_samples is not None
            and total >= max_samples
        ):
            break

    metrics = {
        "split": args.split,
        "head_type": head_type,
        "total": total,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "wrong": total - correct,
        "tau": tau,
        "stop_rate": stopped / max(total, 1),
        "avg_steps": sum_steps / max(total, 1),
        "avg_pmax": sum_pmax / max(total, 1),
        "avg_p_true": sum_p_true / max(total, 1),
        "force_full": args.force_full,
        "global_init": global_init_effective,
        "selector_checkpoint": selector_path,
        "sa_checkpoint": args.sa_checkpoint,
        "correct_examples_saved": correct_examples,
        "wrong_examples_saved": wrong_examples,
        "confusion_matrix": confusion.tolist(),
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== RNN variant inference metrics  [head={head_type}] ===")
    for key in [
        "total", "accuracy", "correct", "wrong",
        "stop_rate", "avg_steps", "avg_pmax", "avg_p_true",
    ]:
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved metrics : {metrics_path}")
    print(f"Correct examples : {correct_dir}")
    print(f"Wrong examples   : {wrong_dir}")


if __name__ == "__main__":
    main()
