#!/usr/bin/env python3
"""
Test-set evaluation + per-class grid visualization for GlobalAttn RNN selector.

For each class c in [0..num_classes):
  <out_dir>/class_{c}_correct.png   — up to --max_per_class correctly predicted samples
  <out_dir>/class_{c}_incorrect.png — up to --max_per_class incorrectly predicted samples

Each sample row shows:
  [original image] [SA recon] [slot t=1] [slot t=2] [slot t=3] [slot t=4] [slot t=5]

Slot column with the first step where pmax >= tau gets a green highlight border (★).
Saves <out_dir>/metrics.json with overall + per-class accuracy and t* statistics.

Usage:
  python scripts/eval_visualize_test_global_attn.py \
    --env_path scripts/envs/.envA_ch7_global_attn \
    --sa_checkpoint out/sa_ch7_64_4/checkpoints/sa \
    --selector_checkpoint out/rnn_ch7_global_attn_<date>/rnn_selector_best.pt \
    --out_dir out/rnn_ch7_global_attn_<date>/test_viz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.classification.training import batch_slots
from src.clevr_hans_dataset import setup_dataloaders, ch7_classes
from src.config import Config
from src.explanation.rnn_selector_variants import build_policy
from src.explanation.rnn_selector_training import run_rnn_inference_batch
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


# ── helpers ──────────────────────────────────────────────────────────────────

def to_uint8(t: torch.Tensor) -> np.ndarray:
    """[-1,1] image tensor → (H,W,3) uint8 numpy."""
    return ((t.detach().cpu() * 127.5) + 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def slot_to_uint8(slot_img: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Slot image + mask → white-background uint8 numpy (H,W,3)."""
    composite = slot_img * mask + (1 - mask)
    return ((composite.detach().cpu() * 127.5) + 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def load_policy(path: str, policy: torch.nn.Module, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})


# ── per-sample step-by-step trace ────────────────────────────────────────────

@torch.no_grad()
def run_trace(policy, slots: torch.Tensor, y_true: int, tau: float) -> list[dict]:
    """Run greedy full-order inference on one image, return per-step info."""
    bsz, k, _ = slots.shape
    assert bsz == 1, "trace expects a single image"
    device = slots.device

    h = policy.init_hidden(1, device)
    E = policy.init_evidence(1, device)
    slot_cache = policy.init_slot_cache(1, device)
    selected_mask = torch.zeros(1, k, device=device, dtype=torch.bool)
    slot_embeds = policy.precompute_slot_embeds(slots)
    active = torch.ones(1, device=device, dtype=torch.bool)
    trace: list[dict] = []
    t_star_recorded = False

    for step in range(k):
        logits = policy.apply_action_mask(policy.forward_logits(h, slot_embeds), selected_mask)
        if logits.size(1) > k:
            logits = logits.clone()
            logits[:, k:] = torch.finfo(logits.dtype).min / 2

        slot_id = int(logits.argmax(dim=-1).item())
        h_prev = h
        h, x = policy.step_hidden(slots[:, slot_id], h)
        delta = h - h_prev
        slot_cache = policy.update_slot_cache(slot_cache, x, active)
        E = policy.step_evidence(delta, x, E, slot_cache=slot_cache)
        selected_mask[0, slot_id] = True

        cls_logits, _ = policy.class_head(h, slot_embeds, selected_mask, E=E)
        probs = F.softmax(cls_logits, dim=-1)
        pred = int(cls_logits.argmax(dim=-1).item())
        pmax = float(probs.max().item())
        p_true = float(probs[0, y_true].item())

        # t* = first step where pmax >= tau (matches training definition; not conditioned on correctness)
        is_tstar = (pmax >= tau) and not t_star_recorded
        if is_tstar:
            t_star_recorded = True

        trace.append({
            "step": step + 1,
            "slot": slot_id,
            "pred": pred,
            "pmax": pmax,
            "p_true": p_true,
            "is_tstar": is_tstar,
        })

    return trace


# ── per-class grid figure ─────────────────────────────────────────────────────

N_SLOT_COLS = 5   # how many slot columns to display (first N steps in selection order)
N_COLS = 2 + N_SLOT_COLS + 1  # original + recon + slots + confidence curve


def _border_ax(ax, color: str, lw: float = 3.5) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_visible(True)


def _plot_conf_curve(ax, trace: list[dict], tau: float, gt: int, pred: int) -> None:
    """Plot per-step p_true and pmax confidence curves with t* marked."""
    steps   = [t["step"]   for t in trace]
    p_true  = [t["p_true"] for t in trace]
    pmax    = [t["pmax"]   for t in trace]

    ax.plot(steps, p_true, color="steelblue",  lw=1.5, marker="o", ms=3, label="p(GT)")
    ax.plot(steps, pmax,   color="tomato",     lw=1.0, marker="s", ms=2, ls="--", label="pmax")
    ax.axhline(tau, color="gray", lw=0.8, ls=":")

    tstar_step = next((t["step"] for t in trace if t["is_tstar"]), None)
    if tstar_step is not None:
        ax.axvline(tstar_step, color="limegreen", lw=1.2, ls="--")
        ax.text(tstar_step + 0.1, tau + 0.03, f"t*={tstar_step}", fontsize=6, color="darkgreen")

    ax.set_xlim(0.5, len(steps) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(steps[::2])
    ax.tick_params(labelsize=5)
    ax.set_xlabel("step", fontsize=6)
    ax.set_ylabel("prob", fontsize=6)
    ax.legend(fontsize=5, loc="upper left", framealpha=0.6)
    ax.set_title("Confidence curve", fontsize=6)


def save_class_grid(
    samples: list[dict],
    class_id: int,
    kind: str,            # "correct" | "incorrect"
    class_name: str,
    tau: float,
    out_path: str,
) -> None:
    n = len(samples)
    if n == 0:
        return

    col_widths = [2.9] * (2 + N_SLOT_COLS) + [3.2]  # last col is the curve
    fig_w = sum(col_widths)
    fig, axes = plt.subplots(
        n, N_COLS,
        figsize=(fig_w, n * 3.0),
        gridspec_kw={"width_ratios": col_widths},
        squeeze=False,
    )

    marker = "✓" if kind == "correct" else "✗"
    fig.suptitle(
        f"Class {class_id}: {class_name}  |  {kind.upper()} predictions  {marker}"
        f"  (test split, τ={tau:.2f})",
        fontsize=11, fontweight="bold", y=1.002,
    )

    for row_i, s in enumerate(samples):
        trace        = s["trace"]
        image_np     = s["image_np"]
        recon_np     = s["recon_np"]
        slot_imgs_np = s["slot_imgs_np"]
        masks_np     = s["masks_np"]
        gt           = s["gt"]
        pred         = s["pred"]
        pmax_final   = s["pmax"]

        tstar_step_idx = next((i for i, t in enumerate(trace) if t["is_tstar"]), None)
        tstar_str = f"t*={tstar_step_idx+1}" if tstar_step_idx is not None else "t*=—"

        # --- Col 0: original image ---
        ax = axes[row_i, 0]
        ax.imshow(image_np)
        ax.set_title(f"GT={gt}  Pred={pred}  {marker}\n{tstar_str}  pmax={pmax_final:.2f}",
                     fontsize=7.5)
        ax.axis("off")
        _border_ax(ax, "green" if kind == "correct" else "crimson", 2.5)

        # --- Col 1: SA reconstruction ---
        ax = axes[row_i, 1]
        ax.imshow(recon_np)
        ax.set_title("SA recon", fontsize=7)
        ax.axis("off")

        # --- Cols 2 .. 2+N_SLOT_COLS-1: selected slots in order ---
        for col_off in range(N_SLOT_COLS):
            ax = axes[row_i, 2 + col_off]
            ax.axis("off")
            if col_off >= len(trace):
                continue
            item = trace[col_off]
            sid = item["slot"]
            slot_vis = slot_to_uint8(
                torch.tensor(slot_imgs_np[sid]).permute(2, 0, 1).float() / 127.5 - 1.0,
                torch.tensor(masks_np[sid]).permute(2, 0, 1).float() / 255.0,
            )
            ax.imshow(slot_vis)
            title = f"t={item['step']} s={sid}\np={item['p_true']:.2f}/{item['pmax']:.2f}"
            if item["is_tstar"]:
                title = f"★ {title}"
                _border_ax(ax, "limegreen", 3)
                ax.set_title(title, fontsize=6.5, color="darkgreen", fontweight="bold")
            else:
                ax.set_title(title, fontsize=6.5)

        # --- Last col: confidence curve ---
        ax = axes[row_i, N_COLS - 1]
        _plot_conf_curve(ax, trace, tau, gt, pred)

    plt.tight_layout(rect=[0, 0, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-set evaluation + per-class visualization for GlobalAttn RNN selector"
    )
    parser.add_argument("--env_path", default="scripts/envs/.envA_ch7_global_attn")
    parser.add_argument("--sa_checkpoint", default="out/sa_ch7_64_4/checkpoints/sa")
    parser.add_argument("--selector_checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max_per_class", type=int, default=10,
                        help="Max correct/incorrect examples to visualize per class")
    parser.add_argument("--out_dir", default="out/test_viz_global_attn")
    args = parser.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    device = torch.device(config.device if config.use_gpu else "cpu")
    tau = config.ds_tau
    num_classes = config.ds_num_classes

    # ── dataloaders ───────────────────────────────────────────────────────────
    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config, eval=True)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # ── models ────────────────────────────────────────────────────────────────
    sa, _ = reconstruct_autoencoder(
        resolve_checkpoint_path(args.sa_checkpoint), config
    )
    sa.to(device).eval()
    for p in sa.parameters():
        p.requires_grad = False

    head_type = os.getenv("RNN_SEL_HEAD_TYPE", "all_slots_attn").strip().lower()
    policy = build_policy(
        head_type=head_type,
        slot_dim=config.slot_dim,
        embed_dim=config.rnn_sel_embed_dim,
        hidden_dim=config.rnn_sel_hidden_dim,
        num_slots=config.num_slots,
        num_classes=num_classes,
        use_stop_action=not config.rnn_sel_force_full_rollout,
    ).to(device)
    sel_path = resolve_checkpoint_path(args.selector_checkpoint)
    meta = load_policy(sel_path, policy, device)
    policy.eval()
    print(f"Loaded {policy.__class__.__name__}  from  {sel_path}")
    if meta:
        print(f"Checkpoint meta: {meta}")

    slot_kw: dict = {"num_slots": config.num_slots, "slot_dim": config.slot_dim}
    if config.rnn_sel_sa_deterministic_slots:
        slot_kw["sa_deterministic"] = True
        slot_kw["sa_noise_seed"] = config.rnn_sel_sa_noise_seed

    # ── class names ───────────────────────────────────────────────────────────
    class_names: dict[int, str] = ch7_classes if num_classes == 7 else {
        i: f"Class {i}" for i in range(num_classes)
    }

    # ── per-class sample banks ────────────────────────────────────────────────
    correct_bank: dict[int, list] = {c: [] for c in range(num_classes)}
    wrong_bank:   dict[int, list] = {c: [] for c in range(num_classes)}

    # ── per-class statistics accumulators ─────────────────────────────────────
    cls_total    = {c: 0 for c in range(num_classes)}
    cls_correct  = {c: 0 for c in range(num_classes)}
    cls_tstar_sum   = {c: 0.0 for c in range(num_classes)}
    cls_tstar_found = {c: 0   for c in range(num_classes)}
    total_samples = 0
    total_correct = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def _need_more() -> bool:
        return any(
            len(correct_bank[c]) < args.max_per_class
            or len(wrong_bank[c]) < args.max_per_class
            for c in range(num_classes)
        )

    loader = tqdm(dataloaders[args.split], desc=f"[{args.split}]")

    for batch in loader:
        images = batch[0].to(device)
        labels = batch[2].to(device)
        bsz = images.size(0)

        slots = batch_slots(sa, images, device, **slot_kw)
        out = run_rnn_inference_batch(
            policy, slots, labels, tau,
            force_full=config.rnn_sel_force_full_rollout,
            global_init=config.rnn_sel_global_init,
        )
        preds = out["pred"]
        is_correct_batch = (preds == labels)

        # ── update global stats ───────────────────────────────────────────────
        total_samples += bsz
        total_correct += int(is_correct_batch.sum().item())
        t_star_b   = out["t_star"]
        t_star_found_b = out["t_star_found"]
        for i, (gt, pd) in enumerate(zip(labels.cpu().tolist(), preds.cpu().tolist())):
            confusion[gt, pd] += 1
            cls_total[gt]   += 1
            cls_correct[gt] += int(gt == pd)
            # Accumulate t* over ALL samples (use K when not found, matching training eval)
            t_val = float(t_star_b[i].item())
            cls_tstar_sum[gt]   += t_val
            cls_tstar_found[gt] += int(t_star_found_b[i].item())

        # ── collect per-class visualization samples ───────────────────────────
        if _need_more():
            for row in range(bsz):
                gt   = int(labels[row].item())
                pred = int(preds[row].item())
                correct = bool(is_correct_batch[row].item())

                need_c = correct and len(correct_bank[gt]) < args.max_per_class
                need_w = (not correct) and len(wrong_bank[gt]) < args.max_per_class
                if not (need_c or need_w):
                    continue

                # Run SA decoder to get slot images for this sample
                one_slots = slots[row : row + 1]   # [1, K, D]
                one_img   = images[row : row + 1]
                _, ch, ht, wd = one_img.shape
                recon_combined, slot_recons, slot_masks, _, _ = sa._decode_slots_to_recon(
                    1, ch, ht, wd, one_slots, None
                )

                # Per-step trace (greedy, full order)
                trace = run_trace(policy, one_slots, gt, tau)

                # Convert tensors to numpy once for storage
                image_np    = to_uint8(one_img[0])
                recon_np    = to_uint8(recon_combined[0])
                slot_imgs_np = [
                    to_uint8(slot_recons[0, k]) for k in range(config.num_slots)
                ]
                masks_np = [
                    (slot_masks[0, k].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    for k in range(config.num_slots)
                ]

                sample_record = {
                    "gt": gt, "pred": pred,
                    "pmax": float(out["pmax"][row].item()),
                    "steps": int(out["steps"][row].item()),
                    "trace": trace,
                    "image_np": image_np,
                    "recon_np": recon_np,
                    "slot_imgs_np": slot_imgs_np,
                    "masks_np": masks_np,
                }

                if need_c:
                    correct_bank[gt].append(sample_record)
                elif need_w:
                    wrong_bank[gt].append(sample_record)

        loader.set_postfix(
            acc=f"{total_correct/max(total_samples,1):.3f}",
            banks=f"{sum(len(v) for v in correct_bank.values())}c/"
                  f"{sum(len(v) for v in wrong_bank.values())}w",
        )

    # ── print metrics ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Split: {args.split}  |  Total: {total_samples}  |  Overall acc: {total_correct/max(total_samples,1):.4f}")
    print(f"{'='*60}")
    print(f"{'Class':<6} {'Name':<42} {'Acc':>6} {'N':>5} {'t*(avg)':>8} {'%found':>7}")
    print("-" * 76)
    for c in range(num_classes):
        acc_c = cls_correct[c] / max(cls_total[c], 1)
        name  = class_names.get(c, f"class_{c}").replace("\n", " ")[:40]
        # avg t*: use t_star value as-is (K when not found, matching training eval)
        avg_tstar = cls_tstar_sum[c] / max(cls_total[c], 1)
        pct_found = 100.0 * cls_tstar_found[c] / max(cls_total[c], 1)
        print(f"  {c:<4} {name:<42} {acc_c:>6.3f} {cls_total[c]:>5} {avg_tstar:>8.2f} {pct_found:>6.1f}%")
    print("-" * 75)

    print("\nConfusion matrix (rows=GT, cols=Pred):")
    header = "      " + "".join(f"{c:>5}" for c in range(num_classes))
    print(header)
    for r in range(num_classes):
        row_str = f"GT={r}  " + "".join(f"{confusion[r,c].item():>5}" for c in range(num_classes))
        print(row_str)

    # ── save metrics JSON ──────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "split": args.split,
        "selector_checkpoint": sel_path,
        "tau": tau,
        "total": total_samples,
        "overall_accuracy": total_correct / max(total_samples, 1),
        "per_class": {
            str(c): {
                "name": class_names.get(c, f"class_{c}").replace("\n", " "),
                "total": cls_total[c],
                "correct": cls_correct[c],
                "accuracy": cls_correct[c] / max(cls_total[c], 1),
                "avg_t_star": cls_tstar_sum[c] / max(cls_total[c], 1),
                "t_star_found": cls_tstar_found[c],
                "t_star_found_pct": 100.0 * cls_tstar_found[c] / max(cls_total[c], 1),
            }
            for c in range(num_classes)
        },
        "confusion_matrix": confusion.tolist(),
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved: {metrics_path}")

    # ── generate per-class visualization figures ──────────────────────────────
    print("\nGenerating per-class figures ...")
    for c in range(num_classes):
        cname = class_names.get(c, f"class_{c}").replace("\n", " ")

        save_class_grid(
            samples=correct_bank[c],
            class_id=c,
            kind="correct",
            class_name=cname,
            tau=tau,
            out_path=str(out_dir / f"class_{c}_correct.png"),
        )
        save_class_grid(
            samples=wrong_bank[c],
            class_id=c,
            kind="incorrect",
            class_name=cname,
            tau=tau,
            out_path=str(out_dir / f"class_{c}_incorrect.png"),
        )

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
