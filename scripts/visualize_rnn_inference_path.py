#!/usr/bin/env python3
"""Visualize the deploy-time RNN-only inference path for one dataset image."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid a common Windows torch/numpy/matplotlib OpenMP duplicate runtime crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.clevr_hans_dataset import setup_dataloaders
from src.config import Config
from src.explanation.rnn_selector import SlotSelectionPolicyGRU
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def load_selector(path: str, model: SlotSelectionPolicyGRU, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})


def to_uint8_image(image: torch.Tensor) -> torch.Tensor:
    return ((image.detach().cpu() * 127.5) + 127.5).clamp(0, 255).byte()


def slot_to_uint8(slot_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((slot_img * mask + (1 - mask)) * 127.5) + 127.5).detach().cpu().clamp(0, 255).byte()


def get_single_batch(loader, sample_index: int):
    seen = 0
    for batch in loader:
        images, obj_count, labels = batch
        bsz = images.size(0)
        if seen + bsz > sample_index:
            row = sample_index - seen
            return images[row : row + 1], obj_count[row : row + 1], labels[row : row + 1]
        seen += bsz
    raise IndexError(f"sample_index={sample_index} is out of range; dataset has {seen} samples")


@torch.no_grad()
def run_rnn_inference(
    policy: SlotSelectionPolicyGRU,
    slots: torch.Tensor,
    y_true: int,
    tau: float,
    force_full: bool,
    global_init: bool,
) -> list[dict]:
    bsz, k, _ = slots.shape
    if bsz != 1:
        raise ValueError("visualization expects a single image")

    device = slots.device
    h = policy.init_hidden(1, device)
    E = policy.init_evidence(1, device)
    selected_mask = torch.zeros(1, k, device=device, dtype=torch.bool)
    trace: list[dict] = []

    for step in range(k):
        action_logits = policy.apply_action_mask(policy.forward_logits(h), selected_mask)
        if action_logits.size(1) > k:
            # Deploy-time path never uses the legacy STOP action.
            fill = torch.finfo(action_logits.dtype).min / 2
            action_logits = action_logits.clone()
            action_logits[:, k:] = fill

        slot_id = int(action_logits.argmax(dim=-1).item())
        h_prev = h
        h, x = policy.step_hidden(slots[:, slot_id], h)
        delta = h - h_prev
        E = policy.step_evidence(delta, x, E)
        selected_mask[0, slot_id] = True

        cls_logits, _ = policy.class_head(h, None, selected_mask, E=E)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RNN-only inference path")
    parser.add_argument("--env_path", type=str, default="scripts/envs/.envA_ch3_rnn")
    parser.add_argument("--sa_checkpoint", type=str, default="out/sa.pt")
    parser.add_argument("--selector_checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--tau", type=float, default=None, help="RNN confidence stop threshold")
    parser.add_argument(
        "--force_full",
        action="store_true",
        help="Continue selecting until all slots are selected, ignoring the tau stop.",
    )
    parser.add_argument("--out_path", type=str, default="out/rnn_inference_path.png")
    args = parser.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    device = torch.device(config.device if config.use_gpu else "cpu")
    tau = config.ds_tau if args.tau is None else args.tau

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    images, _, labels = get_single_batch(dataloaders[args.split], args.sample_index)
    image = images.to(device)
    label = int(labels.item())

    sa, _ = reconstruct_autoencoder(resolve_checkpoint_path(args.sa_checkpoint), config)
    sa.to(device).eval()

    policy = SlotSelectionPolicyGRU(
        slot_dim=config.slot_dim,
        embed_dim=config.rnn_sel_embed_dim,
        hidden_dim=config.rnn_sel_hidden_dim,
        num_slots=config.num_slots,
        num_classes=config.ds_num_classes,
        use_stop_action=not config.rnn_sel_force_full_rollout,
    ).to(device)
    meta = load_selector(resolve_checkpoint_path(args.selector_checkpoint), policy, device)
    policy.eval()

    recon_combined, slot_imgs, masks, slots, _ = sa(image)
    trace = run_rnn_inference(
        policy, slots, label, tau, args.force_full, config.rnn_sel_global_init
    )

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
    ax0.set_title(f"Original\ntrue={label}, final_pred={final_pred}\npmax={final_pmax:.3f}, {stop_text}")
    ax0.axis("off")

    if cols > 1:
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(to_uint8_image(recon_combined[0]).permute(1, 2, 0).numpy())
        ax1.set_title("SA reconstruction")
        ax1.axis("off")

    if cols > 2:
        ax_text = fig.add_subplot(gs[0, 2:])
        ax_text.axis("off")
        meta_line = f"selector={Path(args.selector_checkpoint).name}"
        if meta.get("epoch") is not None:
            meta_line += f", epoch={meta['epoch']}"
        ax_text.text(
            0.0,
            0.5,
            f"RNN-only inference, no DeepSets loaded\n"
            f"split={args.split}, sample_index={args.sample_index}, tau={tau:.3f}\n"
            f"global_init={config.rnn_sel_global_init}\n"
            f"{meta_line}\n"
            "Each slot title: step / slot / RNN prediction / confidence",
            va="center",
            fontsize=11,
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

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {args.out_path}")
    print(f"Order: {' -> '.join(str(item['slot']) for item in trace)}")
    print(f"true={label}, final_pred={final_pred}, final_pmax={final_pmax:.6f}, stop_step={stop_step}")


if __name__ == "__main__":
    main()
