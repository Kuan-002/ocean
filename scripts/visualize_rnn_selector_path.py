#!/usr/bin/env python3
"""Visualize one RNN selector full slot-order trajectory on a dataset image."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Windows environments can load Intel OpenMP via multiple binary packages
# (for example torch + numpy/matplotlib). This keeps the visualization script usable.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.classification.checkpoints import load_deepsets_checkpoint
from src.classification.deepsets import DeepSetsClassifier
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
    # White background makes the selected object stand out from the soft mask.
    return (((slot_img * mask + (1 - mask)) * 127.5) + 127.5).detach().cpu().clamp(0, 255).byte()


@torch.no_grad()
def run_full_order(
    policy: SlotSelectionPolicyGRU,
    clf: DeepSetsClassifier,
    slots: torch.Tensor,
    y_full: int,
    y_true: int,
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
        logits = policy.apply_action_mask(policy.forward_logits(h), selected_mask)
        if logits.size(1) > k:
            fill = torch.finfo(logits.dtype).min / 2
            logits = logits.clone()
            logits[:, k:] = fill

        action = int(logits.argmax(dim=-1).item())
        h_prev = h
        h, x = policy.step_hidden(slots[:, action], h)
        delta = h - h_prev
        E = policy.step_evidence(delta, x, E)
        selected_mask[0, action] = True

        rnn_logits, _ = policy.class_head(h, None, selected_mask, E=E)
        rnn_probs = F.softmax(rnn_logits, dim=-1)
        rnn_pred = int(rnn_logits.argmax(dim=-1).item())
        rnn_pmax = float(rnn_probs.max(dim=-1).values.item())
        rnn_p_y_full = float(rnn_probs[0, y_full].item())
        rnn_p_true = float(rnn_probs[0, y_true].item())

        ds_logits = clf(slots, selected_mask.float())
        ds_probs = F.softmax(ds_logits, dim=-1)
        ds_pred = int(ds_logits.argmax(dim=-1).item())
        ds_p_y_full = float(ds_probs[0, y_full].item())
        ds_p_true = float(ds_probs[0, y_true].item())

        trace.append(
            {
                "step": step + 1,
                "slot": action,
                "rnn_pred": rnn_pred,
                "rnn_pmax": rnn_pmax,
                "rnn_p_y_full": rnn_p_y_full,
                "rnn_p_true": rnn_p_true,
                "ds_pred": ds_pred,
                "ds_p_y_full": ds_p_y_full,
                "ds_p_true": ds_p_true,
            }
        )

    return trace


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RNN selector slot path")
    parser.add_argument("--env_path", type=str, default="scripts/envs/.envA_ch3_rnn")
    parser.add_argument("--sa_checkpoint", type=str, default="out/sa.pt")
    parser.add_argument("--cls_checkpoint", type=str, default="out/deepsets_classifier_best.pt")
    parser.add_argument("--selector_checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_path", type=str, default="out/rnn_selector_path.png")
    args = parser.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    device = torch.device(config.device if config.use_gpu else "cpu")

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

    clf = DeepSetsClassifier(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)
    clf.load_state_dict(load_deepsets_checkpoint(resolve_checkpoint_path(args.cls_checkpoint), device)[0])
    clf.eval()

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
    full_logits = clf(slots, None)
    full_probs = F.softmax(full_logits, dim=-1)
    y_full = int(full_logits.argmax(dim=-1).item())
    p_full = float(full_probs[0, y_full].item())
    p_true_full = float(full_probs[0, label].item())
    trace = run_full_order(policy, clf, slots, y_full, label, config.rnn_sel_global_init)

    k = config.num_slots
    cols = min(4, k)
    slot_rows = math.ceil(k / cols)
    fig = plt.figure(figsize=(4 * cols, 3.7 * (slot_rows + 1)))
    gs = fig.add_gridspec(slot_rows + 1, cols, height_ratios=[1.1] + [1.0] * slot_rows)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(to_uint8_image(image[0]).permute(1, 2, 0).numpy())
    ax0.set_title(
        f"Original\ntrue={label}, full_pred={y_full}\n"
        f"p(full_pred)={p_full:.3f}, p(true)={p_true_full:.3f}"
    )
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(to_uint8_image(recon_combined[0]).permute(1, 2, 0).numpy())
    ax1.set_title("SA reconstruction")
    ax1.axis("off")

    ax_text = fig.add_subplot(gs[0, 2:])
    ax_text.axis("off")
    meta_line = f"selector={Path(args.selector_checkpoint).name}"
    if meta.get("epoch") is not None:
        meta_line += f", epoch={meta['epoch']}"
    ax_text.text(
        0.0,
        0.5,
        f"split={args.split}, sample_index={args.sample_index}\n"
        f"global_init={config.rnn_sel_global_init}\n"
        f"{meta_line}\n"
        "Each slot title: step / slot / probabilities for full_pred and true class",
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
        ax.set_title(
            f"t={item['step']} slot={slot_id}\n"
            f"RNN p_full={item['rnn_p_y_full']:.3f} p_true={item['rnn_p_true']:.3f}\n"
            f"DS p_full={item['ds_p_y_full']:.3f} p_true={item['ds_p_true']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    for i in range(k, slot_rows * cols):
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
    print(f"full_pred={y_full}, p_full={p_full:.6f}, true={label}, p_true_full={p_true_full:.6f}")


if __name__ == "__main__":
    main()
