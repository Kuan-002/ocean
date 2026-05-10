#!/usr/bin/env python3
"""
Evaluate frozen SA + DeepSetsClassifier (4.1) + DeepSetsSubsetEvaluator (4.2) together.

Reports:
  - val accuracy (classifier on full slots)
  - mean KL(teacher_probs || student_probs) on val with stratified random masks (same as 4.2 training)
  - mean |p_full(y_hat) - p_student(y_hat)| for y_hat = classifier prediction (subset calibration gap)
  - per-sample table: write to --report_txt (TSV, labeled columns) with --max_samples rows
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from src.clevr_hans_dataset import setup_dataloaders
from src.classification.checkpoints import load_deepsets_checkpoint
from src.classification.deepsets import (
    DeepSetsClassifier,
    DeepSetsSubsetEvaluator,
    kl_distill_loss,
)
from src.classification.subset_sampling import stratified_subset_masks
from src.classification.training import batch_slots, eval_classifier_accuracy
from src.config import Config
from src.mds_dataset import setup_dataloaders_mds
from src.utils import reconstruct_autoencoder, resolve_checkpoint_path, seed_all


def _mask_fixed_k(
    k: int, row_seed: int, num_slots: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Deterministic k-hot mask: one row [1,K]."""
    g = torch.Generator()
    g.manual_seed(row_seed)
    perm = torch.randperm(num_slots, generator=g)
    m = torch.zeros(1, num_slots, device=device, dtype=dtype)
    m[0, perm[:k].to(device)] = 1.0
    return m


def _stratified_mask_deterministic(
    row_seed: int, num_slots: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    """Stratified single-sample mask (same recipe as training), fixed RNG for reproducible reports."""
    g = torch.Generator()
    g.manual_seed(row_seed)
    k = int(torch.randint(1, num_slots + 1, (1,), generator=g).item())
    perm = torch.randperm(num_slots, generator=g)
    m = torch.zeros(1, num_slots, device=device, dtype=dtype)
    m[0, perm[:k].to(device)] = 1.0
    return m, k


REPORT_LEGEND = """
================================================================================
COLUMN GLOSSARY (per row = one validation image)
================================================================================
sample_index          Running index in this report (0 .. N-1).

y_true                Ground-truth class id from dataset.

y_hat                 Predicted class from DeepSetsClassifier (4.1) on FULL slots S.

is_correct            1 if y_hat == y_true else 0.

num_slots_K           Number of slots K from config (SA_NUM_SLOTS).

p_full_y_hat          Classifier softmax probability of y_hat using ALL slots (full set).

p_sub_y_hat_k1        SubsetEvaluator (4.2) softmax P(y_hat | S') with |S'|=k1_mask_size
                      (slots chosen deterministically from row seed; see k1_mask_size).

k1_mask_size          Always 1 for column p_sub_y_hat_k1.

p_sub_y_hat_k_half    Same for subset size k_half = max(1, floor(K/2)).

k_half_mask_size      Subset size used for p_sub_y_hat_k_half.

p_sub_y_hat_k_prev    Same for subset size K-1 (one slot dropped).

k_prev_mask_size      Always K-1 when K>1; if K==1 equals 1.

p_sub_y_hat_strat     Stratified random subset (same distribution as 4.2 training),
                      deterministic per row via row_seed.

k_stratified          The k ~ Uniform{1..K} drawn for p_sub_y_hat_strat.

row_seed              Hash used so masks for this row are reproducible.
                      row_seed = base_seed + sample_index * 100003.

Notes:
  - Teacher scores on full S are from 4.1; subset scores are from 4.2 (student), frozen.
  - "y_hat" is always the full-set prediction; subset columns show faithfulness under missing slots.
================================================================================
""".strip()


@torch.no_grad()
def eval_subset_metrics(
    sa,
    clf: DeepSetsClassifier,
    ev: DeepSetsSubsetEvaluator,
    loader,
    device: torch.device,
    num_slots: int,
    temperature: float,
) -> tuple[float, float, int]:
    clf.eval()
    ev.eval()
    total_kl = 0.0
    total_pgap = 0.0
    total_n = 0
    n_batches = 0
    for images, _, _ in loader:
        b = images.size(0)
        slots = batch_slots(sa, images, device)
        y_hat = clf(slots, None).argmax(dim=-1)

        t_logits = clf(slots, None)
        t_probs = F.softmax(t_logits / temperature, dim=-1)
        mask = stratified_subset_masks(b, num_slots, device)
        s_logits = ev(slots, mask)
        kl = kl_distill_loss(s_logits, t_probs, temperature).item()

        s_probs = F.softmax(s_logits, dim=-1)
        p_teacher = t_probs[torch.arange(b, device=device), y_hat]
        p_student = s_probs[torch.arange(b, device=device), y_hat]
        pgap = (p_teacher - p_student).abs().mean().item()

        total_kl += kl * b
        total_pgap += pgap * b
        total_n += b
        n_batches += 1

    return total_kl / max(total_n, 1), total_pgap / max(total_n, 1), n_batches


def _write_sample_table_txt(
    f,
    sa,
    clf: DeepSetsClassifier,
    ev: DeepSetsSubsetEvaluator,
    val_loader,
    device: torch.device,
    num_slots: int,
    max_samples: int,
    base_seed: int,
) -> int:
    """Append TSV table; return number of rows written."""
    clf.eval()
    ev.eval()
    kh = max(1, num_slots // 2)
    kprev = max(1, num_slots - 1)
    dt = torch.float32

    header = (
        "sample_index\ty_true\ty_hat\tis_correct\tnum_slots_K\t"
        "p_full_y_hat\t"
        "p_sub_y_hat_k1\tk1_mask_size\t"
        f"p_sub_y_hat_k_half\tk_half_mask_size\t"
        f"p_sub_y_hat_k_prev\tk_prev_mask_size\t"
        "p_sub_y_hat_strat\tk_stratified\trow_seed\n"
    )
    f.write(header)

    n_written = 0
    sample_index = 0
    for images, _, labels in val_loader:
        if n_written >= max_samples:
            break
        slots = batch_slots(sa, images, device)
        labels = labels.to(device)
        b = slots.size(0)
        logits_full = clf(slots, None)
        y_hat = logits_full.argmax(dim=-1)
        p_full = F.softmax(logits_full, dim=-1)

        for i in range(b):
            if n_written >= max_samples:
                break
            row_seed = base_seed + sample_index * 100_003
            s = slots[i : i + 1]
            y_true = labels[i].item()
            pred = y_hat[i].item()
            correct = 1 if pred == y_true else 0
            pf = p_full[i, pred].item()

            m1 = _mask_fixed_k(1, row_seed + 1, num_slots, device, dt)
            mkh = _mask_fixed_k(kh, row_seed + 2, num_slots, device, dt)
            mkp = _mask_fixed_k(kprev, row_seed + 3, num_slots, device, dt)
            mstrat, kstrat = _stratified_mask_deterministic(
                row_seed + 4, num_slots, device, dt
            )

            def psub(mask: torch.Tensor) -> float:
                lp = ev(s, mask)
                pr = F.softmax(lp, dim=-1)
                return pr[0, pred].item()

            line = (
                f"{sample_index}\t{y_true}\t{pred}\t{correct}\t{num_slots}\t"
                f"{pf:.6f}\t"
                f"{psub(m1):.6f}\t1\t"
                f"{psub(mkh):.6f}\t{kh}\t"
                f"{psub(mkp):.6f}\t{kprev}\t"
                f"{psub(mstrat):.6f}\t{kstrat}\t{row_seed}\n"
            )
            f.write(line)
            n_written += 1
            sample_index += 1

    return n_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_path", type=str, default=".env")
    p.add_argument("--sa_checkpoint", type=str, required=True)
    p.add_argument("--cls_checkpoint", type=str, required=True)
    p.add_argument("--eval_checkpoint", type=str, required=True)
    p.add_argument(
        "--report_txt",
        type=str,
        default="",
        help="Write full report (metrics + glossary + TSV table) to this path. "
        "Example: ./out/deepsets_eval_report.txt",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Max validation rows in the TSV table (traversal order: val DataLoader).",
    )
    p.add_argument(
        "--table_seed",
        type=int,
        default=-1,
        help="Base seed for deterministic per-row subset masks; default = config.seed.",
    )
    args = p.parse_args()

    config = Config(args.env_path, None)
    seed_all(config.seed, config.deterministic)
    table_seed = config.seed if args.table_seed < 0 else args.table_seed

    if config.dataset == "ch":
        dataloaders = setup_dataloaders(config)
    elif config.dataset == "mds":
        dataloaders = setup_dataloaders_mds(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    device = torch.device(config.device if config.use_gpu else "cpu")

    sa_ckpt = resolve_checkpoint_path(args.sa_checkpoint)
    cls_ckpt = resolve_checkpoint_path(args.cls_checkpoint)
    ev_ckpt = resolve_checkpoint_path(args.eval_checkpoint)

    lines_stdout = [
        f"SA:         {sa_ckpt}",
        f"Classifier: {cls_ckpt}",
        f"Evaluator:  {ev_ckpt}",
    ]

    sa, _ = reconstruct_autoencoder(sa_ckpt, config)
    sa.to(device)
    sa.eval()
    for par in sa.parameters():
        par.requires_grad = False

    clf = DeepSetsClassifier(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)
    clf.load_state_dict(load_deepsets_checkpoint(cls_ckpt, map_location=device)[0])
    clf.eval()

    ev = DeepSetsSubsetEvaluator(
        slot_dim=config.slot_dim,
        phi_hidden=config.ds_phi_hidden,
        rho_hidden=config.ds_rho_hidden,
        num_classes=config.ds_num_classes,
        aggregate=config.ds_aggregate,
    ).to(device)
    ev_state, ev_meta = load_deepsets_checkpoint(ev_ckpt, map_location=device)
    ev.load_state_dict(ev_state)
    ev.eval()
    if ev_meta:
        lines_stdout.append(f"Evaluator checkpoint meta: {ev_meta}")

    T = config.ds_distill_temperature
    acc = eval_classifier_accuracy(sa, clf, dataloaders["val"], device)
    mean_kl, mean_pgap, nb = eval_subset_metrics(
        sa, clf, ev, dataloaders["val"], device, config.num_slots, T
    )

    summary = f"""
=== Aggregated metrics (validation) ===
Val classifier accuracy (full slots):     {acc:.6f}
Distillation temperature T:                 {T}
Mean KL(teacher || student) [per-sample]:  {mean_kl:.6f}
  (teacher = 4.1 full-slot softmax; student = 4.2 on stratified random mask; same as training)
Mean |p_teacher(y_hat) - p_student(y_hat)|: {mean_pgap:.6f}
  (y_hat = 4.1 prediction on full slots; mask = one random stratified mask per image in metric loop)
Val batches in KL loop:                    {nb}

num_slots_K: {config.num_slots}
num_classes: {config.ds_num_classes}
ds_aggregate: {config.ds_aggregate}
table_seed (per-row masks): {table_seed}
max_samples in table:      {args.max_samples}
""".strip()

    for ln in lines_stdout:
        print(ln)
    print(summary)

    if args.report_txt:
        path = Path(args.report_txt).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with path.open("w", encoding="utf-8") as f:
            f.write("# DeepSets evaluation report (4.1 classifier + 4.2 subset evaluator)\n")
            f.write(f"# generated_utc: {iso}\n\n")
            f.write("[CHECKPOINTS]\n")
            f.write(f"sa_checkpoint={sa_ckpt}\n")
            f.write(f"cls_checkpoint={cls_ckpt}\n")
            f.write(f"eval_checkpoint={ev_ckpt}\n")
            if ev_meta:
                f.write(f"eval_meta={ev_meta}\n")
            f.write(f"env_path={args.env_path}\n\n")
            f.write("[SUMMARY]\n")
            f.write(summary + "\n\n")
            f.write(REPORT_LEGEND + "\n\n")
            f.write("[TSV_TABLE]\n")
            f.write(
                "# Tab-separated. First row after this comment block is the header line.\n\n"
            )
            n = _write_sample_table_txt(
                f,
                sa,
                clf,
                ev,
                dataloaders["val"],
                device,
                config.num_slots,
                args.max_samples,
                table_seed,
            )
            f.write(f"\n# END rows_written={n}\n")
        print(f"\nWrote report: {path} ({n} sample rows)")


if __name__ == "__main__":
    main()
