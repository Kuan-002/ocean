"""DeepSets classifier and independent subset evaluator (plan §4.1–4.2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _DeepSetsCore(nn.Module):
    """phi(slot) -> aggregate over masked slots -> rho -> logits."""

    def __init__(
        self,
        slot_dim: int,
        phi_hidden: int,
        rho_hidden: int,
        num_classes: int,
        aggregate: str,
    ):
        super().__init__()
        if aggregate not in ("sum", "mean"):
            raise ValueError("aggregate must be 'sum' or 'mean'")
        self.aggregate = aggregate
        self.phi = nn.Sequential(
            nn.Linear(slot_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, num_classes),
        )

    def forward(self, slots: Tensor, mask: Tensor) -> Tensor:
        """
        slots: [B, K, D]
        mask:  [B, K] — 1 = include in aggregate, 0 = dropped (plan: empty -> zero agg -> rho(0))
        """
        h = self.phi(slots)
        w = mask.unsqueeze(-1)
        summed = (h * w).sum(dim=1)
        if self.aggregate == "sum":
            agg = summed
        else:
            denom = mask.sum(dim=1).clamp(min=1.0)
            agg = summed / denom.unsqueeze(-1)
        return self.rho(agg)


class DeepSetsClassifier(nn.Module):
    """Full-set (or masked) classification; permutation-invariant over slot dimension."""

    def __init__(
        self,
        slot_dim: int,
        phi_hidden: int,
        rho_hidden: int,
        num_classes: int,
        aggregate: str = "sum",
    ):
        super().__init__()
        self._body = _DeepSetsCore(slot_dim, phi_hidden, rho_hidden, num_classes, aggregate)

    def forward(self, slots: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            mask = torch.ones(
                slots.shape[0], slots.shape[1], device=slots.device, dtype=slots.dtype
            )
        return self._body(slots, mask)

    @torch.inference_mode()
    def predict_proba(self, slots: Tensor, mask: Tensor | None = None) -> Tensor:
        logits = self.forward(slots, mask)
        return F.softmax(logits, dim=-1)


class DeepSetsSubsetEvaluator(nn.Module):
    """
    Same architecture as DeepSetsClassifier, **independent weights** (plan §4.2).
    Trained only via distillation to a frozen classifier; do not share parameters.
    """

    def __init__(
        self,
        slot_dim: int,
        phi_hidden: int,
        rho_hidden: int,
        num_classes: int,
        aggregate: str = "sum",
    ):
        super().__init__()
        self._body = _DeepSetsCore(slot_dim, phi_hidden, rho_hidden, num_classes, aggregate)

    def forward(self, slots: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            mask = torch.ones(
                slots.shape[0], slots.shape[1], device=slots.device, dtype=slots.dtype
            )
        return self._body(slots, mask)

    @torch.inference_mode()
    def predict_proba(self, slots: Tensor, mask: Tensor | None = None) -> Tensor:
        logits = self.forward(slots, mask)
        return F.softmax(logits, dim=-1)


class EnhancedDeepSetsClassifier(nn.Module):
    """
    Improved permutation-invariant classifier for CLEVR-Hans7.

    Over DeepSetsClassifier:
    - phi: 3-layer MLP with LayerNorm + GELU — richer per-slot representations
    - Aggregation: concat(sum_pool, max_pool) — sum captures additive evidence,
      max captures the single most informative slot (key for rule-based concepts)
    - rho: 3-layer MLP with Dropout — regularised multi-object combination
    """

    def __init__(
        self,
        slot_dim: int,
        phi_hidden: int,
        rho_hidden: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(slot_dim, phi_hidden),
            nn.LayerNorm(phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.LayerNorm(phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.LayerNorm(phi_hidden),
        )
        # sum_pool + max_pool → 2 * phi_hidden
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden * 2, rho_hidden),
            nn.LayerNorm(rho_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden, rho_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden // 2, num_classes),
        )

    def forward(self, slots: Tensor, mask: Tensor | None = None) -> Tensor:
        """slots: [B, K, D]  mask: [B, K] (1=include, 0=drop)"""
        B, K, _ = slots.shape
        if mask is None:
            mask = torch.ones(B, K, device=slots.device, dtype=slots.dtype)
        h = self.phi(slots)                             # [B, K, phi_hidden]
        w = mask.unsqueeze(-1)                          # [B, K, 1]
        sum_agg = (h * w).sum(dim=1)                    # [B, phi_hidden]
        h_masked = h + (1.0 - w) * (-1e9)
        max_agg = h_masked.max(dim=1).values            # [B, phi_hidden]
        agg = torch.cat([sum_agg, max_agg], dim=-1)     # [B, 2*phi_hidden]
        return self.rho(agg)

    @torch.inference_mode()
    def predict_proba(self, slots: Tensor, mask: Tensor | None = None) -> Tensor:
        return F.softmax(self.forward(slots, mask), dim=-1)


def kl_distill_loss(
    student_logits: Tensor, teacher_probs: Tensor, temperature: float
) -> Tensor:
    """KL( teacher || student ) with temperature, standard response distillation scaling."""
    t = temperature
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    target = teacher_probs.clamp(min=1e-8)
    target = target / target.sum(dim=-1, keepdim=True)
    return F.kl_div(s_logp, target, reduction="batchmean") * (t * t)
